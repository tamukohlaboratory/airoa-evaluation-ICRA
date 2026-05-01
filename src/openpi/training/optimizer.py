import dataclasses
import functools
import math
import re
from typing import NamedTuple
from typing import Protocol
from typing import runtime_checkable

import jax
import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


@runtime_checkable
class LRScheduleConfig(Protocol):
    def create(self) -> optax.Schedule: ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@runtime_checkable
class OptimizerConfig(Protocol):
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


# Default routing rules for Muon. These are intentionally conservative: embeddings, classifier heads,
# positional embeddings, and class tokens stay on the auxiliary AdamW branch.
_DEFAULT_AUX_ADAM_PATH_PATTERNS = (
    r"(^|/)(input_embedding|pos_embedding|posembed_input|embedding|embedder|head|lm_head|cls)(/|$)",
)
# Flax convolution kernels usually live under names like conv, conv1, StdConv, or ConvTranspose.
_DEFAULT_CONV_PATH_PATTERNS = (
    r"(^|/)([^/]*conv[^/]*|[^/]*stdconv[^/]*)(/|$)",
)


class ScaleByMuonState(NamedTuple):
    momentum: optax.Updates


@functools.lru_cache(maxsize=None)
def _compile_patterns(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p) for p in patterns)


def _unwrap_leaf(x):
    # Handles NNX variable wrappers if they are ever presented directly.
    return x.value if hasattr(x, "value") else x


def _tree_path_to_str(path: tuple[object, ...]) -> str:
    parts = []
    for entry in path:
        if hasattr(entry, "key"):
            part = entry.key
        elif hasattr(entry, "name"):
            part = entry.name
        elif hasattr(entry, "idx"):
            part = entry.idx
        else:
            part = str(entry)
        parts.append(str(part))
    if parts and parts[-1] == "value":
        parts.pop()
    return "/".join(parts)


def _matches_any(path: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(p.search(path) is not None for p in patterns)


def _scaled_lr(lr: optax.ScalarOrSchedule, scale: float) -> optax.ScalarOrSchedule:
    if callable(lr):
        return lambda step: lr(step) * scale
    return lr * scale


def _zeropower_via_newton_schulz5(
    matrix: at.Array, *, steps: int, eps: float
) -> at.Array:
    """Approximate semi-orthogonalization with the Muon Newton-Schulz iteration.

    This mirrors the public Muon reference implementation: normalize by the Frobenius norm,
    use the tuned quintic coefficients, operate in bfloat16 for efficiency, and transpose tall
    matrices before applying the iteration.
    """
    if matrix.ndim < 2:
        return matrix

    a, b, c = (3.4445, -4.7750, 2.0315)
    rows, cols = matrix.shape[-2], matrix.shape[-1]

    x = matrix.astype(jnp.bfloat16)
    if rows > cols:
        x = jnp.swapaxes(x, -2, -1)

    x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)

    def body(_, current):
        a_mat = current @ jnp.swapaxes(current, -2, -1)
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        return a * current + b_mat @ current

    x = jax.lax.fori_loop(0, steps, body, x)

    if rows > cols:
        x = jnp.swapaxes(x, -2, -1)
    return x.astype(matrix.dtype)


def _reshape_conv_kernel_to_matrix(kernel: at.Array) -> tuple[at.Array, tuple[int, ...]]:
    # Flax convolution kernels are typically (... spatial ..., in_channels, out_channels).
    # Muon expects a single 2D matrix for hidden convolutions, so we treat the output channel
    # axis as rows and flatten the remaining axes into columns.
    moved = jnp.moveaxis(kernel, -1, 0)
    return moved.reshape((kernel.shape[-1], -1)), moved.shape


def _restore_conv_kernel_from_matrix(matrix: at.Array, moved_shape: tuple[int, ...]) -> at.Array:
    return jnp.moveaxis(matrix.reshape(moved_shape), 0, -1)


def _orthogonalize_update(
    update: at.Array,
    *,
    path: str,
    ns_steps: int,
    ns_eps: float,
    conv_path_patterns: tuple[re.Pattern[str], ...],
) -> at.Array:
    if update.ndim < 2:
        return update

    lower_path = path.lower()
    if _matches_any(lower_path, conv_path_patterns) and update.ndim >= 3:
        matrix, moved_shape = _reshape_conv_kernel_to_matrix(update)
        rows, cols = matrix.shape[-2], matrix.shape[-1]
        orthogonal = _zeropower_via_newton_schulz5(matrix, steps=ns_steps, eps=ns_eps)
        orthogonal = orthogonal * math.sqrt(max(1.0, rows / cols))
        return _restore_conv_kernel_from_matrix(orthogonal, moved_shape)

    rows, cols = update.shape[-2], update.shape[-1]
    orthogonal = _zeropower_via_newton_schulz5(update, steps=ns_steps, eps=ns_eps)
    orthogonal = orthogonal * math.sqrt(max(1.0, rows / cols))
    return orthogonal


def muon_param_labels(
    params: at.PyTree,
    *,
    aux_adam_path_patterns: tuple[str, ...] = _DEFAULT_AUX_ADAM_PATH_PATTERNS,
) -> at.PyTree:
    """Route hidden weight matrices to Muon and everything else to AdamW.

    Leaves with fewer than 2 dimensions always go to AdamW. For higher-rank tensors we still route
    them to Muon by default, then decide how to reshape them inside the Muon transform.
    """
    compiled_aux_patterns = _compile_patterns(tuple(p.lower() for p in aux_adam_path_patterns))
    flat_params, treedef = jax.tree_util.tree_flatten_with_path(params)

    labels = []
    for path, leaf in flat_params:
        value = _unwrap_leaf(leaf)
        ndim = getattr(value, "ndim", None)
        path_str = _tree_path_to_str(path).lower()

        if ndim is None or ndim < 2 or _matches_any(path_str, compiled_aux_patterns):
            labels.append("adam")
        else:
            labels.append("muon")

    return jax.tree_util.tree_unflatten(treedef, labels)


def scale_by_muon(
    *,
    beta: float = 0.95,
    ns_steps: int = 5,
    ns_eps: float = 1e-7,
    nesterov: bool = True,
    conv_path_patterns: tuple[str, ...] = _DEFAULT_CONV_PATH_PATTERNS,
) -> optax.GradientTransformation:
    """Muon's momentum + Newton-Schulz orthogonalization transform.

    The incoming updates are interpreted as raw gradients. This transform first applies the Muon
    momentum rule, then approximately orthogonalizes the resulting matrix update. The learning rate
    scaling and decoupled weight decay are intentionally left to later transforms in the chain.
    """
    compiled_conv_patterns = _compile_patterns(tuple(p.lower() for p in conv_path_patterns))

    def init_fn(params):
        return ScaleByMuonState(momentum=jax.tree_util.tree_map(jnp.zeros_like, params))

    def update_fn(updates, state, params=None):  # noqa: ARG001
        del params
        flat_updates, treedef = jax.tree_util.tree_flatten_with_path(updates)
        flat_momentum, momentum_treedef = jax.tree_util.tree_flatten(state.momentum)
        if treedef != momentum_treedef:
            raise ValueError("Momentum state tree does not match the updates tree.")

        new_updates = []
        new_momentum = []
        for (path, grad), momentum in zip(flat_updates, flat_momentum, strict=True):
            grad = _unwrap_leaf(grad)
            momentum = _unwrap_leaf(momentum)
            new_m = momentum + (1.0 - beta) * (grad - momentum)
            update = grad + beta * (new_m - grad) if nesterov else new_m
            new_u = _orthogonalize_update(
                update,
                path=_tree_path_to_str(path),
                ns_steps=ns_steps,
                ns_eps=ns_eps,
                conv_path_patterns=compiled_conv_patterns,
            )
            new_updates.append(new_u)
            new_momentum.append(new_m)

        return jax.tree_util.tree_unflatten(treedef, new_updates), ScaleByMuonState(
            momentum=jax.tree_util.tree_unflatten(treedef, new_momentum)
        )

    return optax.GradientTransformation(init_fn, update_fn)


@dataclasses.dataclass(frozen=True)
class Muon(OptimizerConfig):
    """Muon optimizer with an auxiliary AdamW branch.

    Hidden weight matrices are optimized with Muon; embeddings, classifier heads, positional embeddings,
    class-token parameters, and all leaves with ndim < 2 are optimized with AdamW.
    """

    beta: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    ns_eps: float = 1e-7
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    # Auxiliary AdamW branch for embeddings / heads / biases / scales.
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    adam_eps: float = 1e-8
    adam_lr_scale: float = 1.0
    adam_weight_decay: float | None = None

    # Path-based routing knobs.
    aux_adam_path_patterns: tuple[str, ...] = _DEFAULT_AUX_ADAM_PATH_PATTERNS
    conv_path_patterns: tuple[str, ...] = _DEFAULT_CONV_PATH_PATTERNS

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        aux_weight_decay = self.weight_decay if self.adam_weight_decay is None else self.adam_weight_decay

        muon_tx = optax.chain(
            scale_by_muon(
                beta=self.beta,
                ns_steps=self.ns_steps,
                ns_eps=self.ns_eps,
                nesterov=self.nesterov,
                conv_path_patterns=self.conv_path_patterns,
            ),
            optax.add_decayed_weights(self.weight_decay, mask=weight_decay_mask),
            optax.scale_by_schedule(lr),
            optax.scale(-1.0),
        )
        adam_tx = optax.adamw(
            _scaled_lr(lr, self.adam_lr_scale),
            b1=self.adam_b1,
            b2=self.adam_b2,
            eps=self.adam_eps,
            weight_decay=aux_weight_decay,
            mask=weight_decay_mask,
        )

        partition = getattr(optax, "partition", optax.multi_transform)
        tx = partition(
            {"muon": muon_tx, "adam": adam_tx},
            functools.partial(muon_param_labels, aux_adam_path_patterns=self.aux_adam_path_patterns),
        )
        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)

