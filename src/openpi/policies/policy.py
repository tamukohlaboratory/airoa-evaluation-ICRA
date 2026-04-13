from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from policy_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

_TRUE_STRINGS = {"1", "true", "yes", "on", "y"}
_FALSE_STRINGS = {"0", "false", "no", "off", "n"}
_MBR_HUBER_DELTA = 0.5
_MBR_TRAJECTORY_DELTA_WEIGHT = 0.25


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Expected a scalar array, got shape={value.shape}")
        return value.reshape(-1)[0].item()
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected a scalar tensor, got shape={tuple(value.shape)}")
        return value.detach().cpu().reshape(-1)[0].item()
    return value


def _parse_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None

    value = _coerce_scalar(value)
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "auto", "default", "none"}:
            return None
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def _parse_optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None

    value = _coerce_scalar(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "auto", "default", "none"}:
            return None
        value = int(text)

    value = int(value)
    if value <= 0:
        raise ValueError(f"Expected a positive integer, got {value!r}")
    return value


def _normalize_action_name(name: Any) -> str:
    return str(name).strip().lower().replace("-", "_")


def _extract_action_names(metadata: dict[str, Any]) -> tuple[str, ...]:
    action_names = metadata.get("action_names")
    if not isinstance(action_names, Sequence) or isinstance(action_names, (str, bytes)):
        return ()
    return tuple(_normalize_action_name(name) for name in action_names)


def _prepare_mbr_actions(action_candidates: Sequence[np.ndarray]) -> np.ndarray:
    if not action_candidates:
        raise ValueError("action_candidates must not be empty")

    prepared = []
    for candidate in action_candidates:
        action = np.asarray(candidate, dtype=np.float32)
        if action.ndim == 0:
            raise ValueError("action candidates must have at least one dimension")
        if action.ndim == 1:
            action = action[None, :]
        elif action.ndim > 2:
            action = action.reshape(action.shape[0], -1)
        prepared.append(action)

    stacked = np.stack(prepared, axis=0)
    return stacked.astype(np.float64, copy=False)


def _build_mbr_dim_weights(action_dim: int, action_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    dim_weights = np.ones(action_dim, dtype=np.float64)
    angular_dims = np.zeros(action_dim, dtype=bool)

    for idx, name in enumerate(action_names[:action_dim]):
        if "gripper" in name or "hand_motor" in name:
            dim_weights[idx] *= 2.0
        elif name.startswith("base_"):
            dim_weights[idx] *= 1.25

        if name == "base_theta" or any(token in name for token in ("roll_joint", "pan_joint", "theta", "yaw")):
            angular_dims[idx] = True
            if name == "base_theta" or "yaw" in name or "theta" in name:
                dim_weights[idx] *= 1.25

    return dim_weights, angular_dims


def _make_mbr_time_weights(action_horizon: int) -> np.ndarray:
    weights = 1.0 / (1.0 + np.arange(action_horizon, dtype=np.float64))
    return weights / np.mean(weights)


def _wrap_angle_difference(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _huber_loss(delta: np.ndarray, huber_delta: float = _MBR_HUBER_DELTA) -> np.ndarray:
    abs_delta = np.abs(delta)
    quadratic = np.minimum(abs_delta, huber_delta)
    linear = abs_delta - quadratic
    return 0.5 * quadratic**2 + huber_delta * linear


def _estimate_mbr_risks(
    action_candidates: Sequence[np.ndarray],
    reference_candidates: Sequence[np.ndarray] | None = None,
    *,
    action_names: Sequence[str] = (),
) -> np.ndarray:
    decisions = _prepare_mbr_actions(action_candidates)
    references = _prepare_mbr_actions(reference_candidates if reference_candidates is not None else action_candidates)

    if decisions.shape[1:] != references.shape[1:]:
        raise ValueError(
            "Decision and reference candidates must have matching action shapes, "
            f"got {decisions.shape[1:]} vs {references.shape[1:]}"
        )

    action_horizon = decisions.shape[1]
    action_dim = decisions.shape[2]
    time_weights = _make_mbr_time_weights(action_horizon)
    dim_weights, angular_dims = _build_mbr_dim_weights(action_dim, action_names)

    pairwise_delta = decisions[:, None, :, :] - references[None, :, :, :]
    if np.any(angular_dims):
        pairwise_delta[..., angular_dims] = _wrap_angle_difference(pairwise_delta[..., angular_dims])

    pairwise_loss = _huber_loss(pairwise_delta)
    pairwise_loss *= time_weights[None, None, :, None]
    pairwise_loss *= dim_weights[None, None, None, :]
    risks = pairwise_loss.mean(axis=(-1, -2))

    if action_horizon > 1:
        decision_deltas = decisions[:, None, 1:, :] - decisions[:, None, :-1, :]
        reference_deltas = references[None, :, 1:, :] - references[None, :, :-1, :]
        trajectory_delta = decision_deltas - reference_deltas
        if np.any(angular_dims):
            trajectory_delta[..., angular_dims] = _wrap_angle_difference(trajectory_delta[..., angular_dims])

        trajectory_loss = _huber_loss(trajectory_delta)
        trajectory_loss *= time_weights[None, None, 1:, None]
        trajectory_loss *= dim_weights[None, None, None, :]
        risks += _MBR_TRAJECTORY_DELTA_WEIGHT * trajectory_loss.mean(axis=(-1, -2))

    return risks.mean(axis=1)


def _select_mbr_candidate(
    action_candidates: Sequence[np.ndarray],
    reference_candidates: Sequence[np.ndarray] | None = None,
    *,
    action_names: Sequence[str] = (),
) -> tuple[int, np.ndarray]:
    risks = _estimate_mbr_risks(action_candidates, reference_candidates, action_names=action_names)
    best_idx = int(np.argmin(risks))
    return best_idx, risks


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        use_mbr: bool = False,
        mbr_num_candidates: int = 8,
        mbr_num_reference_candidates: int | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            use_mbr: Whether to enable Minimum Bayes-Risk decoding by default during inference.
            mbr_num_candidates: Number of decision candidates to sample when MBR is enabled.
            mbr_num_reference_candidates: Number of reference action chunks to sample when MBR is enabled.
                If omitted, defaults to `mbr_num_candidates`.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._action_names = _extract_action_names(self._metadata)
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._use_mbr = bool(use_mbr)
        if mbr_num_candidates < 1:
            raise ValueError(f"mbr_num_candidates must be >= 1, got {mbr_num_candidates}")
        self._mbr_num_candidates = int(mbr_num_candidates)
        if mbr_num_reference_candidates is not None and mbr_num_reference_candidates < 1:
            raise ValueError(
                "mbr_num_reference_candidates must be >= 1, "
                f"got {mbr_num_reference_candidates}"
            )
        self._mbr_num_reference_candidates = (
            int(mbr_num_reference_candidates) if mbr_num_reference_candidates is not None else None
        )

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    def _resolve_mbr_settings(self, obs: dict[str, Any]) -> tuple[bool, int, int]:
        use_mbr = self._use_mbr
        if "use_mbr" in obs:
            override = _parse_optional_bool(obs.pop("use_mbr"))
            if override is not None:
                use_mbr = override

        mbr_num_candidates = self._mbr_num_candidates
        has_explicit_reference_count = self._mbr_num_reference_candidates is not None
        mbr_num_reference_candidates = (
            self._mbr_num_reference_candidates if has_explicit_reference_count else mbr_num_candidates
        )
        if "mbr_num_candidates" in obs:
            override = _parse_optional_positive_int(obs.pop("mbr_num_candidates"))
            if override is not None:
                mbr_num_candidates = override
                if not has_explicit_reference_count:
                    mbr_num_reference_candidates = override

        if "mbr_num_reference_candidates" in obs:
            override = _parse_optional_positive_int(obs.pop("mbr_num_reference_candidates"))
            if override is not None:
                mbr_num_reference_candidates = override

        return use_mbr, mbr_num_candidates, mbr_num_reference_candidates

    def _convert_inputs_to_model_arrays(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self._is_pytorch_model:
            return jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)
        return jax.tree.map(jnp.asarray, inputs)

    def _sample_once(
        self,
        sample_rng_or_pytorch_device: at.KeyArrayLike | str,
        model_inputs: dict[str, Any],
        *,
        noise: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        batched_inputs = jax.tree.map(lambda x: x[None, ...], model_inputs)

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            if self._is_pytorch_model:
                model_noise = torch.from_numpy(np.asarray(noise)).to(self._pytorch_device)
            else:
                model_noise = jnp.asarray(noise)

            if model_noise.ndim == 2:
                model_noise = model_noise[None, ...]
            sample_kwargs["noise"] = model_noise

        observation = _model.Observation.from_dict(batched_inputs)
        outputs = {
            "state": batched_inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }

        if self._is_pytorch_model:
            return jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        return jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

    def _expand_noises_for_mbr(
        self,
        noise: np.ndarray | None,
        num_decision_candidates: int,
        num_reference_candidates: int,
    ) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
        if noise is None:
            return [None] * num_decision_candidates, [None] * num_reference_candidates

        noise = np.asarray(noise)
        if noise.ndim == 2:
            return [noise, *([None] * (num_decision_candidates - 1))], [None] * num_reference_candidates

        if noise.ndim != 3:
            raise ValueError(
                "noise must have shape (action_horizon, action_dim) or "
                f"(num_candidates, action_horizon, action_dim), got shape={noise.shape}"
            )

        total_candidates = num_decision_candidates + num_reference_candidates
        if noise.shape[0] == total_candidates:
            return (
                [noise[i] for i in range(num_decision_candidates)],
                [noise[num_decision_candidates + i] for i in range(num_reference_candidates)],
            )
        if noise.shape[0] == num_decision_candidates:
            return [noise[i] for i in range(num_decision_candidates)], [None] * num_reference_candidates
        if noise.shape[0] == 1:
            return [noise[0], *([None] * (num_decision_candidates - 1))], [None] * num_reference_candidates

        raise ValueError(
            "When MBR is enabled, batched noise must have first dimension equal to 1, "
            "num_decision_candidates, or num_decision_candidates + num_reference_candidates, "
            f"got shape={noise.shape}, num_decision_candidates={num_decision_candidates}, "
            f"num_reference_candidates={num_reference_candidates}"
        )

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        use_mbr, mbr_num_candidates, mbr_num_reference_candidates = self._resolve_mbr_settings(inputs)
        inputs = self._input_transform(inputs)
        model_inputs = self._convert_inputs_to_model_arrays(inputs)

        start_time = time.monotonic()
        mbr_enabled = bool(use_mbr and mbr_num_candidates > 1)
        selected_idx = 0
        selected_risk = 0.0

        if not mbr_enabled:
            if not self._is_pytorch_model:
                self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
            else:
                sample_rng_or_pytorch_device = self._pytorch_device

            outputs = self._sample_once(sample_rng_or_pytorch_device, model_inputs, noise=noise)
            outputs = self._output_transform(outputs)
        else:
            total_candidates = mbr_num_candidates + mbr_num_reference_candidates
            if not self._is_pytorch_model:
                split_keys = jax.random.split(self._rng, total_candidates + 1)
                self._rng = split_keys[0]
                sample_rngs_or_devices = list(split_keys[1:])
            else:
                sample_rngs_or_devices = [self._pytorch_device] * total_candidates

            decision_noises, reference_noises = self._expand_noises_for_mbr(
                noise,
                mbr_num_candidates,
                mbr_num_reference_candidates,
            )
            candidate_outputs: list[dict[str, np.ndarray]] = []
            decision_actions: list[np.ndarray] = []
            reference_actions: list[np.ndarray] = []

            # We intentionally sample candidates sequentially to keep memory usage stable,
            # especially for pi0.5 checkpoints on GPU.
            decision_rngs_or_devices = sample_rngs_or_devices[:mbr_num_candidates]
            reference_rngs_or_devices = sample_rngs_or_devices[mbr_num_candidates:]
            for sample_rng_or_pytorch_device, candidate_noise in zip(
                decision_rngs_or_devices, decision_noises, strict=True
            ):
                candidate_output = self._sample_once(
                    sample_rng_or_pytorch_device,
                    model_inputs,
                    noise=candidate_noise,
                )
                candidate_output = self._output_transform(candidate_output)
                candidate_outputs.append(candidate_output)
                decision_actions.append(np.asarray(candidate_output["actions"], dtype=np.float32))

            for sample_rng_or_pytorch_device, candidate_noise in zip(
                reference_rngs_or_devices, reference_noises, strict=True
            ):
                reference_output = self._sample_once(
                    sample_rng_or_pytorch_device,
                    model_inputs,
                    noise=candidate_noise,
                )
                reference_output = self._output_transform(reference_output)
                reference_actions.append(np.asarray(reference_output["actions"], dtype=np.float32))

            selected_idx, candidate_risks = _select_mbr_candidate(
                decision_actions,
                reference_actions,
                action_names=self._action_names,
            )
            selected_risk = float(candidate_risks[selected_idx])
            outputs = candidate_outputs[selected_idx]

        model_time = time.monotonic() - start_time
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
            "mbr_enabled": mbr_enabled,
            "mbr_num_candidates": mbr_num_candidates if mbr_enabled else 1,
            "mbr_num_decision_candidates": mbr_num_candidates if mbr_enabled else 1,
            "mbr_num_reference_candidates": mbr_num_reference_candidates if mbr_enabled else 0,
            "mbr_selected_index": selected_idx,
            "mbr_selected_risk": selected_risk,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
