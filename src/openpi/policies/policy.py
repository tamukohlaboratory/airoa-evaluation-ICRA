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


def _estimate_mbr_risks(action_candidates: Sequence[np.ndarray]) -> np.ndarray:
    if not action_candidates:
        raise ValueError("action_candidates must not be empty")

    stacked = np.stack([np.asarray(candidate, dtype=np.float32) for candidate in action_candidates], axis=0)
    flat = stacked.reshape(stacked.shape[0], -1).astype(np.float64, copy=False)
    pairwise_sq_dist = np.mean((flat[:, None, :] - flat[None, :, :]) ** 2, axis=-1)
    return pairwise_sq_dist.mean(axis=1)


def _select_mbr_candidate(action_candidates: Sequence[np.ndarray]) -> tuple[int, np.ndarray]:
    risks = _estimate_mbr_risks(action_candidates)
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
            mbr_num_candidates: Number of candidate action chunks to sample when MBR is enabled.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._use_mbr = bool(use_mbr)
        if mbr_num_candidates < 1:
            raise ValueError(f"mbr_num_candidates must be >= 1, got {mbr_num_candidates}")
        self._mbr_num_candidates = int(mbr_num_candidates)

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    def _resolve_mbr_settings(self, obs: dict[str, Any]) -> tuple[bool, int]:
        use_mbr = self._use_mbr
        if "use_mbr" in obs:
            override = _parse_optional_bool(obs.pop("use_mbr"))
            if override is not None:
                use_mbr = override

        mbr_num_candidates = self._mbr_num_candidates
        if "mbr_num_candidates" in obs:
            override = _parse_optional_positive_int(obs.pop("mbr_num_candidates"))
            if override is not None:
                mbr_num_candidates = override

        return use_mbr, mbr_num_candidates

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

    def _expand_noises_for_mbr(self, noise: np.ndarray | None, num_candidates: int) -> list[np.ndarray | None]:
        if noise is None:
            return [None] * num_candidates

        noise = np.asarray(noise)
        if noise.ndim == 2:
            return [noise, *([None] * (num_candidates - 1))]

        if noise.ndim != 3:
            raise ValueError(
                "noise must have shape (action_horizon, action_dim) or "
                f"(num_candidates, action_horizon, action_dim), got shape={noise.shape}"
            )

        if noise.shape[0] == num_candidates:
            return [noise[i] for i in range(num_candidates)]
        if noise.shape[0] == 1:
            return [noise[0], *([None] * (num_candidates - 1))]

        raise ValueError(
            "When MBR is enabled, batched noise must have first dimension equal to 1 or num_candidates, "
            f"got shape={noise.shape}, num_candidates={num_candidates}"
        )

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        use_mbr, mbr_num_candidates = self._resolve_mbr_settings(inputs)
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
            if not self._is_pytorch_model:
                split_keys = jax.random.split(self._rng, mbr_num_candidates + 1)
                self._rng = split_keys[0]
                sample_rngs_or_devices = list(split_keys[1:])
            else:
                sample_rngs_or_devices = [self._pytorch_device] * mbr_num_candidates

            candidate_noises = self._expand_noises_for_mbr(noise, mbr_num_candidates)
            candidate_outputs: list[dict[str, np.ndarray]] = []
            candidate_actions: list[np.ndarray] = []

            # We intentionally sample candidates sequentially to keep memory usage stable,
            # especially for pi0.5 checkpoints on GPU.
            for sample_rng_or_pytorch_device, candidate_noise in zip(
                sample_rngs_or_devices, candidate_noises, strict=True
            ):
                candidate_output = self._sample_once(
                    sample_rng_or_pytorch_device,
                    model_inputs,
                    noise=candidate_noise,
                )
                candidate_output = self._output_transform(candidate_output)
                candidate_outputs.append(candidate_output)
                candidate_actions.append(np.asarray(candidate_output["actions"], dtype=np.float32))

            selected_idx, candidate_risks = _select_mbr_candidate(candidate_actions)
            selected_risk = float(candidate_risks[selected_idx])
            outputs = candidate_outputs[selected_idx]

        model_time = time.monotonic() - start_time
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
            "mbr_enabled": mbr_enabled,
            "mbr_num_candidates": mbr_num_candidates if mbr_enabled else 1,
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
