#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path

import numpy as np

from openpi.policies import policy as policy_lib
from openpi.policies import policy_config
from openpi.training import config as train_config
from runtime_core.websocket_policy_server import WebsocketPolicyServer


DEFAULT_HSR_VALID_ACTION_DIMS = (0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15)
ACTION_SAMPLE_MASK_NAME_KEYWORD = "mask"


def _name_requests_action_sample_mask(*values: str | None) -> bool:
    """Infer masked-action sampling from deploy names.

    Training jobs that use masked action sampling often encode the setting only
    in the experiment/checkpoint name. CLI overrides are not reliably recoverable
    from the checkpoint, so deploy enables the inference-time sample mask when
    the config name or checkpoint path contains "mask".
    """
    return any(ACTION_SAMPLE_MASK_NAME_KEYWORD in str(value).lower() for value in values if value)


def _parse_int_tuple(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a comma-separated list of integer action dims, got {value!r}."
        ) from exc


def _build_action_sample_mask(action_dim: int, valid_dims: tuple[int, ...]) -> np.ndarray:
    invalid_dims = [dim for dim in valid_dims if dim < 0 or dim >= action_dim]
    if invalid_dims:
        raise ValueError(f"action_sample_mask valid dims out of range for action_dim={action_dim}: {invalid_dims}")
    mask = np.zeros((action_dim,), dtype=np.float32)
    mask[list(valid_dims)] = 1.0
    return mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve OpenPI policy as websocket server for HSR client")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--config-name", required=True, help="Train config name (e.g. pi05_hsr)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--default-prompt", default=None, help="Fallback prompt if prompt key is missing")
    parser.add_argument("--record-dir", default=None, help="Optional directory for policy records")
    parser.add_argument(
        "--pytorch-device",
        default=None,
        help='Optional torch device override (e.g. "cuda", "cuda:0", "cpu")',
    )
    parser.add_argument(
        "--enable-action-sample-mask",
        action="store_true",
        help=(
            "Mask unused/padding action dimensions during inference sampling. "
            "Use this only for checkpoints trained with masked action sampling/loss."
        ),
    )
    parser.add_argument(
        "--action-sample-mask-valid-dims",
        type=_parse_int_tuple,
        default=None,
        help=(
            "Comma-separated valid action dims used when --enable-action-sample-mask is set. "
            "Defaults to the HSR 32-dim padded layout: 0,1,2,3,4,6,11,12,13,14,15."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_dir = str(Path(args.checkpoint_dir).expanduser())
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")

    config_name = args.config_name
    config = train_config.get_config(config_name)

    config_action_loss = getattr(config, "action_loss", None)
    config_sample_mask_enabled = bool(getattr(config_action_loss, "masks_actions_and_noise", False))
    explicit_sample_mask_enabled = bool(args.enable_action_sample_mask)
    name_sample_mask_enabled = _name_requests_action_sample_mask(config_name, checkpoint_dir, Path(checkpoint_dir).name)
    effective_sample_mask_enabled = explicit_sample_mask_enabled or name_sample_mask_enabled or config_sample_mask_enabled

    effective_sample_mask_dims = None
    if effective_sample_mask_enabled:
        if args.action_sample_mask_valid_dims is not None:
            effective_sample_mask_dims = tuple(args.action_sample_mask_valid_dims)
        elif config_sample_mask_enabled and config_action_loss is not None:
            effective_sample_mask_dims = tuple(config_action_loss.valid_action_dims)
        else:
            effective_sample_mask_dims = DEFAULT_HSR_VALID_ACTION_DIMS

    sample_kwargs = None
    if effective_sample_mask_dims is not None:
        sample_kwargs = {
            "action_sample_mask": _build_action_sample_mask(config.model.action_dim, effective_sample_mask_dims)
        }

    policy = policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        sample_kwargs=sample_kwargs,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )

    if args.record_dir:
        policy = policy_lib.PolicyRecorder(policy, args.record_dir)

    metadata = dict(policy.metadata)
    data_cfg = getattr(config, "data", None)
    metadata.update(
        {
            "config_name": config_name,
            "checkpoint_dir": checkpoint_dir,
            "server_host": args.host,
            "server_port": args.port,
            "action_mode": getattr(data_cfg, "action_mode", None),
            "convert_gripper": getattr(data_cfg, "convert_gripper", None),
            "adapt_to_pi": getattr(data_cfg, "adapt_to_pi", None),
            "base_action_dim": getattr(data_cfg, "base_action_dim", None),
            "action_sample_mask_enabled": effective_sample_mask_enabled,
            "action_sample_mask_explicit": explicit_sample_mask_enabled,
            "action_sample_mask_auto_by_name": name_sample_mask_enabled,
            "action_sample_mask_from_config": config_sample_mask_enabled,
            "action_sample_mask_valid_action_dims": effective_sample_mask_dims,
        }
    )

    logging.info(
        "Serving policy config=%s action_mode=%s sample_mask=%s explicit_sample_mask=%s auto_by_name=%s checkpoint=%s on %s:%s",
        config_name,
        metadata.get("action_mode"),
        effective_sample_mask_enabled,
        explicit_sample_mask_enabled,
        name_sample_mask_enabled,
        checkpoint_dir,
        args.host,
        args.port,
    )
    # NOTE: Keep the OpenPI implementation as needed, but do not change the next two lines.
    # They are the fixed websocket serving contract for the HSR client runtime.
    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
