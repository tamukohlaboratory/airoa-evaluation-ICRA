"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.hsr_policy as hsr_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None
    select_episodes: list[int] | None = None

class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
        
@dataclasses.dataclass(frozen=True)
class LeRobotHSRDataConfig(DataConfigFactory):
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    # If true, this will convert the joint and gripper values from the HSR space to
    # the space used by the pi internal runtime (trossen mobile) which was used to train the base model. People who
    # use the HSR data should set this to true.
    adapt_to_pi: bool = True

    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action.state_diff", "action.relative")

    # Select which action source to use.
    # - "relative": use only action.relative
    # - "absolute_arm_head_relative_gripper_base": use arm/head from action.absolute and gripper/base from action.relative
    # - "state_diff_arm_head_relative_gripper_base": use arm/head from action.state_diff and gripper/base from action.relative
    action_mode: str = "relative"

    # If true, apply gripper conversion between HSR and pi0 angular space.
    convert_gripper: bool = False

    # Base action dimension appended from action.relative when action_mode is state_diff_with_base.
    base_action_dim: int = 3
    select_episodes: list[int] | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:

        if self.action_mode == "relative":
            action_sequence_keys = ("action.relative",)
            repack_transform = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "head_rgb": "observation.image.head",
                            "hand_rgb": "observation.image.hand",
                            "state": "observation.state",
                            "actions": "action.relative",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
            data_transforms = _transforms.Group(
                inputs=[
                    hsr_policy.HSRInputs(
                        action_dim=model_config.action_dim,
                        adapt_to_pi=self.adapt_to_pi,
                        convert_gripper=self.convert_gripper,
                    )
                ],
                outputs=[
                    hsr_policy.HSROutputs(
                        adapt_to_pi=self.adapt_to_pi,
                        convert_gripper=self.convert_gripper,
                    )
                ],
            )
        elif self.action_mode == "absolute_arm_head_relative_gripper_base":
            action_sequence_keys = ("action.absolute", "action.relative")
            repack_transform = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "head_rgb": "observation.image.head",
                            "hand_rgb": "observation.image.hand",
                            "state": "observation.state",
                            "actions_absolute": "action.absolute",
                            "actions_relative": "action.relative",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
            data_transforms = _transforms.Group(
                inputs=[
                    _transforms.CombineStateDiffArmHeadRelativeGripperBase(
                        state_diff_key="actions_absolute",
                        relative_key="actions_relative",
                        base_dim=self.base_action_dim,
                    ),
                    hsr_policy.HSRInputs(
                        action_dim=model_config.action_dim,
                        adapt_to_pi=self.adapt_to_pi,
                        convert_gripper=self.convert_gripper,
                    )
                ],
                outputs=[
                    hsr_policy.HSROutputs(
                        adapt_to_pi=self.adapt_to_pi,
                        convert_gripper=self.convert_gripper,
                    )
                ],
            )
        elif self.action_mode == "state_diff_arm_head_relative_gripper_base":
            action_sequence_keys = ("action.state_diff", "action.relative")
            repack_transform = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "head_rgb": "observation.image.head",
                            "hand_rgb": "observation.image.hand",
                            "state": "observation.state",
                            "actions_state_diff": "action.state_diff",
                            "actions_relative": "action.relative",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
            data_transforms = _transforms.Group(
                inputs=[
                    _transforms.CombineStateDiffArmHeadRelativeGripperBase(
                        base_dim=self.base_action_dim
                    ),
                    hsr_policy.HSRInputs(
                        action_dim=model_config.action_dim,
                        adapt_to_pi=self.adapt_to_pi,
                        convert_gripper=self.convert_gripper,
                    ),
                ],
                outputs=[
                    hsr_policy.HSROutputs(
                        adapt_to_pi=self.adapt_to_pi,
                        convert_gripper=self.convert_gripper,
                    )
                ],
            )
        else:
            raise ValueError(
                "Invalid action_mode. Expected 'relative', "
                "'absolute_arm_head_relative_gripper_base', or "
                "'state_diff_arm_head_relative_gripper_base'."
            )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs,model_config=model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=action_sequence_keys,
            select_episodes=self.select_episodes,
        )

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "hsr_openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"
    
    # sample the first batch and send to the wandb
    pytorch_sample_data: bool = False

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000
    # If set, derive the total number of train steps from the dataset size and global batch size.
    # This value represents the total number of epochs from step 0 (not additional epochs when resuming).
    # If provided, this overrides num_train_steps.
    num_train_epochs: float | None = 1.0

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if self.num_train_steps <= 0:
            raise ValueError("num_train_steps must be > 0.")
        if self.num_train_epochs is not None and self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be > 0 when set.")

# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_hsr",
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotHSRDataConfig(
            repo_id="processed/2025-05-06-07-v3.1-success-only",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m"
        ).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=256,
        num_train_steps=200_000,
        pytorch_weight_path="/home/user_00103_25b505/shared-storage/dev/models/pi0",
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),

    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi05_hsr",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotHSRDataConfig(
            repo_id="processed/2025-05-06-07-v3.1-success-only",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=200_000,
        batch_size=512,
        num_workers=8, # Increase num_workers to speed up data loading with larger datasets.
        pytorch_weight_path="/home/user_00103_25b505/shared-storage/dev/models/pi05",
    ),
    TrainConfig(
        name="pi05_hsr_task47_ep50",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotHSRDataConfig(
            repo_id="/home/hyamaguchi23/learning_ws/airoa-moma",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        batch_size=32,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=10000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),
    TrainConfig(
        name="pi05_hsr_task47_ep50_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotHSRDataConfig(
            repo_id="lerobot/airoa-moma",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        batch_size=50,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=10000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),
    TrainConfig(
        name="pi05_hsr_single_task",
        num_train_epochs=3.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotHSRDataConfig(
            repo_id="airoa-moma",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        batch_size=32,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=10000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),

    TrainConfig(
        name="pi05_hsr_single_task_aws",
        num_train_epochs=3.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotHSRDataConfig(
            repo_id="/mnt/AiroaMomaDataset",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        batch_size=32,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=10000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),
    TrainConfig(
        name="pi05_hsr_03_16_aws",
        num_train_epochs=3.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotHSRDataConfig(
            repo_id="/mnt/AiroaMomaDataset",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            select_episodes=[1571579, 1568828, 1570249, 1563362, 1570645, 2535191, 1555261, 1573907, 1574791, 1559073, 1573918, 1570325, 1575443, 1563706, 1575833, 1573036, 1565831, 1558909, 1564681, 1568238, 1560071, 1568584, 1554969, 1574801, 2535254, 1574778, 1563635, 1569521, 1575700, 1560814, 1569708, 1558730, 2535232, 1572550, 1564743, 2534377, 1572951, 1572185, 1561400, 1560016, 1567226, 1575708, 1560374, 1570403, 2533669, 1570265, 1554895, 1571683, 1565315, 1569214, 1555572, 1558942, 1566434, 2534260, 2533961, 1564965, 1563123, 1556654, 1563158, 1570834, 1557105, 1557264, 1575117, 1569188, 2534056, 1556913, 1575687, 1570277, 1575784, 1559292, 1559319, 1573440, 1568968, 1570583, 1560556, 1563667, 1570237, 2535242, 1572073, 1567960, 1562226, 1575489, 1575087, 1574153, 1561625, 1574635, 1565056, 1567541, 1573760, 1573215, 1567223, 1568297, 1570030, 1568538, 1559224, 1563274, 1562704, 1556788, 1565818, 1555631, 2534413, 2534235, 1566199, 2534415, 1565942, 1555041, 1558870, 2535331, 2534791, 1558309, 1566179, 1558755, 1556819, 1570612, 2534013, 1566577, 1568482, 2535091, 2533807, 1565809, 2534147, 1562487, 2534329, 2534355, 2533728, 1566979, 1573554, 1564841, 1560011, 1560150, 2535029, 1574624, 1571509, 1574327, 1573716, 2533702, 1557894, 2535121, 2534992, 2534914, 1560268, 1558406, 1573394, 1570936, 1560902, 1567323, 1565022, 2534137, 1575417, 1562769, 1574263, 1564656, 2533671, 1567354, 1559363, 1575171, 2534206, 1560265, 1557672, 2534984, 2534152, 1555779, 1559957, 1566426, 1564000, 1573496, 2533772, 1565802, 1573376, 1563961, 1572479, 1554920, 1572819, 1567863, 2533619, 1568816, 1574323, 2535256, 2534244, 2535042, 2535417, 2533818, 1563556, 1564840, 1569413, 1565852, 1558287, 2534367, 2534158, 1570195, 1558267, 1557624, 2534391, 2533753, 2533932, 2534117, 1563622, 1558246, 2533966, 1559487, 1556158, 1556114, 1558204, 1563640, 1564198, 1574141, 1555934, 1562895, 1575495, 1571390, 1563702, 1572453, 1566331, 1555116, 1561183, 1571480, 1569058, 1566327, 1560951, 1563549, 1568882, 1558596, 1558125, 1570309, 1558367, 1569602, 1569146, 1565821, 1556535, 1572778, 1575186, 1559512, 1570252, 1557558, 1575703, 1567292, 1569662, 1562695, 1557252, 1556605, 1564054, 1567172, 1557593, 1564204, 1558504, 1570311, 1566329, 1572567, 1569765, 1561274, 1569925, 1569514, 1563279, 1565932, 1557313, 1561678, 1575132, 1564907, 1561299, 1572844, 1570303, 1566065, 1575864, 1563679, 1568416, 1556905, 1564110, 1556160, 1568178, 1570930, 1565976, 1557198, 1563340, 1568143, 1563420, 1573954, 1570798, 1572755, 1560438, 1565838, 1560294, 1565013, 1575291, 1565747, 1571738, 1570903, 1569677, 1563658, 1560244, 1574247, 1573761, 1557994, 1556626, 1558928, 1560854, 1561196, 1571464, 1557136, 1570434, 1570355],
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        batch_size=32,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=10000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),

    TrainConfig(
        name="pi05_hsr_curation_hf_dataset",
        num_train_epochs=1.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotHSRDataConfig(
            repo_id="airoa-moma",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        batch_size=32,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=10000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),

    TrainConfig(
        name="pi05_hsr_curation_hf_dataset_v2",
        num_train_epochs=3.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotHSRDataConfig(
            repo_id="airoa-moma",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            select_episodes = [0, 10, 52, 54, 57, 78, 84, 87, 92, 94, 98, 107, 119, 125, 129, 132, 141, 142, 147, 160, 165, 173, 174, 194, 195, 205, 207, 208, 229, 255, 258, 259, 273, 283, 286, 291, 296, 298, 317, 319, 334, 341, 342, 349, 355, 377, 383, 386, 401, 420, 423, 424, 427, 429, 441, 446, 447, 457, 463, 464, 465, 471, 472, 477, 480, 485, 486, 490, 498, 505, 508, 509, 521, 531, 532, 535, 543, 546, 549, 555, 561, 565, 569, 579, 594, 596, 618, 628, 630, 631, 639, 640, 649, 654, 661, 668, 670, 681, 692, 695, 696, 716, 721, 724, 727, 745, 752, 755, 758, 770, 782, 827, 830, 844, 848, 849, 862, 888, 897, 901, 904, 913, 919, 931, 932, 935, 956, 966, 973, 992, 1009, 1036, 1053, 1059, 1060, 1063, 1069, 1076, 1086, 1115, 1116, 1165, 1180, 1183, 1195, 1206, 1224, 1268, 1296, 1299, 1323, 1338, 1341, 1345, 1369, 1402, 1404, 1428, 1434, 1445, 1453, 1455, 1461, 1464, 1476, 1479, 1480, 1484, 1495, 1501, 1507, 1513, 1515, 1522, 1525, 1532, 1534, 1542, 1554, 1556, 1570, 1571, 1589, 1594, 1595, 1618, 1621, 1625, 1660, 1689, 1698, 1722, 1769, 1776, 1786, 1789, 1800, 1808, 1809, 1814, 1815, 1822, 1824, 1835, 1837, 1842, 1850, 1852, 1856, 1872, 1894, 1922, 1924, 1964, 1966, 1967, 1969, 1981, 1994, 1995, 2002, 2004, 2005, 2012, 2014, 2023, 2056, 2057, 2061, 2066, 2073, 2103, 2121, 2133, 2148, 2162, 2175, 2192, 2207, 2211, 2215, 2225, 2251, 2267, 2278, 2300, 2303, 2339, 2376, 2385, 2395, 2398, 2419, 2428, 2435, 2444, 2445, 2468, 2486, 2492, 2498, 2503, 2515, 2524, 2526, 2540, 2546, 2547, 2558, 2559, 2560, 2566, 2568, 2583, 2587, 2607, 2609, 2618, 2619, 2621, 2622, 2639, 2652, 2653, 2668, 2671, 2714, 2716, 2720, 2725, 2726, 2728, 2733, 2737, 2741, 2742, 2750, 2758, 2764, 2768, 2774, 2797, 2799, 2811, 2818, 2822, 2833, 2836, 2837, 2841, 2869, 2892, 2912, 2996, 2997, 3004, 3011, 3012, 3037, 3040, 3041, 3056, 3091, 3097, 3111, 3118, 3128, 3144, 3154, 3159, 3181, 3192, 3223, 3259, 3260, 3261, 3264, 3266, 3306, 3311, 3325, 3334, 3339, 3351, 3363, 3387, 3391, 3396, 3410, 3413, 3418, 3423, 3430, 3456, 3474, 3488, 3495, 3509, 3515, 3529, 3536, 3541, 3548, 3557, 3562, 3570, 3574, 3607, 3611, 3631, 3633, 3637, 3644, 3645, 3646, 3651, 3653, 3662, 3665, 3667, 3676, 3684, 3689, 3696, 3708, 3711, 3717, 3720, 3740, 3757, 3758, 3765, 3770, 3780, 3784, 3818, 3827, 3830, 3831, 3857, 3865, 3888, 3899, 3916, 3917, 3924, 3929, 3931, 3937, 3939, 3945, 3952, 3971, 3973, 3999, 4019, 4042, 4046, 4055, 4069, 4081, 4086, 4089, 4104, 4113, 4119, 4128, 4147, 4166, 4167, 4169, 4172, 4173, 4185, 4192, 4212, 4216, 4225, 4227, 4228, 4238, 4243, 4247, 4259, 4261, 4264, 4268, 4275, 4276, 4277, 4280, 4289, 4293, 4294, 4303, 4304, 4312, 4313, 4318, 4321, 4325, 4327, 4333, 4339, 4343, 4355, 4357, 4360, 4362, 4363, 4369, 4372, 4373, 4376, 4377, 4386, 4392, 4394, 4395, 4397, 4398, 4410, 4420, 4421, 4430, 4434, 4445, 4449, 4490, 4494, 4500, 4504, 4513, 4536, 4543, 4545, 4548, 4549, 4556, 4559, 4567, 4572, 4575, 4576, 4579, 4582, 4598, 4602, 4610, 4613, 4623, 4624, 4651, 4653, 4660, 4682, 4686, 4693, 4705, 4708, 4712, 4720, 4721, 4722, 4727, 4729, 4739, 4740, 4749, 4770, 4781, 4785, 4788, 4807, 4851, 4882, 4948, 4952, 4953, 4956, 4963, 4968, 4969, 4985, 4987, 4999, 5000, 5011, 5013, 5020, 5025, 5036, 5048, 5054, 5057, 5062, 5068, 5070, 5071, 5079, 5083, 5088, 5090, 5102, 5107, 5122, 5123, 5128, 5129, 5130, 5148, 5152, 5154, 5172, 5174, 5192, 5194, 5200, 5214, 5219, 5224, 5231, 5234, 5244, 5248, 5264, 5269, 5273, 5279, 5282, 5284, 5287, 5290, 5295, 5299, 5301, 5302, 5304, 5305, 5307, 5320, 5323, 5334, 5344, 5347, 5359, 5362, 5370, 5378, 5379, 5381, 5388, 5393, 5418, 5425, 5426, 5427, 5449, 5452, 5466, 5477, 5478, 5502, 5503, 5506, 5507, 5508, 5513, 5514, 5520, 5524, 5528, 5533, 5550, 5564, 5567, 5573, 5578, 5582, 5584, 5585, 5590, 5592, 5598, 5600, 5602, 5606, 5610, 5611, 5638, 5641, 5646, 5648, 5658, 5664, 5679, 5681, 5692, 5699, 5701, 5706, 5724, 5740, 5745, 5769, 5774, 5781, 5782, 5799, 5802, 5811, 5812, 5846, 5847, 5859, 5869, 5872, 5879, 5882, 5884, 5890, 5892, 5894, 5895, 5898, 5905, 5915, 5927, 5930, 5934, 5953, 5975, 5985, 5987, 6035, 6050, 6052, 6055, 6060, 6084, 6099, 6103, 6135, 6149, 6159, 6163, 6168, 6181, 6203, 6217, 6238, 6239, 6247, 6250, 6254, 6265, 6268, 6288, 6290, 6298, 6299, 6307, 6359, 6373, 6395, 6400, 6423, 6430, 6432, 6438, 6455, 6456, 6460, 6478, 6483, 6497, 6503, 6509, 6515, 6527, 6534, 6538, 6539, 6546, 6547, 6548, 6552, 6558, 6572, 6573, 6579, 6605, 6614, 6625, 6630, 6633, 6635, 6638, 6639, 6649, 6653, 6656, 6662, 6666, 6667, 6669, 6671, 6679, 6683, 6695, 6697, 6702, 6705, 6716, 6720, 6724, 6729, 6730, 6741, 6754, 6756, 6757, 6759, 6760, 6766, 6774, 6775, 6779, 6790, 6800, 6808, 6812, 6817, 6826, 6827, 6836, 6845, 6855, 6859, 6861, 6862, 6864, 6872, 6883, 6904, 6907, 6909, 6918, 6925, 6931, 6935, 6937, 6948, 6953, 6969, 6971, 6988, 6993, 6997, 7052, 7054, 7060, 7062, 7082, 7084, 7094, 7098, 7113, 7144, 7150, 7170, 7181, 7184, 7195, 7196, 7205, 7207, 7212, 7222, 7225, 7237, 7245, 7258, 7259, 7262, 7265, 7281, 7283, 7284, 7302, 7310, 7313, 7314, 7318, 7324, 7331, 7333, 7340, 7341, 7350, 7360, 7361, 7381, 7383, 7389, 7393, 7397, 7398, 7418, 7426, 7430, 7433, 7444, 7446, 7453, 7461, 7462, 7468, 7482, 7490, 7496, 7497, 7511, 7532, 7543, 7546, 7556, 7561, 7564, 7575, 7580, 7585, 7601, 7602, 7617, 7635, 7660, 7667, 7672, 7683, 7688, 7694, 7700, 7715, 7736, 7742, 7759, 7791, 7808, 7819, 7824, 7829, 7833, 7842, 7856, 7874, 7882, 7885, 7894, 7896, 7923, 7936, 7962, 7976, 8003, 8027, 8049, 8082, 8099, 8117, 8119, 8126, 8150, 8151, 8160, 8173, 8191, 8192, 8194, 8209, 8220, 8232, 8244, 8262, 8288, 8299, 8302, 8322, 8335, 8355, 8374, 8391, 8441, 8458, 8466, 8472, 8475, 8478, 8495, 8507, 8539, 8615, 8630, 8638, 8646, 8648, 8655, 8664, 8666, 8713, 8716, 8740, 8741, 8745, 8747, 8750, 8765, 8769, 8776, 8787, 8790, 8792, 8796, 8801, 8805, 8806, 8815, 8828, 8837, 8839, 8840, 8847, 8853, 8854, 8855, 8856, 8859, 8864, 8881, 8885, 8888, 8895, 8919, 8922, 8928, 8960, 8990, 8995, 9004, 9005, 9011, 9040, 9061, 9080, 9084, 9088, 9089, 9092, 9111, 9126, 9150, 9160, 9162, 9169, 9184, 9197, 9213, 9231, 9234, 9245, 9249, 9250, 9254, 9257, 9258, 9261, 9268, 9272, 9275, 9287, 9290, 9294, 9307, 9309, 9323, 9342, 9350, 9352, 9388, 9390, 9445, 9449, 9478, 9481, 9502, 9503, 9504, 9513, 9532, 9534, 9535, 9536, 9540, 9545, 9546, 9561, 9570, 9573, 9576, 9577, 9583, 9592, 9600, 9605, 9610, 9617, 9622, 9627, 9647, 9653, 9659, 9663, 9669, 9681, 9686, 9696, 9700, 9704, 9706, 9708, 9709, 9714, 9716, 9719, 9722, 9739, 9746, 9785, 9788, 9796, 9804, 9808, 9811, 9812, 9832, 9836, 9837, 9840, 9844, 9854, 9863, 9871, 9876, 9881, 9883, 9887, 9894, 9896, 9900, 9903, 9908, 9922, 9928, 9932, 9935, 9938, 9941, 9956, 9957, 9966, 9985, 9995, 10005, 10006, 10010, 10017, 10021, 10022, 10031, 10039, 10054, 10072, 10083, 10085, 10095, 10097, 10098, 10101, 10105, 10109, 10110, 10122, 10123, 10124, 10130, 10131, 10140, 10149, 10159, 10178, 10184, 10187, 10188, 10199, 10206, 10208, 10210, 10215, 10220, 10223, 10226, 10230, 10234, 10236, 10248, 10264, 10270, 10272, 10277, 10278, 10279, 10280, 10281, 10288, 10290, 10291, 10296, 10297, 10298, 10299, 10303, 10304, 10305, 10315, 10322, 10324, 10325, 10327, 10339, 10345, 10363, 10365, 10379, 10381, 10382, 10387, 10388, 10389, 10395, 10404, 10405, 10409, 10410, 10414, 10417, 10420, 10421, 10422, 10423, 10427, 10429, 10432, 10436, 10443, 10444, 10446, 10454, 10457, 10458, 10463, 10472, 10473, 10487, 10493, 10494, 10497, 10498, 10500, 10503, 10507, 10516, 10519, 10520, 10522, 10533, 10542, 10546, 10547, 10549, 10552, 10556, 10557, 10562, 10563, 10570, 10574, 10586, 10589, 10593, 10596, 10599, 10610, 10612, 10620, 10625, 10630, 10633, 10639, 10640, 10644, 10646, 10651, 10653, 10656, 10661, 10673, 10676, 10684, 10694, 10695, 10697, 10701, 10722, 10729, 10733, 10736, 10744, 10746, 10747, 10750, 10759, 10765, 10766, 10771, 10778, 10790, 10800, 10809, 10812, 10820, 10833, 10834, 10838, 10841, 10845, 10847, 10860, 10864, 10869, 10871, 10872, 10874, 10875, 10888, 10889, 10894, 10896, 10897, 10907, 10911, 10922, 10936, 10937, 10938, 10944, 10952, 10954, 10985, 10993, 11000, 11002, 11018, 11038, 11045, 11057, 11059, 11065, 11086, 11088, 11089, 11095, 11102, 11103, 11107, 11115, 11116, 11120, 11127, 11129, 11132, 11136, 11137, 11147, 11157, 11159, 11161, 11163, 11166, 11171, 11178, 11179, 11185, 11188, 11189, 11197, 11201, 11203, 11209, 11210, 11214, 11216, 11217, 11229, 11234, 11238, 11239, 11241, 11247, 11249, 11252, 11254, 11257, 11264, 11270, 11282, 11285, 11289, 11291, 11301, 11303, 11307, 11309, 11317, 11320, 11322, 11323, 11329, 11343, 11344, 11349, 11352, 11353, 11356, 11362, 11363, 11382, 11394, 11399, 11403, 11405, 11406, 11408, 11413, 11421, 11422, 11423, 11428, 11431, 11435, 11441, 11444, 11453, 11465, 11466, 11471, 11472, 11473, 11481, 11488, 11496, 11501, 11503, 11508, 11510, 11512, 11527, 11532, 11556, 11567, 11570, 11584, 11585, 11590, 11595, 11600, 11604, 11618, 11619, 11628, 11630, 11631, 11636, 11640, 11651, 11653, 11657, 11658, 11662, 11663, 11666, 11667, 11676, 11687, 11693, 11699, 11714, 11720, 11723, 11725, 11726, 11728, 11733, 11738, 11742, 11744, 11746, 11755, 11778, 11789, 11790, 11797, 11799, 11802, 11805, 11825, 11827, 11841, 11846, 11848, 11854, 11869, 11877, 11879, 11881, 11886, 11887, 11889, 11892, 11894, 11910, 11918, 11919, 11932, 11935, 11936, 11942, 11943, 11944, 11985, 11988, 11998, 12039, 12050, 12053, 12069, 12074, 12075, 12076, 12078, 12085, 12089, 12093, 12095, 12096, 12098, 12099, 12106, 12107, 12108, 12109, 12117, 12120, 12131, 12132, 12133, 12138, 12140, 12145, 12146, 12161, 12163, 12166, 12169, 12176, 12179, 12184, 12198, 12202, 12211, 12219, 12220, 12224, 12229, 12238, 12240, 12246, 12247, 12252, 12253, 12255, 12264, 12269, 12274, 12275, 12276, 12279, 12285, 12287, 12301, 12302, 12306, 12307, 12308, 12312, 12316, 12319, 12322, 12353, 12367, 12381, 12407, 12417, 12418, 12433, 12438, 12443, 12463, 12464, 12467, 12468, 12472, 12489, 12502, 12513, 12515, 12518, 12531, 12537, 12539, 12542, 12554, 12559, 12564, 12583, 12589, 12598, 12607, 12609, 12615, 12616, 12622, 12632, 12635, 12636, 12645, 12669, 12688, 12689, 12693, 12694, 12700, 12704, 12705, 12707, 12709, 12712, 12717, 12718, 12723, 12733, 12734, 12746, 12768, 12770, 12789, 12792, 12798, 12805, 12820, 12822, 12829, 12835, 12846, 12858, 12862, 12863, 12878, 12888, 12889, 12906, 12910, 12935, 12936, 12944, 12953, 12958, 12959, 12960, 12961, 12962, 12963, 12967, 12969, 12979, 12983, 12991, 12995, 12998, 13004, 13019, 13021, 13031, 13034, 13035, 13037, 13055, 13056, 13068, 13070, 13072, 13073, 13086, 13097, 13101, 13103, 13104, 13111, 13114, 13116, 13127, 13141, 13154, 13158, 13159, 13167, 13208, 13210, 13215, 13217, 13273, 13274, 13277, 13280, 13281, 13283, 13285, 13292, 13296, 13306, 13312, 13318, 13330, 13335, 13337, 13356, 13365, 13368, 13373, 13375, 13380, 13381, 13395, 13418, 13428, 13434, 13436, 13437, 13441, 13443, 13445, 13450, 13462, 13467, 13469, 13485, 13488, 13493, 13495, 13497, 13506, 13513, 13517, 13522, 13523, 13530, 13536, 13541, 13544, 13550, 13555, 13573, 13581, 13587, 13588, 13594, 13603, 13611, 13614, 13618, 13622, 13625, 13630, 13636, 13637, 13671, 13677, 13688, 13693, 13695, 13710, 13714, 13719, 13724, 13727, 13733, 13747, 13748, 13758, 13766, 13788, 13791, 13792, 13799, 13802, 13806, 13808, 13809, 13819, 13829, 13831, 13832, 13842, 13845, 13861, 13864, 13884, 13885, 13896, 13898, 13905, 13960, 13984, 14028, 14031, 14039, 14068, 14088, 14101, 14181, 14224, 14282, 14295, 14296, 14305, 14320, 14322, 14330, 14368, 14395, 14398, 14413, 14429, 14442, 14463, 14511, 14532, 14534, 14613, 14618, 14857, 14893, 14899, 14908, 14916, 14922, 14941, 15066, 15091, 15108, 15134, 15141, 15165, 15181, 15201, 15227, 15260, 15269, 15293, 15316, 15365, 15397, 15403, 15410, 15447, 15450, 15457, 15486, 15506, 15529, 15558, 15601, 15634, 15646, 15725, 15748, 15749, 15762, 15775, 15786, 15802, 15812, 15850, 15881, 15921, 15927, 15954, 15997, 16038, 16081, 16098, 16120, 16125, 16135, 16144, 16159, 16260, 16267, 16290, 16310, 16315, 16346, 16380, 16386, 16389, 16392, 16432, 16442, 16443, 16455, 16491, 16505, 16521, 16557, 16785, 16792, 16820, 16823, 16883, 16920, 16941, 16951, 16970, 17003, 17064, 17066, 17070, 17078, 17087, 17102, 17123, 17180, 17215, 17224, 17231, 17266, 17285, 17303, 17344, 17356, 17377, 17379, 17394, 17427, 17453, 17472, 17481, 17482, 17495, 17497, 17520, 17523, 17541, 17570, 17573, 17600, 17611, 17622, 17670, 17676, 17679, 17695, 17712, 17730, 17742, 17743, 17749, 17765, 17767, 17774, 17778, 17789, 17799, 17815, 17816, 17828, 17880, 17887, 17893, 17894, 17912, 17917, 17937, 17943, 17988, 17994, 18044, 18045, 18054, 18057, 18060, 18080, 18097, 18135, 18156, 18200, 18265, 18267, 18271, 18318, 18340, 18341, 18362, 18419, 18423, 18435, 18477, 18484, 18517, 18537, 18616, 18636, 18704, 18710, 18741, 18820, 18838, 18854, 18918, 18933, 18936, 18984, 19095, 19115, 19129, 19133, 19146, 19152, 19153, 19159, 19177, 19198, 19212, 19304, 19310, 19311, 19391, 19503, 19510, 19529, 19536, 19544, 19613, 19619, 19635, 19702, 19888, 19900, 19980, 20068, 20069, 20070, 20080, 20112, 20139, 20197, 20213, 20272, 20276, 20312, 20356, 20368, 20390, 20419, 20424, 20427, 20461, 20504, 20532, 20553, 20595, 20629, 20656, 20681, 20687, 20707, 20748, 20810, 20879, 20886, 20905, 20909, 20948, 20958, 21003, 21045, 21054, 21062, 21068, 21089, 21108, 21121, 21158, 21198, 21211, 21217, 21219, 21229, 21235, 21236, 21267, 21274, 21308, 21340, 21358, 21431, 21439, 21440, 21453, 21465, 21467, 21490, 21501, 21510, 21519, 21527, 21537, 21539, 21551, 21561, 21564, 21580, 21585, 21588, 21589, 21596, 21602, 21607, 21614, 21623, 21630, 21632, 21638, 21645, 21647, 21649, 21653, 21655, 21657, 21658, 21659, 21661, 21671, 21677, 21681, 21691, 21696, 21697, 21699, 21709, 21732, 21734, 21735, 21736, 21739, 21743, 21756, 21758, 21761, 21765, 21781, 21787, 21799, 21801, 21803, 21817, 21824, 21825, 21829, 21830, 21832, 21859, 21861, 21863, 21864, 21870, 21871, 21878, 21883, 21886, 21891, 21894, 21899, 21902, 21912, 21919, 21924, 21949, 21958, 21965, 21967, 21970, 21974, 21981, 21982, 21983, 21994, 21999, 22002, 22010, 22013, 22017, 22019, 22021, 22041, 22044, 22048, 22050, 22070, 22100, 22102, 22104, 22109, 22110, 22112, 22117, 22118, 22124, 22126, 22128, 22139, 22148, 22153, 22160, 22161, 22167, 22170, 22186, 22190, 22193, 22201, 22216, 22229, 22231, 22236, 22241, 22255, 22267, 22287, 22289, 22295, 22321, 22329, 22334, 22336, 22347, 22367, 22368, 22375, 22379, 22386, 22389, 22405, 22416, 22420, 22423, 22443, 22471, 22477, 22487, 22490, 22503, 22559, 22565, 22576, 22578, 22636, 22644, 22657, 22659, 22689, 22691, 22699, 22707, 22711, 22713, 22761, 22765, 22768, 22782, 22795, 22806, 22843, 22844, 22863, 22864, 22872, 22921, 22928, 22933, 22946, 22947, 22958, 22974, 22975, 22981, 22984, 22989, 22992, 22999, 23022, 23034, 23081, 23103, 23104, 23135, 23137, 23147, 23163, 23175, 23198, 23207, 23209, 23211, 23221, 23223, 23228, 23236, 23253, 23290, 23293, 23320, 23332, 23337, 23349, 23354, 23364, 23384, 23389, 23393, 23397, 23400, 23405, 23414, 23454, 23459, 23469, 23487, 23517, 23563, 23579, 23580, 23582, 23585, 23588, 23601, 23602, 23604, 23613, 23622, 23623, 23632, 23645, 23658, 23661, 23671, 23689, 23696, 23713, 23718, 23721],
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        batch_size=32,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=10000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),

    TrainConfig(
        name="pi05_hsr_curation_pull_the_chain_to_turn_off_the_light",
        num_train_epochs=3.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotHSRDataConfig(
            repo_id="airoa-moma",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        batch_size=32,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=10000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),
   TrainConfig(
        name="pi05_ep50_epoch5_curation_pick_up_a_slice_of_bread_on_the_plate",
        num_train_epochs=5.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotHSRDataConfig(
            repo_id="airoa-moma",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            select_episodes = [721, 727, 758, 844, 897, 904, 935, 956, 1183, 1206, 1299, 1404, 1445, 1453, 1476, 1484, 1542, 1556, 1594, 1698, 1786, 1815, 1852, 1856, 1872, 1924, 1966, 1969, 2014, 2121, 2133, 2492, 2515, 2559, 2609, 2716, 2742, 2750, 2764, 2799, 2836, 3474, 3509, 3541, 3548, 3633, 3696, 3720, 3758, 3830],
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        batch_size=32,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=10000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),
   TrainConfig(
        name="Pick_up_the_mug_ep50_100epock",
        num_train_epochs=100.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotHSRDataConfig(
            repo_id="/mnt/AiroaMomaDataset",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            select_episodes = [2536226, 2536013, 2535738, 2536152, 2536393, 2536365, 2536315, 2535724, 2536341, 2535877, 2535988, 2536297, 2536272, 2535961, 2536033, 2536125, 2536049, 2535629, 2535786, 2535759, 2535554, 2535918, 2535497, 2536202, 2536164, 2535763, 2535750, 2535480, 2535563, 2536335, 2536254, 2535551, 2535558, 2535511, 2535906, 2536123, 2535775, 2535841, 2536296, 2536096, 2535739, 2535646, 2536352, 2536184, 2536188, 2536079, 2535783, 2536440, 2536473, 2535997],
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        batch_size=32,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=10000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),
    TrainConfig(
        name="mug_pick_and_place_ep100_epoch100",
        num_train_epochs=100.0, 
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotHSRDataConfig(
            repo_id="/mnt/AiroaMomaDataset",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            select_episodes = [2536226, 2536013, 2535738, 2536152, 2536393, 2536365, 2536315, 2535724, 2536341, 2535877, 2535988, 2536297, 2536272, 2535961, 2536033, 2536125, 2536049, 2535629, 2535786, 2535759, 2535554, 2535918, 2535497, 2536202, 2536164, 2535763, 2535750, 2535480, 2535563, 2536335, 2536254, 2535551, 2535558, 2535511, 2535906, 2536123, 2535775, 2535841, 2536296, 2536096, 2535739, 2535646, 2536352, 2536184, 2536188, 2536079, 2535783, 2536440, 2536473, 2535997, 2534305, 2533864, 2534326, 2533614, 2534711, 2534907, 2534026, 2534391, 2534775, 2533898, 2534395, 2533775, 2534577, 2534094, 2534592, 2534869, 2534559, 2533926, 2533855, 2534194, 2534878, 2533873, 2534458, 2534800, 2534436, 2534911, 2534553, 2534129, 2534378, 2533647, 2533777, 2534187, 2534115, 2533991, 2533706, 2534747, 2534497, 2533745, 2533754, 2534368, 2533718, 2534441, 2533810, 2534696, 2534614, 2534357, 2534465, 2533879, 2534896, 2534795],
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=64,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=10000, # Save more frequently since the dataset is smaller.
    ),

   TrainConfig(
        name="relocate_all_ep600_epoch10_3_16",
        num_train_epochs=100.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ),
        data=LeRobotHSRDataConfig(
            repo_id="/mnt/AiroaMomaDataset",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            select_episodes = [2536226, 2536013, 2535738, 2536152, 2536393, 2536365, 2536315, 2535724, 2536341, 2535877, 2535988, 2536297, 2536272, 2535961, 2536033, 2536125, 2536049, 2535629, 2535786, 2535759, 2535554, 2535918, 2535497, 2536202, 2536164, 2535763, 2535750, 2535480, 2535563, 2536335, 2536254, 2535551, 2535558, 2535511, 2535906, 2536123, 2535775, 2535841, 2536296, 2536096, 2535739, 2535646, 2536352, 2536184, 2536188, 2536079, 2535783, 2536440, 2536473, 2535997, 2535778, 2535649, 2535650, 2536354, 2536087, 2536012, 2536080, 2536068, 2536133, 2536458, 2535570, 2536376, 2536350, 2536340, 2535652, 2535586, 2536060, 2536454, 2535995, 2535661, 2535859, 2535780, 2536210, 2536106, 2535707, 2535764, 2535903, 2536124, 2535624, 2536100, 2536222, 2535566, 2536346, 2536212, 2535499, 2535647, 2536515, 2535840, 2535745, 2536064, 2536150, 2536146, 2535808, 2536058, 2535581, 2535743, 2536038, 2535482, 2536046, 2535882, 2534305, 2533864, 2534326, 2533614, 2534711, 2534907, 2534026, 2534391, 2534775, 2533898, 2534395, 2533775, 2534577, 2534094, 2534592, 2534869, 2534559, 2533926, 2533855, 2534194, 2534878, 2533873, 2534458, 2534800, 2534436, 2534911, 2534553, 2534129, 2534378, 2533647, 2533777, 2534187, 2534115, 2533991, 2533706, 2534747, 2534497, 2533745, 2533754, 2534368, 2533718, 2534441, 2533810, 2534696, 2534614, 2534357, 2534465, 2533879, 2534896, 2534795, 2533995, 2534700, 2534481, 2534446, 2534012, 2534702, 2534508, 2534286, 2534862, 2534507, 2534239, 2534181, 2533956, 2533828, 2534403, 2534384, 2533755, 2534710, 2533678, 2534828, 2533788, 2533849, 2534564, 2533860, 2534754, 2534634, 2534278, 2534526, 2533703, 2534213, 2534209, 2534341, 2534428, 2534000, 2534927, 2534466, 2534900, 2533620, 2534676, 2533793, 2534635, 2534852, 2534444, 2534706, 2534016, 2534728, 2534585, 2534130, 2533792, 2534054, 2535360, 2535123, 2535407, 2535404, 2535176, 2535373, 2535061, 2534986, 2534966, 2535081, 2535121, 2534992, 2535084, 2535004, 2535239, 2535115, 2535303, 2535183, 2535044, 2535224, 2535173, 2535071, 2535108, 2534987, 2535050, 2535352, 2535091, 2535406, 2535306, 2535411, 2535111, 2535362, 2535078, 2535141, 2534974, 2535425, 2534957, 2535135, 2535265, 2535395, 2534984, 2535073, 2535358, 2535387, 2535327, 2535253, 2535305, 2535031, 2535105, 2535029, 2534995, 2534949, 2535118, 2535102, 2535092, 2535019, 2534991, 2535415, 2534982, 2535443, 2535332, 2534953, 2534952, 2534983, 2535085, 2535096, 2535388, 2535464, 2535426, 2535072, 2535449, 2535330, 2535089, 2535346, 2535120, 2534939, 2535038, 2535444, 2535280, 2535413, 2535035, 2535439, 2535279, 2535451, 2535440, 2535311, 2534962, 2535049, 2535399, 2535047, 2535140, 2535013, 2535323, 2534946, 2535297, 2535094, 2535113, 2534968, 2535066, 2534958],
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        batch_size=50,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=10000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),

   TrainConfig(
        name="relocate_all_ep300_epoch100_convert_gripper_True",
        num_train_epochs=100.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ),
        data=LeRobotHSRDataConfig(
            repo_id="/mnt/AiroaMomaDataset",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            convert_gripper = True,
            select_episodes = [2536226, 2536013, 2535738, 2536152, 2536393, 2536365, 2536315, 2535724, 2536341, 2535877, 2535988, 2536297, 2536272, 2535961, 2536033, 2536125, 2536049, 2535629, 2535786, 2535759, 2535554, 2535918, 2535497, 2536202, 2536164, 2535763, 2535750, 2535480, 2535563, 2536335, 2536254, 2535551, 2535558, 2535511, 2535906, 2536123, 2535775, 2535841, 2536296, 2536096, 2535739, 2535646, 2536352, 2536184, 2536188, 2536079, 2535783, 2536440, 2536473, 2535997, 2535778, 2535649, 2535650, 2536354, 2536087, 2536012, 2536080, 2536068, 2536133, 2536458, 2535570, 2536376, 2536350, 2536340, 2535652, 2535586, 2536060, 2536454, 2535995, 2535661, 2535859, 2535780, 2536210, 2536106, 2535707, 2535764, 2535903, 2536124, 2535624, 2536100, 2536222, 2535566, 2536346, 2536212, 2535499, 2535647, 2536515, 2535840, 2535745, 2536064, 2536150, 2536146, 2535808, 2536058, 2535581, 2535743, 2536038, 2535482, 2536046, 2535882, 2534305, 2533864, 2534326, 2533614, 2534711, 2534907, 2534026, 2534391, 2534775, 2533898, 2534395, 2533775, 2534577, 2534094, 2534592, 2534869, 2534559, 2533926, 2533855, 2534194, 2534878, 2533873, 2534458, 2534800, 2534436, 2534911, 2534553, 2534129, 2534378, 2533647, 2533777, 2534187, 2534115, 2533991, 2533706, 2534747, 2534497, 2533745, 2533754, 2534368, 2533718, 2534441, 2533810, 2534696, 2534614, 2534357, 2534465, 2533879, 2534896, 2534795, 2533995, 2534700, 2534481, 2534446, 2534012, 2534702, 2534508, 2534286, 2534862, 2534507, 2534239, 2534181, 2533956, 2533828, 2534403, 2534384, 2533755, 2534710, 2533678, 2534828, 2533788, 2533849, 2534564, 2533860, 2534754, 2534634, 2534278, 2534526, 2533703, 2534213, 2534209, 2534341, 2534428, 2534000, 2534927, 2534466, 2534900, 2533620, 2534676, 2533793, 2534635, 2534852, 2534444, 2534706, 2534016, 2534728, 2534585, 2534130, 2533792, 2534054, 2535360, 2535123, 2535407, 2535404, 2535176, 2535373, 2535061, 2534986, 2534966, 2535081, 2535121, 2534992, 2535084, 2535004, 2535239, 2535115, 2535303, 2535183, 2535044, 2535224, 2535173, 2535071, 2535108, 2534987, 2535050, 2535352, 2535091, 2535406, 2535306, 2535411, 2535111, 2535362, 2535078, 2535141, 2534974, 2535425, 2534957, 2535135, 2535265, 2535395, 2534984, 2535073, 2535358, 2535387, 2535327, 2535253, 2535305, 2535031, 2535105, 2535029, 2534995, 2534949, 2535118, 2535102, 2535092, 2535019, 2534991, 2535415, 2534982, 2535443, 2535332, 2534953, 2534952, 2534983, 2535085, 2535096, 2535388, 2535464, 2535426, 2535072, 2535449, 2535330, 2535089, 2535346, 2535120, 2534939, 2535038, 2535444, 2535280, 2535413, 2535035, 2535439, 2535279, 2535451, 2535440, 2535311, 2534962, 2535049, 2535399, 2535047, 2535140, 2535013, 2535323, 2534946, 2535297, 2535094, 2535113, 2534968, 2535066, 2534958],
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        batch_size=50,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=1000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),

   TrainConfig(
        name="relocate_all_ep300_epoch100_convert_gripper_False",
        num_train_epochs=100.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ),
        data=LeRobotHSRDataConfig(
            repo_id="/mnt/AiroaMomaDataset",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            convert_gripper = False,
            select_episodes = [2536226, 2536013, 2535738, 2536152, 2536393, 2536365, 2536315, 2535724, 2536341, 2535877, 2535988, 2536297, 2536272, 2535961, 2536033, 2536125, 2536049, 2535629, 2535786, 2535759, 2535554, 2535918, 2535497, 2536202, 2536164, 2535763, 2535750, 2535480, 2535563, 2536335, 2536254, 2535551, 2535558, 2535511, 2535906, 2536123, 2535775, 2535841, 2536296, 2536096, 2535739, 2535646, 2536352, 2536184, 2536188, 2536079, 2535783, 2536440, 2536473, 2535997, 2535778, 2535649, 2535650, 2536354, 2536087, 2536012, 2536080, 2536068, 2536133, 2536458, 2535570, 2536376, 2536350, 2536340, 2535652, 2535586, 2536060, 2536454, 2535995, 2535661, 2535859, 2535780, 2536210, 2536106, 2535707, 2535764, 2535903, 2536124, 2535624, 2536100, 2536222, 2535566, 2536346, 2536212, 2535499, 2535647, 2536515, 2535840, 2535745, 2536064, 2536150, 2536146, 2535808, 2536058, 2535581, 2535743, 2536038, 2535482, 2536046, 2535882, 2534305, 2533864, 2534326, 2533614, 2534711, 2534907, 2534026, 2534391, 2534775, 2533898, 2534395, 2533775, 2534577, 2534094, 2534592, 2534869, 2534559, 2533926, 2533855, 2534194, 2534878, 2533873, 2534458, 2534800, 2534436, 2534911, 2534553, 2534129, 2534378, 2533647, 2533777, 2534187, 2534115, 2533991, 2533706, 2534747, 2534497, 2533745, 2533754, 2534368, 2533718, 2534441, 2533810, 2534696, 2534614, 2534357, 2534465, 2533879, 2534896, 2534795, 2533995, 2534700, 2534481, 2534446, 2534012, 2534702, 2534508, 2534286, 2534862, 2534507, 2534239, 2534181, 2533956, 2533828, 2534403, 2534384, 2533755, 2534710, 2533678, 2534828, 2533788, 2533849, 2534564, 2533860, 2534754, 2534634, 2534278, 2534526, 2533703, 2534213, 2534209, 2534341, 2534428, 2534000, 2534927, 2534466, 2534900, 2533620, 2534676, 2533793, 2534635, 2534852, 2534444, 2534706, 2534016, 2534728, 2534585, 2534130, 2533792, 2534054, 2535360, 2535123, 2535407, 2535404, 2535176, 2535373, 2535061, 2534986, 2534966, 2535081, 2535121, 2534992, 2535084, 2535004, 2535239, 2535115, 2535303, 2535183, 2535044, 2535224, 2535173, 2535071, 2535108, 2534987, 2535050, 2535352, 2535091, 2535406, 2535306, 2535411, 2535111, 2535362, 2535078, 2535141, 2534974, 2535425, 2534957, 2535135, 2535265, 2535395, 2534984, 2535073, 2535358, 2535387, 2535327, 2535253, 2535305, 2535031, 2535105, 2535029, 2534995, 2534949, 2535118, 2535102, 2535092, 2535019, 2534991, 2535415, 2534982, 2535443, 2535332, 2534953, 2534952, 2534983, 2535085, 2535096, 2535388, 2535464, 2535426, 2535072, 2535449, 2535330, 2535089, 2535346, 2535120, 2534939, 2535038, 2535444, 2535280, 2535413, 2535035, 2535439, 2535279, 2535451, 2535440, 2535311, 2534962, 2535049, 2535399, 2535047, 2535140, 2535013, 2535323, 2534946, 2535297, 2535094, 2535113, 2534968, 2535066, 2534958],
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        batch_size=50,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=1000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),

   TrainConfig(
        name="relocate_all_ep300_epoch100_false",
        num_train_epochs=100.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ),
        data=LeRobotHSRDataConfig(
            repo_id="/mnt/AiroaMomaDataset",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            select_episodes = [2536226, 2536013, 2535738, 2536152, 2536393, 2536365, 2536315, 2535724, 2536341, 2535877, 2535988, 2536297, 2536272, 2535961, 2536033, 2536125, 2536049, 2535629, 2535786, 2535759, 2535554, 2535918, 2535497, 2536202, 2536164, 2535763, 2535750, 2535480, 2535563, 2536335, 2536254, 2535551, 2535558, 2535511, 2535906, 2536123, 2535775, 2535841, 2536296, 2536096, 2535739, 2535646, 2536352, 2536184, 2536188, 2536079, 2535783, 2536440, 2536473, 2535997, 2535778, 2535649, 2535650, 2536354, 2536087, 2536012, 2536080, 2536068, 2536133, 2536458, 2535570, 2536376, 2536350, 2536340, 2535652, 2535586, 2536060, 2536454, 2535995, 2535661, 2535859, 2535780, 2536210, 2536106, 2535707, 2535764, 2535903, 2536124, 2535624, 2536100, 2536222, 2535566, 2536346, 2536212, 2535499, 2535647, 2536515, 2535840, 2535745, 2536064, 2536150, 2536146, 2535808, 2536058, 2535581, 2535743, 2536038, 2535482, 2536046, 2535882, 2534305, 2533864, 2534326, 2533614, 2534711, 2534907, 2534026, 2534391, 2534775, 2533898, 2534395, 2533775, 2534577, 2534094, 2534592, 2534869, 2534559, 2533926, 2533855, 2534194, 2534878, 2533873, 2534458, 2534800, 2534436, 2534911, 2534553, 2534129, 2534378, 2533647, 2533777, 2534187, 2534115, 2533991, 2533706, 2534747, 2534497, 2533745, 2533754, 2534368, 2533718, 2534441, 2533810, 2534696, 2534614, 2534357, 2534465, 2533879, 2534896, 2534795, 2533995, 2534700, 2534481, 2534446, 2534012, 2534702, 2534508, 2534286, 2534862, 2534507, 2534239, 2534181, 2533956, 2533828, 2534403, 2534384, 2533755, 2534710, 2533678, 2534828, 2533788, 2533849, 2534564, 2533860, 2534754, 2534634, 2534278, 2534526, 2533703, 2534213, 2534209, 2534341, 2534428, 2534000, 2534927, 2534466, 2534900, 2533620, 2534676, 2533793, 2534635, 2534852, 2534444, 2534706, 2534016, 2534728, 2534585, 2534130, 2533792, 2534054, 2535360, 2535123, 2535407, 2535404, 2535176, 2535373, 2535061, 2534986, 2534966, 2535081, 2535121, 2534992, 2535084, 2535004, 2535239, 2535115, 2535303, 2535183, 2535044, 2535224, 2535173, 2535071, 2535108, 2534987, 2535050, 2535352, 2535091, 2535406, 2535306, 2535411, 2535111, 2535362, 2535078, 2535141, 2534974, 2535425, 2534957, 2535135, 2535265, 2535395, 2534984, 2535073, 2535358, 2535387, 2535327, 2535253, 2535305, 2535031, 2535105, 2535029, 2534995, 2534949, 2535118, 2535102, 2535092, 2535019, 2534991, 2535415, 2534982, 2535443, 2535332, 2534953, 2534952, 2534983, 2535085, 2535096, 2535388, 2535464, 2535426, 2535072, 2535449, 2535330, 2535089, 2535346, 2535120, 2534939, 2535038, 2535444, 2535280, 2535413, 2535035, 2535439, 2535279, 2535451, 2535440, 2535311, 2534962, 2535049, 2535399, 2535047, 2535140, 2535013, 2535323, 2534946, 2535297, 2535094, 2535113, 2534968, 2535066, 2534958],
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        batch_size=50,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=10000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),

   TrainConfig(
        name="relocate_all_ep300_epoch100_convert_gripper_False_fix_select_episodes",
        num_train_epochs=100.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ),
        data=LeRobotHSRDataConfig(
            repo_id="/mnt/AiroaMomaDataset",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            convert_gripper = False,
            select_episodes = [2536226, 2536013, 2535738, 2536152, 2536393, 2536365, 2536315, 2535724, 2536341, 2535877, 2535988, 2536297, 2536272, 2535961, 2536033, 2536125, 2536049, 2535629, 2535786, 2535759, 2535554, 2535918, 2535497, 2536202, 2536164, 2535763, 2535750, 2535480, 2535563, 2536335, 2536254, 2535551, 2535558, 2535511, 2535906, 2536123, 2535775, 2535841, 2536296, 2536096, 2535739, 2535646, 2536352, 2536184, 2536188, 2536079, 2535783, 2536440, 2536473, 2535997, 2535778, 2535649, 2535650, 2536354, 2536087, 2536012, 2536080, 2536068, 2536133, 2536458, 2535570, 2536376, 2536350, 2536340, 2535652, 2535586, 2536060, 2536454, 2535995, 2535661, 2535859, 2535780, 2536210, 2536106, 2535707, 2535764, 2535903, 2536124, 2535624, 2536100, 2536222, 2535566, 2536346, 2536212, 2535499, 2535647, 2536515, 2535840, 2535745, 2536064, 2536150, 2536146, 2535808, 2536058, 2535581, 2535743, 2536038, 2535482, 2536046, 2535882, 2534305, 2533864, 2534326, 2533614, 2534711, 2534907, 2534026, 2534391, 2534775, 2533898, 2534395, 2533775, 2534577, 2534094, 2534592, 2534869, 2534559, 2533926, 2533855, 2534194, 2534878, 2533873, 2534458, 2534800, 2534436, 2534911, 2534553, 2534129, 2534378, 2533647, 2533777, 2534187, 2534115, 2533991, 2533706, 2534747, 2534497, 2533745, 2533754, 2534368, 2533718, 2534441, 2533810, 2534696, 2534614, 2534357, 2534465, 2533879, 2534896, 2534795, 2533995, 2534700, 2534481, 2534446, 2534012, 2534702, 2534508, 2534286, 2534862, 2534507, 2534239, 2534181, 2533956, 2533828, 2534403, 2534384, 2533755, 2534710, 2533678, 2534828, 2533788, 2533849, 2534564, 2533860, 2534754, 2534634, 2534278, 2534526, 2533703, 2534213, 2534209, 2534341, 2534428, 2534000, 2534927, 2534466, 2534900, 2533620, 2534676, 2533793, 2534635, 2534852, 2534444, 2534706, 2534016, 2534728, 2534585, 2534130, 2533792, 2534054, 2535360, 2535123, 2535407, 2535404, 2535176, 2535373, 2535061, 2534986, 2534966, 2535081, 2535121, 2534992, 2535084, 2535004, 2535239, 2535115, 2535303, 2535183, 2535044, 2535224, 2535173, 2535071, 2535108, 2534987, 2535050, 2535352, 2535091, 2535406, 2535306, 2535411, 2535111, 2535362, 2535078, 2535141, 2534974, 2535425, 2534957, 2535135, 2535265, 2535395, 2534984, 2535073, 2535358, 2535387, 2535327, 2535253, 2535305, 2535031, 2535105, 2535029, 2534995, 2534949, 2535118, 2535102, 2535092, 2535019, 2534991, 2535415, 2534982, 2535443, 2535332, 2534953, 2534952, 2534983, 2535085, 2535096, 2535388, 2535464, 2535426, 2535072, 2535449, 2535330, 2535089, 2535346, 2535120, 2534939, 2535038, 2535444, 2535280, 2535413, 2535035, 2535439, 2535279, 2535451, 2535440, 2535311, 2534962, 2535049, 2535399, 2535047, 2535140, 2535013, 2535323, 2534946, 2535297, 2535094, 2535113, 2534968, 2535066, 2534958],
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        batch_size=150,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=1000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),

   TrainConfig(
        name="pi05_ep50_epoch20_curation_pick_up_a_slice_of_bread_on_the_plate",
        num_train_epochs=20.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotHSRDataConfig(
            repo_id="airoa-moma",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            select_episodes = [721, 727, 758, 844, 897, 904, 935, 956, 1183, 1206, 1299, 1404, 1445, 1453, 1476, 1484, 1542, 1556, 1594, 1698, 1786, 1815, 1852, 1856, 1872, 1924, 1966, 1969, 2014, 2121, 2133, 2492, 2515, 2559, 2609, 2716, 2742, 2750, 2764, 2799, 2836, 3474, 3509, 3541, 3548, 3633, 3696, 3720, 3758, 3830],
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        batch_size=32,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=10000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),
   TrainConfig(
        name="pi05_wakamatsu_ct_pro",
        num_train_epochs=100.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ),
        data=LeRobotHSRDataConfig(
            repo_id="/mnt/share_for_HPC/yano21/wakamatsu_ct_pro",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            select_episodes = None,
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        batch_size=150,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=5000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),
   TrainConfig(
        name="pi05_wakamatsu_ct_pro_norm_stats",
        num_train_epochs=100.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ),
        data=LeRobotHSRDataConfig(
            repo_id="/mnt/share_for_HPC/yano21/wakamatsu_ct_pro",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            select_episodes = None,
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        batch_size=50,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=5000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),
   TrainConfig(
        name="pi05_wakamatsu_ct_pro_convert_gripper_true",
        num_train_epochs=100.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ),
        data=LeRobotHSRDataConfig(
            repo_id="/mnt/share_for_HPC/yano21/wakamatsu_ct_pro",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            convert_gripper = True,
            select_episodes = None,
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        batch_size=150,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=1000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),
   TrainConfig(
        name="pi05_wakamatsu_ct_pro_old_hsr_policy",
        num_train_epochs=100.0,  
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ),
        data=LeRobotHSRDataConfig(
            repo_id="/mnt/share_for_HPC/yano21/wakamatsu_ct_pro",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            convert_gripper = True,
            select_episodes = None,
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=60_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        batch_size=150,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=1000, # Save more frequently since the dataset is smaller.
        pytorch_weight_path="/mnt/share_for_HPC/hyamaguchi23/checkpoints",
    ),
    TrainConfig(
        name="curation_pick_up_a_slice_of_bread_on_the_plate_",
        num_train_epochs=10.0, 
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotHSRDataConfig(
            repo_id="airoa-moma",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            select_episodes = [0, 57, 78, 84, 92, 94, 98, 141, 147, 165, 173, 194, 208, 258, 273, 283, 286, 296, 298, 319, 341, 355, 377, 383, 386, 401, 420, 424, 427, 447, 464, 471, 477, 485, 486, 508, 532, 546, 549, 565, 569, 579, 628, 630, 639, 640, 649, 654, 692, 695],
            assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=64,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=10000, # Save more frequently since the dataset is smaller.
    ),
    TrainConfig(
        name="pi05_lora_ep50_all",
        num_train_epochs=5.0, 
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotHSRDataConfig(
            repo_id="airoa-moma",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            select_episodes = [84, 160, 194, 196, 255, 314, 317, 335, 386, 427, 429, 452, 464, 480, 508, 535, 594, 604, 640, 692, 770, 848, 849, 862, 897, 901, 902, 973, 983, 1003, 1060, 1068, 1122, 1224, 1323, 1392, 1434, 1507, 1555, 1618, 1769, 1842, 1856, 1911, 2022, 2061, 2066, 2121, 2148, 2175, 2398, 2445, 2465, 2560, 2605, 2618, 2668, 2700, 2728, 2741, 2742, 2768, 2833, 2996, 2997, 3037, 3099, 3181, 3339, 3352, 3515, 3517, 3548, 3550, 3570, 3611, 3629, 3637, 3645, 3696, 3986, 3999, 4128, 4157, 4172, 4185, 4225, 4228, 4268, 4277, 4318, 4342, 4343, 4355, 4358, 4360, 4363, 4377, 4397, 4410, 4434, 4445, 4452, 4484, 4491, 4545, 4579, 4686, 4708, 4820, 4853, 4946, 4953, 5112, 5130, 5172, 5200, 5285, 5334, 5344, 5374, 5395, 5473, 5486, 5514, 5564, 5579, 5598, 5606, 5617, 5641, 5648, 5701, 5706, 5720, 5769, 5802, 5847, 5927, 5953, 5987, 6101, 6135, 6149, 6221, 6235, 6239, 6265, 6297, 6352, 6478, 6509, 6548, 6579, 6667, 6668, 6695, 6725, 6757, 6760, 6774, 6822, 6864, 6883, 6937, 7060, 7087, 7195, 7255, 7262, 7280, 7314, 7318, 7331, 7341, 7361, 7381, 7415, 7418, 7427, 7453, 7468, 7532, 7534, 7602, 7700, 7701, 8043, 8150, 8194, 8220, 8495, 8615, 8638, 8648, 8779, 8805, 8864, 8874, 8960, 8961, 8995, 9004, 9048, 9089, 9150, 9153, 9290, 9309, 9419, 9513, 9554, 9573, 9577, 9692, 9708, 9719, 9804, 9809, 9831, 9852, 9938, 9957, 10005, 10025, 10054, 10095, 10101, 10122, 10130, 10159, 10223, 10234, 10254, 10277, 10290, 10310, 10315, 10379, 10405, 10443, 10444, 10503, 10519, 10553, 10557, 10560, 10570, 10571, 10589, 10653, 10661, 10676, 10691, 10695, 10725, 10747, 10877, 10903, 10907, 10937, 10985, 11086, 11132, 11147, 11179, 11184, 11193, 11239, 11249, 11257, 11289, 11303, 11309, 11321, 11340, 11356, 11403, 11408, 11421, 11423, 11431, 11435, 11471, 11473, 11503, 11567, 11570, 11601, 11636, 11653, 11662, 11670, 11676, 11678, 11711, 11723, 11735, 11746, 11755, 11795, 11799, 11802, 11910, 12108, 12146, 12157, 12201, 12246, 12252, 12274, 12287, 12301, 12306, 12319, 12450, 12495, 12518, 12537, 12542, 12564, 12616, 12627, 12645, 12667, 12669, 12700, 12709, 12723, 12751, 12822, 12886, 12967, 12979, 13034, 13055, 13070, 13111, 13141, 13215, 13281, 13306, 13418, 13445, 13517, 13550, 13555, 13581, 13587, 13603, 13637, 13677, 13714, 13801, 13802, 13806, 13861, 13864, 13866, 13885, 13949, 14016, 14068, 14292, 14322, 14395, 14534, 14540, 14893, 15055, 15108, 15130, 15397, 15667, 15761, 15775, 16083, 16085, 16342, 16505, 17377, 17379, 17481, 17699, 17712, 17730, 17816, 17880, 17943, 18001, 18044, 18362, 18477, 18507, 18710, 18884, 18936, 19060, 19095, 19375, 19529, 19585, 19761, 19941, 20154, 20301, 20411, 20553, 20722, 20748, 20789, 20846, 20848, 21003, 21158, 21229, 21235, 21267, 21307, 21481, 21580, 21614, 21627, 21659, 21739, 21789, 21801, 21859, 21861, 21863, 21886, 21888, 21894, 21910, 21919, 21949, 21981, 22019, 22037, 22050, 22068, 22070, 22109, 22112, 22161, 22190, 22231, 22295, 22420, 22487, 22618, 22644, 22761, 22921, 22928, 22946, 22974, 23033, 23099, 23207, 23211, 23253, 23320, 23354, 23361, 23400, 23580, 23582, 23645, 23724],
                assets=AssetsConfig(
                asset_id="hsr",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=40_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        batch_size=64,
        num_workers=2, # Increase num_workers to speed up data loading with larger datasets.
        save_interval=10000, # Save more frequently since the dataset is smaller.
    ),


    TrainConfig(
        name="pi0_task8",
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotHSRDataConfig(
            repo_id="lerobot_datasets/task8",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule( # batch 512
            warmup_steps=1_000,
            peak_lr=1.0e-4,    # 2.5e-5 × √16 = 1.0e-4
            decay_steps=80_000,
            decay_lr=1.0e-5,   # 2.5e-6 × √16 = 1.0e-5
        ),
        batch_size=512,
        num_workers=16,
        num_train_steps=80_000,
        # lr_schedule=_optimizer.CosineDecaySchedule( # batch 128
        #     warmup_steps=1_000,
        #     peak_lr=5.0e-5,     # 2.5e-5 × 2 = 5.0e-5
        #     decay_steps=320_000,  # Match num_train_steps.
        #     decay_lr=5.0e-6,    # 2.5e-6 × 2 = 5.0e-6
        # ),
        # batch_size=128,
        # num_workers=4,
        # num_train_steps=320_000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_task8",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotHSRDataConfig(
            repo_id="lerobot_datasets/task8",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_task8",
                asset_id="lerobot_datasets/task8",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule( # batch 512
            warmup_steps=1_000,
            peak_lr=1.0e-4,    # 2.5e-5 × √16 = 1.0e-4
            decay_steps=80_000,
            decay_lr=1.0e-5,   # 2.5e-6 × √16 = 1.0e-5
        ),
        batch_size=512,
        num_workers=16,
        num_train_steps=80_000,
        # lr_schedule=_optimizer.CosineDecaySchedule( # batch 128
        #     warmup_steps=1_000,
        #     peak_lr=5.0e-5,     # 2.5e-5 × 2 = 5.0e-5
        #     decay_steps=320_000,  # Match num_train_steps.
        #     decay_lr=5.0e-6,    # 2.5e-6 × 2 = 5.0e-6
        # ),
        # batch_size=128,
        # num_workers=4,
        # num_train_steps=320_000,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
    #
    # Sample checkpoint config
    #
    TrainConfig(
        name="pi05_hsr_task6891011_level12_v2.5_train_adaptive",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotHSRDataConfig(
            repo_id="lerobot_datasets/task6891011_level12_v2.5_train",
            assets=AssetsConfig(
                assets_dir="./assets/pi05_hsr_task6891011_level12_v2.5_train_adaptive",
                asset_id="lerobot_datasets/task6891011_level12_v2.5_train",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule( # batch 64
            warmup_steps=1_000,
            peak_lr=3.5e-5,     # 2.5e-5 × √2 = 3.5e-5
            decay_steps=1_300_000,  # Match num_train_steps.
            decay_lr=3.5e-6,    # 2.5e-6 × √2 = 3.5e-6
        ),
        batch_size=64,
        num_workers=8,
        num_train_steps=1_300_000,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
