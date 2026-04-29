from __future__ import annotations

import collections
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

try:
    from PIL import Image
except ImportError:  # pragma: no cover - pillow is expected to be available.
    Image = None

from lerobot.common.datasets.utils import check_delta_timestamps, get_delta_indices
from lerobot.common.datasets.video_utils import decode_video_frames, get_safe_default_codec

_INFO_PATH = Path("meta/info.json")
_TASKS_PATH = Path("meta/tasks.jsonl")
_EPISODE_INDEX_CACHE = Path("meta/openpi_episode_index_v1.npz")
_DEFAULT_EPISODE_CACHE_SIZE = int(os.getenv("OPENPI_LEROBOT_EPISODE_CACHE_SIZE", "8"))

_NP_DTYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "bfloat16": np.float32,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "bool": np.bool_,
}

_TORCH_DTYPE_MAP = {
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
    np.dtype(np.int8): torch.int8,
    np.dtype(np.int16): torch.int16,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.bool_): torch.bool,
}


def is_local_lerobot_dataset_root(path_like: str | Path | None) -> bool:
    if path_like is None:
        return False
    path = Path(path_like).expanduser()
    return path.exists() and path.is_dir() and (path / _INFO_PATH).is_file()


class LocalLeRobotMetadata:
    """Lightweight metadata loader for local LeRobot v2.x datasets.

    The stock LeRobot v2.1 metadata loader eagerly reads ``meta/episodes_stats.jsonl``
    and aggregates it. For large real-robot datasets this can exceed host memory.
    This class intentionally loads only the small metadata required for training:
    ``meta/info.json`` and ``meta/tasks.jsonl``.
    """

    def __init__(self, root: str | Path, *, repo_id: str | None = None):
        self.root = Path(root).expanduser().resolve()
        self.repo_id = repo_id or str(self.root)
        self.info = self._load_info()
        self.tasks = self._load_tasks()
        self.task_to_task_index = {task: task_index for task_index, task in self.tasks.items()}

    def _load_info(self) -> dict[str, Any]:
        with (self.root / _INFO_PATH).open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_tasks(self) -> dict[int, str]:
        tasks_path = self.root / _TASKS_PATH
        if not tasks_path.is_file():
            return {}

        tasks: dict[int, str] = {}
        with tasks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                task_index = int(row["task_index"])
                tasks[task_index] = str(row["task"])
        return tasks

    @property
    def data_path(self) -> str:
        return str(self.info["data_path"])

    @property
    def video_path(self) -> str | None:
        return self.info.get("video_path")

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        return self.info["features"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft.get("dtype") == "video"]

    @property
    def image_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft.get("dtype") == "image"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft.get("dtype") in {"video", "image"}]

    @property
    def total_episodes(self) -> int:
        return int(self.info["total_episodes"])

    @property
    def total_frames(self) -> int:
        return int(self.info["total_frames"])

    @property
    def chunks_size(self) -> int:
        return int(self.info["chunks_size"])

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    def _format_path(self, template: str, *, ep_index: int, video_key: str | None = None) -> Path:
        episode_chunk = self.get_episode_chunk(ep_index)
        format_kwargs = {
            "episode_index": ep_index,
            "episode_chunk": episode_chunk,
            "chunk_index": episode_chunk,
            "chunk_idx": episode_chunk,
            "file_index": ep_index,
        }
        if video_key is not None:
            format_kwargs["video_key"] = video_key
        return Path(template.format(**format_kwargs))

    def get_data_file_path(self, ep_index: int) -> Path:
        return self._format_path(self.data_path, ep_index=ep_index)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        if self.video_path is None:
            raise ValueError("This dataset does not define a video_path in info.json")
        return self._format_path(self.video_path, ep_index=ep_index, video_key=vid_key)


class LocalLeRobotDataset(torch.utils.data.Dataset):
    """LeRobot-compatible local dataset without Hugging Face Dataset eager loading.

    The stock ``LeRobotDataset`` uses ``datasets.load_dataset("parquet", ...)`` which
    is convenient but can spend a long time constructing Arrow state on a large shared
    filesystem, and it still eagerly materializes a fair amount of metadata. This class:

    1. reads only ``meta/info.json`` and ``meta/tasks.jsonl`` at startup,
    2. builds a compact per-episode frame index from Parquet footers,
    3. loads one episode Parquet file at a time on demand,
    4. keeps only a tiny LRU cache of recent episode tables in host RAM.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms=None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        del revision, force_cache_sync, download_videos
        super().__init__()

        self.repo_id = repo_id
        self.root = Path(root).expanduser().resolve() if root is not None else Path(repo_id).expanduser().resolve()
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.meta = LocalLeRobotMetadata(self.root, repo_id=repo_id)
        self.stats = None
        self._episode_table_cache: collections.OrderedDict[int, Any] = collections.OrderedDict()
        self._episode_cache_size = max(1, _DEFAULT_EPISODE_CACHE_SIZE)
        self._logged_episode_loads: set[int] = set()
        self._logged_video_decodes: set[tuple[int, str]] = set()

        self._feature_np_dtypes = self._build_feature_dtype_map()
        self._full_episode_data_index = self._load_or_build_episode_index()
        self.episode_data_index = self._build_selected_episode_index()

        self.delta_indices = None
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        logging.info(
            "LocalLeRobotDataset ready: root=%s selected_episodes=%d total_frames=%d video_keys=%s image_keys=%s",
            self.root,
            self.num_episodes,
            self.num_frames,
            self.meta.video_keys,
            self.meta.image_keys,
        )

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        return int(self._num_frames)

    @property
    def num_episodes(self) -> int:
        return len(self._episode_ids)

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        return self.meta.features

    def _build_feature_dtype_map(self) -> dict[str, np.dtype]:
        feature_np_dtypes: dict[str, np.dtype] = {
            "index": np.dtype(np.int64),
            "episode_index": np.dtype(np.int64),
            "frame_index": np.dtype(np.int64),
            "task_index": np.dtype(np.int64),
            "timestamp": np.dtype(np.float32),
        }
        for key, feature in self.meta.features.items():
            dtype_name = str(feature.get("dtype", "")).lower()
            if dtype_name in _NP_DTYPE_MAP:
                feature_np_dtypes[key] = np.dtype(_NP_DTYPE_MAP[dtype_name])
        return feature_np_dtypes

    def _load_or_build_episode_index(self) -> dict[str, torch.Tensor]:
        cache_path = self.root / _EPISODE_INDEX_CACHE
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cache:
                    if (
                        int(cache["total_episodes"]) == self.meta.total_episodes
                        and int(cache["total_frames"]) == self.meta.total_frames
                    ):
                        logging.info("Loaded compact episode index cache from %s", cache_path)
                        return {
                            "from": torch.from_numpy(cache["from_index"].astype(np.int64, copy=False)),
                            "to": torch.from_numpy(cache["to_index"].astype(np.int64, copy=False)),
                        }
            except Exception as exc:
                logging.warning("Ignoring invalid episode index cache at %s: %s", cache_path, exc)

        logging.info(
            "Building compact local LeRobot episode index from Parquet metadata for %s (episodes=%d)",
            self.root,
            self.meta.total_episodes,
        )
        from_index = np.zeros(self.meta.total_episodes, dtype=np.int64)
        to_index = np.zeros(self.meta.total_episodes, dtype=np.int64)
        cursor = 0
        for ep_idx in range(self.meta.total_episodes):
            parquet_path = self.root / self.meta.get_data_file_path(ep_idx)
            num_rows = int(pq.ParquetFile(parquet_path).metadata.num_rows)
            from_index[ep_idx] = cursor
            cursor += num_rows
            to_index[ep_idx] = cursor

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            total_episodes=np.asarray(self.meta.total_episodes, dtype=np.int64),
            total_frames=np.asarray(self.meta.total_frames, dtype=np.int64),
            from_index=from_index,
            to_index=to_index,
        )

        if cursor != self.meta.total_frames:
            logging.warning(
                "Local LeRobot frame count from Parquet metadata (%d) does not match info.json total_frames (%d).",
                cursor,
                self.meta.total_frames,
            )

        return {
            "from": torch.from_numpy(from_index),
            "to": torch.from_numpy(to_index),
        }

    def _build_selected_episode_index(self) -> dict[str, torch.Tensor]:
        if self.episodes is None:
            self._episode_ids = np.arange(self.meta.total_episodes, dtype=np.int64)
        else:
            self._episode_ids = np.asarray(self.episodes, dtype=np.int64)
            if self._episode_ids.ndim != 1:
                raise ValueError("episodes must be a 1D list of episode indices")

        full_from = self._full_episode_data_index["from"].numpy()
        full_to = self._full_episode_data_index["to"].numpy()
        selected_from = np.zeros(len(self._episode_ids), dtype=np.int64)
        selected_to = np.zeros(len(self._episode_ids), dtype=np.int64)
        cursor = 0
        for pos, ep_idx in enumerate(self._episode_ids.tolist()):
            if ep_idx < 0 or ep_idx >= self.meta.total_episodes:
                raise IndexError(f"Episode index {ep_idx} out of bounds for total_episodes={self.meta.total_episodes}")
            ep_len = int(full_to[ep_idx] - full_from[ep_idx])
            selected_from[pos] = cursor
            cursor += ep_len
            selected_to[pos] = cursor

        self._selected_from = selected_from
        self._selected_to = selected_to
        self._episode_id_to_pos = {int(ep_idx): pos for pos, ep_idx in enumerate(self._episode_ids.tolist())}
        self._num_frames = int(cursor)
        return {"from": torch.from_numpy(selected_from), "to": torch.from_numpy(selected_to)}

    def _get_episode_table(self, ep_idx: int):
        if ep_idx in self._episode_table_cache:
            table = self._episode_table_cache.pop(ep_idx)
            self._episode_table_cache[ep_idx] = table
            return table

        parquet_path = self.root / self.meta.get_data_file_path(ep_idx)
        if ep_idx not in self._logged_episode_loads:
            logging.info("Loading local episode parquet into cache: episode=%d path=%s", ep_idx, parquet_path)
            self._logged_episode_loads.add(ep_idx)
        table = pq.read_table(parquet_path, memory_map=True, use_threads=False)
        self._episode_table_cache[ep_idx] = table
        while len(self._episode_table_cache) > self._episode_cache_size:
            self._episode_table_cache.popitem(last=False)
        return table

    def _find_episode_for_index(self, idx: int) -> tuple[int, int, int]:
        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Index {idx} out of bounds for dataset length {self.num_frames}")
        ep_pos = int(np.searchsorted(self._selected_to, idx, side="right"))
        ep_idx = int(self._episode_ids[ep_pos])
        local_idx = int(idx - self._selected_from[ep_pos])
        return ep_pos, ep_idx, local_idx

    def _read_cell(self, ep_idx: int, key: str, local_idx: int):
        table = self._get_episode_table(ep_idx)
        if key not in table.column_names:
            raise KeyError(f"Column '{key}' not found in parquet table for episode {ep_idx}. Available columns: {table.column_names}")
        return table.column(key)[local_idx].as_py()

    def _to_torch_value(self, key: str, value: Any):
        if isinstance(value, torch.Tensor):
            return value
        if value is None:
            return value
        if key in self.meta.image_keys:
            return self._decode_image_payload(value)
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value
        if isinstance(value, dict) and ("bytes" in value or "path" in value or "byte" in value):
            return value

        np_dtype = self._feature_np_dtypes.get(key)
        if np_dtype is not None:
            array_value = np.asarray(value, dtype=np_dtype)
            if array_value.ndim == 0:
                torch_dtype = _TORCH_DTYPE_MAP.get(np_dtype)
                if torch_dtype is not None:
                    return torch.tensor(array_value.item(), dtype=torch_dtype)
                return torch.tensor(array_value.item())
            contiguous = np.ascontiguousarray(array_value)
            return torch.from_numpy(contiguous)

        if isinstance(value, (np.ndarray, np.generic, list, tuple, int, float, bool)):
            return torch.as_tensor(value)

        return value

    def _decode_image_payload(self, payload: Any) -> torch.Tensor:
        if torch.is_tensor(payload):
            return payload
        if isinstance(payload, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(payload))
        if Image is None:
            raise RuntimeError("Pillow is required to decode image features from the local LeRobot dataset")

        image = None
        if isinstance(payload, dict):
            raw_bytes = payload.get("bytes") or payload.get("byte")
            path = payload.get("path")
            if raw_bytes is not None:
                image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            elif path:
                image_path = Path(path)
                if not image_path.is_absolute():
                    image_path = self.root / image_path
                image = Image.open(image_path).convert("RGB")
        elif isinstance(payload, str):
            image_path = Path(payload)
            if not image_path.is_absolute():
                image_path = self.root / image_path
            image = Image.open(image_path).convert("RGB")

        if image is None:
            raise ValueError(f"Unsupported image payload type for local LeRobot loader: {type(payload)}")

        return torch.from_numpy(np.ascontiguousarray(np.array(image)))

    def _get_query_indices(self, idx: int, ep_pos: int) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        ep_start = int(self._selected_from[ep_pos])
        ep_end = int(self._selected_to[ep_pos])
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor([(idx + delta < ep_start) or (idx + delta >= ep_end) for delta in delta_idx])
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_columns(self, ep_idx: int, ep_pos: int, query_indices: dict[str, list[int]]) -> dict[str, torch.Tensor]:
        query_result: dict[str, torch.Tensor] = {}
        ep_start = int(self._selected_from[ep_pos])
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            stacked = [self._to_torch_value(key, self._read_cell(ep_idx, key, int(global_i - ep_start))) for global_i in q_idx]
            query_result[key] = torch.stack(stacked)
        return query_result

    def _get_query_timestamps(
        self,
        current_ts: float,
        ep_idx: int,
        ep_pos: int,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps: dict[str, list[float]] = {}
        ep_start = int(self._selected_from[ep_pos])
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                query_timestamps[key] = [
                    float(self._read_cell(ep_idx, "timestamp", int(global_i - ep_start))) for global_i in query_indices[key]
                ]
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            log_key = (ep_idx, vid_key)
            if log_key not in self._logged_video_decodes:
                logging.info(
                    "Decoding local episode video frames: episode=%d key=%s path=%s num_timestamps=%d backend=%s",
                    ep_idx,
                    vid_key,
                    video_path,
                    len(query_ts),
                    self.video_backend,
                )
                self._logged_video_decodes.add(log_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)
        return item

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_pos, ep_idx, local_idx = self._find_episode_for_index(int(idx))
        table = self._get_episode_table(ep_idx)
        item = {key: self._to_torch_value(key, table.column(key)[local_idx].as_py()) for key in table.column_names}

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(int(idx), ep_pos)
            query_result = self._query_columns(ep_idx, ep_pos, query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts_raw = item.get("timestamp")
            current_ts = float(current_ts_raw.item()) if isinstance(current_ts_raw, torch.Tensor) else float(current_ts_raw)
            query_timestamps = self._get_query_timestamps(current_ts, ep_idx, ep_pos, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**item, **video_frames}

        if self.image_transforms is not None:
            for cam_key in self.meta.camera_keys:
                if cam_key in item:
                    item[cam_key] = self.image_transforms(item[cam_key])

        if "task_index" in item and int(item["task_index"].item()) in self.meta.tasks:
            item["task"] = self.meta.tasks[int(item["task_index"].item())]

        return item
       
