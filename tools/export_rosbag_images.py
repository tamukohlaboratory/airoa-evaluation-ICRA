#!/usr/bin/env python3
"""Export RGB camera frames from a ROS 2 bag."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image


DEFAULT_TOPICS = {
    "head": "/head_rgbd_sensor/rgb/image_rect_color",
    "head_compressed": "/head_rgbd_sensor/rgb/image_rect_color/compressed",
    "hand": "/hand_camera/image_raw",
    "hand_compressed": "/hand_camera/image_raw/compressed",
}


def topic_to_dir_name(topic: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", topic.strip("/")) or "topic"


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    encoding = str(msg.encoding).lower()
    bytes_per_pixel_map = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
        "yuv422": 2,
        "yuv422_yuy2": 2,
        "yuv422_yuyv": 2,
        "yuyv": 2,
    }
    bytes_per_pixel = bytes_per_pixel_map.get(encoding)
    if bytes_per_pixel is None:
        raise ValueError(f"unsupported image encoding: {msg.encoding}")

    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    used_row_bytes = width * bytes_per_pixel
    expected = height * step
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size < expected:
        raise ValueError(f"image payload too small: {data.size} < {expected}")
    if step < used_row_bytes:
        raise ValueError(f"image step too small: {step} < {used_row_bytes}")

    image = data[:expected].reshape(height, step)[:, :used_row_bytes]
    if bytes_per_pixel == 1:
        image = image.reshape(height, width)
    else:
        image = image.reshape(height, width, bytes_per_pixel)

    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "bgr8":
        return np.array(image)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding == "mono8":
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    yuy2_code = getattr(cv2, "COLOR_YUV2BGR_YUY2", None) or getattr(cv2, "COLOR_YUV2BGR_YUYV", None)
    if yuy2_code is None:
        raise ValueError("OpenCV build cannot convert YUY2 images")
    return cv2.cvtColor(image, yuy2_code)


def compressed_msg_to_bgr(msg: CompressedImage) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cv2.imdecode returned None")
    return image


def make_reader(bag_path: Path) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    return reader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="Path to rosbag2 directory")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Directory for exported images")
    parser.add_argument(
        "--topic",
        action="append",
        dest="topics",
        help="Topic to export. May be repeated. Defaults to B022 head/hand raw and compressed RGB topics.",
    )
    parser.add_argument("--format", choices=("png", "jpg"), default="png", help="Output image format")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames per topic; 0 means all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wanted_topics = set(args.topics or DEFAULT_TOPICS.values())
    reader = make_reader(args.bag)
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    available_targets = {topic: topic_types[topic] for topic in wanted_topics if topic in topic_types}

    if not available_targets:
        print("No requested image topics found in this bag.")
        print("Requested:")
        for topic in sorted(wanted_topics):
            print(f"  {topic}")
        print("Image-like topics present:")
        for topic, msg_type in sorted(topic_types.items()):
            if "Image" in msg_type or "image" in topic.lower() or "camera" in topic.lower():
                print(f"  {topic} [{msg_type}]")
        return 2

    msg_classes: dict[str, Any] = {topic: get_message(msg_type) for topic, msg_type in available_targets.items()}
    counts = {topic: 0 for topic in available_targets}
    errors: dict[str, int] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        if topic not in available_targets:
            continue
        if args.max_frames > 0 and counts[topic] >= args.max_frames:
            continue

        msg = deserialize_message(data, msg_classes[topic])
        try:
            if isinstance(msg, Image):
                image = image_msg_to_bgr(msg)
            elif isinstance(msg, CompressedImage):
                image = compressed_msg_to_bgr(msg)
            else:
                continue
        except Exception as exc:
            errors[topic] = errors.get(topic, 0) + 1
            print(f"[WARN] {topic}: {exc}")
            continue

        topic_dir = args.output_dir / topic_to_dir_name(topic)
        topic_dir.mkdir(parents=True, exist_ok=True)
        frame_idx = counts[topic]
        filename = topic_dir / f"{frame_idx:06d}_{timestamp_ns}.{args.format}"
        cv2.imwrite(str(filename), image)
        counts[topic] += 1

    print("Exported frames:")
    for topic in sorted(available_targets):
        print(f"  {topic}: {counts[topic]} frames")
    if errors:
        print("Decode errors:")
        for topic, count in sorted(errors.items()):
            print(f"  {topic}: {count}")
    return 0 if any(counts.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
