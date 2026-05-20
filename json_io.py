from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ik import Pose6D
from obstacles import Obstacle


MM_TO_M = 0.001


@dataclass(frozen=True)
class PlanningRequest:
    # 从 path/inputs/*.json 读出的规划输入。
    # start_joint_angles 是 RRT 的起点关节角，单位 rad。
    # goal_pose 是抓取点 6D 位姿，位置统一转换成 m，姿态保持 rad。
    start_joint_angles: np.ndarray
    goal_pose: Pose6D
    pre_grasp_pose: Pose6D | None = None


@dataclass(frozen=True)
class SceneModel:
    # 从 path/scenes/*.json 读出的障碍物建模结果。
    # obstacles 中所有尺寸和位置都已经转换成米，后续 IK/RRT 不再关心 JSON 原单位。
    obstacles: list[Obstacle]
    source_path: Path
    position_unit: str


def load_planning_request(path: Path) -> PlanningRequest:
    """Load start joints and goal pose from path/inputs JSON."""
    data = _load_json(path)

    if "start_joint_angles_rad" not in data:
        raise ValueError(f"{path} missing start_joint_angles_rad")
    if "goal_pose_6d" not in data:
        raise ValueError(f"{path} missing goal_pose_6d")

    start_joint_angles = np.asarray(data["start_joint_angles_rad"], dtype=float)
    goal_pose = _pose_from_json(data["goal_pose_6d"], position_unit=_request_position_unit(data))
    pre_grasp_pose = None
    if "pre_grasp_pose_6d" in data:
        pre_grasp_pose = _pose_from_json(data["pre_grasp_pose_6d"], position_unit=_request_position_unit(data))

    return PlanningRequest(
        start_joint_angles=start_joint_angles,
        goal_pose=goal_pose,
        pre_grasp_pose=pre_grasp_pose,
    )


def load_scene_model(path: Path) -> SceneModel:
    """Load supported obstacles from path/scenes JSON."""
    data = _load_json(path)
    position_unit = _scene_position_unit(data)
    obstacles: list[Obstacle] = []

    for index, item in enumerate(data.get("obstacles", [])):
        obstacle_type = str(item.get("type", "")).lower()
        obstacle_id = item.get("id") or f"obstacle_{index}"
        safety_margin = _length_to_m(float(item.get("safety_margin", 0.0)), position_unit)

        if obstacle_type == "box":
            obstacles.append(_box_from_json(item, position_unit, safety_margin, obstacle_id))
        elif obstacle_type == "sphere":
            obstacles.append(_sphere_from_json(item, position_unit, safety_margin, obstacle_id))
        else:
            raise ValueError(f"{path} obstacle {obstacle_id!r} has unsupported type {obstacle_type!r}")

    return SceneModel(obstacles=obstacles, source_path=path, position_unit=position_unit)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _request_position_unit(data: dict[str, Any]) -> str:
    # path/inputs 现有样例没有显式写单位；按项目约定默认位置是 millimeters。
    value_units = data.get("value_units", {})
    return str(value_units.get("position_xyz", data.get("position_unit", "millimeters"))).lower()


def _scene_position_unit(data: dict[str, Any]) -> str:
    reference_frame = data.get("reference_frame", {})
    value_units = data.get("value_units", {})
    return str(
        value_units.get(
            "position_xyz",
            reference_frame.get("position_unit", "millimeters"),
        )
    ).lower()


def _pose_from_json(raw_pose: dict[str, Any], position_unit: str) -> Pose6D:
    # 6D 位姿 = x/y/z 三个位置 + roll/pitch/yaw 三个姿态角。
    # JSON 里的 x/y/z 通常是 mm，这里统一转成内部使用的 m。
    return Pose6D(
        x=_length_to_m(float(raw_pose["x"]), position_unit),
        y=_length_to_m(float(raw_pose["y"]), position_unit),
        z=_length_to_m(float(raw_pose["z"]), position_unit),
        roll=float(raw_pose.get("roll", 0.0)),
        pitch=float(raw_pose.get("pitch", 0.0)),
        yaw=float(raw_pose.get("yaw", 0.0)),
    )


def _pose_from_center_xyz(item: dict[str, Any], position_unit: str) -> Pose6D:
    center = item.get("center_xyz")
    if center is None:
        raise ValueError(f"obstacle {item.get('id', '<unknown>')!r} missing center_xyz")
    roll, pitch, yaw = _orientation_from_json(item)
    return Pose6D(
        x=_length_to_m(float(center[0]), position_unit),
        y=_length_to_m(float(center[1]), position_unit),
        z=_length_to_m(float(center[2]), position_unit),
        roll=roll,
        pitch=pitch,
        yaw=yaw,
    )


def _orientation_from_json(item: dict[str, Any]) -> tuple[float, float, float]:
    # 障碍物姿态如果没写，默认和基坐标系对齐。
    pose_6d = item.get("pose_6d")
    if isinstance(pose_6d, dict):
        return (
            float(pose_6d.get("roll", 0.0)),
            float(pose_6d.get("pitch", 0.0)),
            float(pose_6d.get("yaw", 0.0)),
        )
    rpy = item.get("rpy") or item.get("orientation_rpy")
    if rpy is not None:
        return (float(rpy[0]), float(rpy[1]), float(rpy[2]))
    return (0.0, 0.0, 0.0)


def _box_from_json(item: dict[str, Any], position_unit: str, safety_margin: float, obstacle_id: str) -> Obstacle:
    # 支持 path/scenes 当前的 center_xyz + size_xyz 格式，也支持 min/max corner 的长方体写法。
    if "center_xyz" in item and "size_xyz" in item:
        pose = _pose_from_center_xyz(item, position_unit)
        size = tuple(_length_to_m(float(value), position_unit) for value in item["size_xyz"])
    elif "min_corner_xyz" in item and "max_corner_xyz" in item:
        min_corner = np.asarray([_length_to_m(float(value), position_unit) for value in item["min_corner_xyz"]])
        max_corner = np.asarray([_length_to_m(float(value), position_unit) for value in item["max_corner_xyz"]])
        center = 0.5 * (min_corner + max_corner)
        size_array = max_corner - min_corner
        roll, pitch, yaw = _orientation_from_json(item)
        pose = Pose6D(float(center[0]), float(center[1]), float(center[2]), roll, pitch, yaw)
        size = (float(size_array[0]), float(size_array[1]), float(size_array[2]))
    else:
        raise ValueError(f"box obstacle {obstacle_id!r} needs center_xyz/size_xyz or min_corner_xyz/max_corner_xyz")

    return Obstacle.box(pose, size=size, safety_margin=safety_margin, obstacle_id=obstacle_id)


def _sphere_from_json(item: dict[str, Any], position_unit: str, safety_margin: float, obstacle_id: str) -> Obstacle:
    pose = _pose_from_center_xyz(item, position_unit)
    if "radius" in item:
        radius = float(item["radius"])
    elif "radius_mm" in item:
        radius = float(item["radius_mm"])
        position_unit = "millimeters"
    else:
        raise ValueError(f"sphere obstacle {obstacle_id!r} missing radius")
    return Obstacle.sphere(
        pose,
        radius=_length_to_m(radius, position_unit),
        safety_margin=safety_margin,
        obstacle_id=obstacle_id,
    )


def _length_to_m(value: float, unit: str) -> float:
    normalized = unit.lower()
    if normalized in {"millimeter", "millimeters", "mm"}:
        return value * MM_TO_M
    if normalized in {"meter", "meters", "m"}:
        return value
    raise ValueError(f"unsupported position unit: {unit!r}")
