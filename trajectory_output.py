from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ik import Pose6D, forward_kinematics, transform_to_pose


M_TO_MM = 1000.0


def pose_to_output_dict(pose: Pose6D) -> dict[str, float]:
    # 对外输出时模仿 path/rrt_ik_planner.py：位置用 mm，姿态角继续用 rad。
    return {
        "x": pose.x * M_TO_MM,
        "y": pose.y * M_TO_MM,
        "z": pose.z * M_TO_MM,
        "roll": pose.roll,
        "pitch": pose.pitch,
        "yaw": pose.yaw,
    }


def joint_angles_to_pose(joint_angles: Sequence[float]) -> Pose6D:
    # RRT 路径里存的是 6 个关节角；这里用正运动学 FK 转成法兰 6D 位姿。
    # 这里的“法兰”就是输入 JSON 的 6D pose 所描述的位置，不包含额外 30cm 爪子。
    return transform_to_pose(forward_kinematics(joint_angles))


def build_rrt_trajectory_payload(
    joint_path: Sequence[np.ndarray],
    goal_pose: Pose6D,
    pregrasp_pose: Pose6D,
) -> dict:
    # rrt_stage_waypoints_6d 是你刚才贴的那种 index + pose 列表。
    # rrt_stage_joint_path 额外保留每个路径点的 6 个关节角，方便真正执行 movej。
    # 注意：pose 字段表示法兰，不包含 30cm 爪子。
    pose_path = [joint_angles_to_pose(joints) for joints in joint_path]
    return {
        "value_units": {
            "position_xyz": "millimeters",
            "orientation_rpy": "radians",
            "joint_angles": "radians",
        },
        "reference_frame": {
            "name": "robot_base",
            "origin_xyz": [0.0, 0.0, 0.0],
            "position_unit": "millimeters",
            "angle_unit": "radians",
        },
        "pose_meaning": {
            "pose": "robot flange pose, excluding the 30cm gripper",
        },
        "goal_pose_6d": pose_to_output_dict(goal_pose),
        "pre_grasp_pose_6d": pose_to_output_dict(pregrasp_pose),
        "rrt_stage_waypoints_6d": [
            {
                "index": index,
                "pose": pose_to_output_dict(pose),
            }
            for index, pose in enumerate(pose_path)
        ],
        "rrt_stage_joint_path": [
            {
                "index": index,
                "pose": pose_to_output_dict(pose),
                "joint_angles_rad": [float(value) for value in joints],
            }
            for index, (pose, joints) in enumerate(zip(pose_path, joint_path))
        ],
    }


def write_rrt_trajectory_json(
    output_path: Path,
    joint_path: Sequence[np.ndarray],
    goal_pose: Pose6D,
    pregrasp_pose: Pose6D,
) -> None:
    payload = build_rrt_trajectory_payload(
        joint_path=joint_path,
        goal_pose=goal_pose,
        pregrasp_pose=pregrasp_pose,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
