from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ik import Pose6D, pose_to_transform


@dataclass(frozen=True)
class Obstacle:
    # kind 支持两种障碍物：
    # 1. sphere: 用中心点 + 半径描述
    # 2. box: 用中心位姿 + 长宽高描述
    kind: str
    pose: Pose6D
    radius: float | None = None
    size: tuple[float, float, float] | None = None
    safety_margin: float = 0.0
    obstacle_id: str | None = None

    @classmethod
    def sphere(
        cls,
        pose: Pose6D,
        radius: float,
        safety_margin: float = 0.0,
        obstacle_id: str | None = None,
    ) -> "Obstacle":
        # safety_margin 是从场景 JSON 读来的障碍物额外膨胀距离，单位和 pose 一样是米。
        return cls(kind="sphere", pose=pose, radius=radius, safety_margin=safety_margin, obstacle_id=obstacle_id)

    @classmethod
    def box(
        cls,
        pose: Pose6D,
        size: Sequence[float],
        safety_margin: float = 0.0,
        obstacle_id: str | None = None,
    ) -> "Obstacle":
        sx, sy, sz = size
        return cls(
            kind="box",
            pose=pose,
            size=(float(sx), float(sy), float(sz)),
            safety_margin=safety_margin,
            obstacle_id=obstacle_id,
        )

    def signed_distance(self, point: np.ndarray) -> float:
        # 返回点到障碍物表面的有符号距离：
        # 正数表示在障碍物外部，0 表示刚好接触，负数表示已经进入障碍物内部。
        if self.kind == "sphere":
            if self.radius is None:
                raise ValueError("sphere obstacle requires radius")
            center = np.array([self.pose.x, self.pose.y, self.pose.z], dtype=float)
            return float(np.linalg.norm(point - center) - self.radius)

        if self.kind == "box":
            if self.size is None:
                raise ValueError("box obstacle requires size")
            transform = pose_to_transform(self.pose)
            local_point = transform[:3, :3].T @ (point - transform[:3, 3])
            half_size = 0.5 * np.array(self.size, dtype=float)
            q = np.abs(local_point) - half_size
            outside = np.maximum(q, 0.0)
            outside_distance = float(np.linalg.norm(outside))
            inside_distance = float(min(max(q[0], max(q[1], q[2])), 0.0))
            return outside_distance + inside_distance

        raise ValueError(f"unsupported obstacle kind: {self.kind}")
