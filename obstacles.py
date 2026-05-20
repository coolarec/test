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

    def __post_init__(self) -> None:
        # signed_distance 会被 IK/RRT 高频调用。
        # 这里把障碍物的矩阵、中心点、半尺寸提前缓存下来，避免每次碰撞检测都重复构造。
        transform = pose_to_transform(self.pose)
        object.__setattr__(self, "_transform", transform)
        object.__setattr__(self, "_rotation", transform[:3, :3])
        object.__setattr__(self, "_center", transform[:3, 3])
        if self.size is not None:
            object.__setattr__(self, "_half_size", 0.5 * np.array(self.size, dtype=float))
        else:
            object.__setattr__(self, "_half_size", None)

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
            return float(np.linalg.norm(point - self._center) - self.radius)

        if self.kind == "box":
            if self.size is None:
                raise ValueError("box obstacle requires size")
            local_point = self._rotation.T @ (point - self._center)
            q = np.abs(local_point) - self._half_size
            outside = np.maximum(q, 0.0)
            outside_distance = float(np.linalg.norm(outside))
            inside_distance = float(min(max(q[0], max(q[1], q[2])), 0.0))
            return outside_distance + inside_distance

        raise ValueError(f"unsupported obstacle kind: {self.kind}")
