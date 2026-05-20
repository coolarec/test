from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ik import DEFAULT_DH_LINKS, DHLink, Pose6D, forward_kinematics_chain, inverse_kinematics, pose_to_transform
from obstacles import Obstacle


@dataclass
class RRTNode:
    # 树上的一个节点：保存一组关节角，以及它的父节点编号，便于最后回溯整条路径。
    joint_angles: np.ndarray
    parent: int | None


# -----------------------------
# Pose helpers
# -----------------------------

def make_pregrasp_pose(target_pose: Pose6D, distance: float = 0.0) -> Pose6D:
    # 预抓取点由目标抓取点沿工具朝向反向退 distance 得到。
    # 当前默认 distance=0，表示先让预抓取点和目标点重合，便于调试 IK/RRT 主流程。
    if distance < 0.0:
        raise ValueError("pregrasp_distance should be non-negative")

    target_transform = pose_to_transform(target_pose)
    # 手册 3.5 TCP 默认方向说明：CS612A 末端法兰默认工具方向是法兰 Y 轴负方向。
    # 所以预抓取点沿工具方向的反方向后退时，使用末端局部 -Y 轴作为接近方向。
    approach_axis = -target_transform[:3, 1]
    pregrasp_position = target_transform[:3, 3] - distance * approach_axis
    return Pose6D(
        x=float(pregrasp_position[0]),
        y=float(pregrasp_position[1]),
        z=float(pregrasp_position[2]),
        roll=target_pose.roll,
        pitch=target_pose.pitch,
        yaw=target_pose.yaw,
    )


# -----------------------------
# RRT planner
# -----------------------------

class RRTPlanner:
    """RRT planner in joint space, with simple sphere/box obstacle checks."""

    def __init__(
        self,
        dh_links: Sequence[DHLink] = DEFAULT_DH_LINKS,
        # 机械臂的 DH 参数表，决定正运动学和碰撞检测使用的机器人模型。
        joint_limits: Sequence[tuple[float, float]] | None = None,
        # 每个关节的上下限；如果不传，就默认使用 DHLink 里写好的限位。
        step_size: float = 0.18,
        # 树每次扩展的最大步长。步长大时搜索快，但更容易跨过狭窄可行区域。
        goal_sample_rate: float = 0.15,
        # 以多大概率直接采样目标点。数值越大，树越容易朝目标方向生长。
        max_iterations: int = 3000,
        # 最多扩展多少轮。太小可能来不及找到路，太大则会增加搜索时间。
        collision_check_resolution: float = 0.04,
        # 检查一条边时的插值分辨率。越小越细，漏检概率越低，但计算更慢。
        safety_margin: float = 0.03,
        # 在障碍物半径之外额外留出的安全距离，让规划更保守。
        smoothing_iterations: int = 200,
        # 路径找到后做 shortcut 平滑的次数。次数越多，路径节点通常越少。
        rng_seed: int | None = 7,
        # 随机种子。固定后每次运行更容易复现实验结果。
    ) -> None:
        # RRT 在关节空间里搜索，每个节点保存一组 6 维关节角。
        # 所以它规划的不是笛卡尔空间直线，而是“关节角怎样一步步变化到目标”。
        self.dh_links = tuple(dh_links)
        self.joint_count = len(self.dh_links)
        self.joint_limits = (
            tuple(joint_limits)
            if joint_limits is not None
            else tuple((link.min_angle, link.max_angle) for link in self.dh_links)
        )
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iterations = max_iterations
        self.collision_check_resolution = collision_check_resolution
        self.safety_margin = safety_margin
        self.smoothing_iterations = smoothing_iterations
        self.rng = random.Random(rng_seed)

    def plan(
        self,
        target_pose: Pose6D,
        obstacles: Sequence[Obstacle],
        start_angles: Sequence[float] | None = None,
        pregrasp_distance: float = 0.0,
        pregrasp_pose: Pose6D | None = None,
        allow_approximate_ik: bool = False,
    ) -> list[np.ndarray]:
        """Plan a collision-checked joint-space path to the pregrasp configuration."""
        # 起点默认是基坐标处的零关节角配置。
        # 如果以后接入真实机器人，也可以把 start_angles 改成当前实时关节角。
        start = np.zeros(self.joint_count, dtype=float) if start_angles is None else np.asarray(start_angles, dtype=float)
        if start.shape != (self.joint_count,):
            raise ValueError(f"start_angles must contain {self.joint_count} values")

        # 这里要求调用方显式传入 Obstacle 对象，避免混用旧格式。
        normalized_obstacles = list(obstacles)
        # path/inputs 如果直接给了 pre_grasp_pose_6d，就使用文件里的预抓取点；
        # 否则按抓取点末端 Z 轴反向退 pregrasp_distance 自动生成。
        if pregrasp_pose is None:
            pregrasp_pose = make_pregrasp_pose(target_pose, pregrasp_distance)

        # 先用 IK 把预抓取 6D 位姿转换成 RRT 的目标关节角。
        # 这一步把“空间里的位姿目标”翻译成“关节空间里的终点”。
        ik_result = inverse_kinematics(
            pregrasp_pose,
            initial_angles=start,
            dh_links=self.dh_links,
            obstacles=normalized_obstacles,
            obstacle_margin=self.safety_margin,
        )
        if not ik_result.success and not allow_approximate_ik:
            raise RuntimeError(
                "IK failed for pregrasp pose: "
                f"position_error={ik_result.position_error:.4f}, "
                f"orientation_error={ik_result.orientation_error:.4f}"
            )

        goal = ik_result.joint_angles
        if self._in_collision(start, normalized_obstacles):
            raise RuntimeError("start configuration is in collision")
        if self._in_collision(goal, normalized_obstacles):
            raise RuntimeError("goal configuration is in collision")
        # 如果起点到目标在关节空间直连就安全，直接返回。
        # 这一步能避免 RRT 在简单可行场景里反复随机采样。
        if not self._edge_in_collision(start, goal, normalized_obstacles):
            return [start, goal]

        # RRT-Connect：同时从起点和目标各长一棵树，比单向 RRT 更容易穿过狭窄通道。
        start_nodes = [RRTNode(start, parent=None)]
        goal_nodes = [RRTNode(goal, parent=None)]
        for _ in range(self.max_iterations):
            # 以一定概率直接采样目标点，其他时候随机采样关节空间。
            sampled_goal = self.rng.random() < self.goal_sample_rate
            sample = goal if sampled_goal else self._sample_configuration()
            new_index = self._extend_tree(start_nodes, sample, normalized_obstacles)
            if new_index is None:
                start_nodes, goal_nodes = goal_nodes, start_nodes
                continue

            connect_index = self._connect_tree(
                goal_nodes,
                start_nodes[new_index].joint_angles,
                normalized_obstacles,
            )
            if connect_index is not None:
                path_from_start_tree = self._rebuild_path(start_nodes, new_index)
                path_from_goal_tree = self._rebuild_path(goal_nodes, connect_index)
                if np.allclose(path_from_start_tree[0], start):
                    raw_path = path_from_start_tree + list(reversed(path_from_goal_tree))[1:]
                else:
                    raw_path = path_from_goal_tree + list(reversed(path_from_start_tree))[1:]
                return self._smooth_path(raw_path, normalized_obstacles)

            # 两棵树轮流扩展，避免只在一侧浪费采样。
            start_nodes, goal_nodes = goal_nodes, start_nodes

        raise RuntimeError("RRT-Connect failed to find a path within max_iterations")

    def _extend_tree(
        self,
        nodes: list[RRTNode],
        target: np.ndarray,
        obstacles: Sequence[Obstacle],
    ) -> int | None:
        """Grow one tree by one step toward target."""
        nearest_index = self._nearest_node_index(nodes, target)
        new_config = self._steer(nodes[nearest_index].joint_angles, target)
        if self._edge_in_collision(nodes[nearest_index].joint_angles, new_config, obstacles):
            return None
        nodes.append(RRTNode(new_config, parent=nearest_index))
        return len(nodes) - 1

    def _connect_tree(
        self,
        nodes: list[RRTNode],
        target: np.ndarray,
        obstacles: Sequence[Obstacle],
    ) -> int | None:
        """Keep growing one tree toward target until blocked or connected."""
        current_index = self._nearest_node_index(nodes, target)
        while True:
            current = nodes[current_index].joint_angles
            next_config = self._steer(current, target)
            if self._edge_in_collision(current, next_config, obstacles):
                return None
            nodes.append(RRTNode(next_config, parent=current_index))
            current_index = len(nodes) - 1
            if np.linalg.norm(next_config - target) <= 1e-9:
                return current_index

    def _sample_configuration(self) -> np.ndarray:
        """Sample a random joint configuration inside the joint limits."""
        # 在每个关节的限位范围内随机采样。
        return np.array([self.rng.uniform(low, high) for low, high in self.joint_limits], dtype=float)

    def _nearest_node_index(self, nodes: Sequence[RRTNode], sample: np.ndarray) -> int:
        """Find the tree node closest to the sampled configuration."""
        # 在树里找离采样点最近的老节点，作为本次扩展的出发点。
        distances = [float(np.linalg.norm(node.joint_angles - sample)) for node in nodes]
        return int(np.argmin(distances))

    def _steer(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Move from start toward target by at most one step_size."""
        # 从 start 朝 target 走一步，步长不超过 step_size。
        direction = target - start
        distance = float(np.linalg.norm(direction))
        if distance <= self.step_size:
            return np.clip(target, self._lower_bounds(), self._upper_bounds())
        return np.clip(start + direction / distance * self.step_size, self._lower_bounds(), self._upper_bounds())

    def _in_collision(self, joint_angles: np.ndarray, obstacles: Sequence[Obstacle]) -> bool:
        """Check whether the joint chain collides with any supported obstacle."""
        # 这里把每个关节原点和末端点都拿出来做简化检查，
        # 这样比只看末端点更接近真实机械臂避障。
        chain_transforms = forward_kinematics_chain(joint_angles, self.dh_links)
        for transform in chain_transforms:
            point = transform[:3, 3]
            for obstacle in obstacles:
                # 碰撞阈值 = 规划器全局安全距离 + 障碍物 JSON 自己带的安全距离。
                # 例如场景文件里 safety_margin=6.0 且单位是 mm，读入后就是 0.006m。
                if obstacle.signed_distance(point) <= self.safety_margin + obstacle.safety_margin:
                    return True
        return False

    def _edge_in_collision(self, start: np.ndarray, end: np.ndarray, obstacles: Sequence[Obstacle]) -> bool:
        """Check intermediate points along a joint-space edge."""
        # 一条边不能只检查端点，要沿关节空间插值多检查几个中间点。
        # 采样越密，越不容易漏掉细小障碍物，但计算也会更慢。
        distance = float(np.linalg.norm(end - start))
        steps = max(2, math.ceil(distance / self.collision_check_resolution))
        for alpha in np.linspace(0.0, 1.0, steps):
            config = start + alpha * (end - start)
            if self._in_collision(config, obstacles):
                return True
        return False

    def _rebuild_path(self, nodes: Sequence[RRTNode], goal_index: int) -> list[np.ndarray]:
        """Reconstruct the final path by following parent links backward."""
        # 从目标节点沿 parent 指针回溯，得到从起点到目标的路径。
        path = []
        current: int | None = goal_index
        while current is not None:
            node = nodes[current]
            path.append(node.joint_angles)
            current = node.parent
        path.reverse()
        return path

    def _smooth_path(self, path: list[np.ndarray], obstacles: Sequence[Obstacle]) -> list[np.ndarray]:
        """Shortcut a collision-free path by removing unnecessary intermediate nodes."""
        if len(path) <= 2 or self.smoothing_iterations <= 0:
            return path

        smoothed = [node.copy() for node in path]
        for _ in range(self.smoothing_iterations):
            if len(smoothed) <= 2:
                break

            first = self.rng.randrange(0, len(smoothed) - 2)
            second = self.rng.randrange(first + 2, len(smoothed))
            if not self._edge_in_collision(smoothed[first], smoothed[second], obstacles):
                smoothed = smoothed[: first + 1] + smoothed[second:]

        return smoothed

    def _lower_bounds(self) -> np.ndarray:
        """Return joint lower bounds as a NumPy array."""
        return np.array([low for low, _ in self.joint_limits], dtype=float)

    def _upper_bounds(self) -> np.ndarray:
        """Return joint upper bounds as a NumPy array."""
        return np.array([high for _, high in self.joint_limits], dtype=float)
