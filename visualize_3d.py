from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ik import Pose6D, forward_kinematics_chain, pose_to_transform
from obstacles import Obstacle


def configure_chinese_font() -> None:
    # matplotlib 默认字体 DejaVu Sans 不包含中文字形。
    # 在 macOS 上优先找系统中文字体，找不到时程序仍可运行，只是可能显示方块。
    font_candidates = (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    )
    for font_path in font_candidates:
        path = Path(font_path)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            font_name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def plot_planning_scene_3d(
    joint_angles: Sequence[float],
    obstacles: Sequence[Obstacle],
    target_pose: Pose6D,
    pregrasp_pose: Pose6D | None = None,
    path: Sequence[np.ndarray] | None = None,
    output_path: Path | None = None,
) -> None:
    # 这个函数只负责“展示”，不参与 IK 或 RRT 计算。
    # joint_angles 决定当前机械臂姿态；obstacles 是障碍物；target_pose 是目标抓取点。
    configure_chinese_font()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    _draw_base(ax)
    _draw_robot_arm(ax, joint_angles)
    _draw_obstacles(ax, obstacles)
    _draw_target_nodes(ax, target_pose, pregrasp_pose)

    if path:
        _draw_end_effector_path(ax, path)

    ax.set_title("三维机械臂 RRT 场景")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.legend(loc="upper left")
    _set_axes_equal(ax)
    fig.tight_layout()

    if output_path is None:
        plt.show()
    else:
        fig.savefig(output_path, dpi=180)
        plt.close(fig)


def animate_planning_scene_3d(
    path: Sequence[np.ndarray],
    obstacles: Sequence[Obstacle],
    target_pose: Pose6D,
    pregrasp_pose: Pose6D | None = None,
    output_path: Path | None = None,
    fps: int = 4,
    trail_length: int | None = None,
) -> None:
    # 动画里每一帧取 path 中的一组关节角，重新画一次机械臂。
    # path 仍然是 RRT 输出的“关节角路径”，不是直接的空间坐标路径。
    if not path:
        raise ValueError("path must contain at least one joint configuration")

    configure_chinese_font()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    all_points = _scene_points_for_limits(path, obstacles, target_pose, pregrasp_pose)
    axis_limits = _equal_limits_from_points(all_points)

    def draw_frame(frame_index: int) -> None:
        ax.clear()
        _draw_base(ax)
        _draw_obstacles(ax, obstacles)
        _draw_target_nodes(ax, target_pose, pregrasp_pose)

        if trail_length is None:
            visible_path = path[: frame_index + 1]
        else:
            start = max(0, frame_index + 1 - trail_length)
            visible_path = path[start : frame_index + 1]
        _draw_end_effector_path(ax, visible_path)
        _draw_robot_arm(ax, path[frame_index])

        ax.set_title(f"三维机械臂 RRT 动画  第 {frame_index + 1}/{len(path)} 帧")
        ax.set_xlabel("X / m")
        ax.set_ylabel("Y / m")
        ax.set_zlabel("Z / m")
        ax.set_xlim3d(axis_limits[0])
        ax.set_ylim3d(axis_limits[1])
        ax.set_zlim3d(axis_limits[2])
        ax.legend(loc="upper left")

    animation = FuncAnimation(fig, draw_frame, frames=len(path), interval=1000 / max(fps, 1), repeat=True)

    if output_path is None:
        plt.show()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(output_path, writer=PillowWriter(fps=fps))
        plt.close(fig)


def _draw_base(ax) -> None:
    # 黑点是机械臂基坐标原点，也就是所有 JSON 位姿的参考坐标系。
    ax.scatter(0.0, 0.0, 0.0, c="black", s=60, label="基坐标原点")
    ax.quiver(0, 0, 0, 0.15, 0, 0, color="red", linewidth=1.5)
    ax.quiver(0, 0, 0, 0, 0.15, 0, color="green", linewidth=1.5)
    ax.quiver(0, 0, 0, 0, 0, 0.15, color="blue", linewidth=1.5)


def _draw_robot_arm(ax, joint_angles: Sequence[float]) -> None:
    # forward_kinematics_chain 会返回基座、各关节、末端的 4x4 位姿矩阵。
    # 取每个矩阵最后一列的前三个数，就是对应关节点在基坐标系下的位置。
    chain = forward_kinematics_chain(joint_angles)
    points = np.array([transform[:3, 3] for transform in chain], dtype=float)

    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        "-o",
        color="#1f77b4",
        linewidth=3,
        markersize=5,
        label="机械臂",
    )

    for index, point in enumerate(points):
        ax.text(point[0], point[1], point[2], f"J{index}", fontsize=8)

    _draw_pose_frame(ax, chain[-1], axis_length=0.08, label_prefix="末端")


def _draw_obstacles(ax, obstacles: Sequence[Obstacle]) -> None:
    for index, obstacle in enumerate(obstacles):
        center = np.array([obstacle.pose.x, obstacle.pose.y, obstacle.pose.z], dtype=float)
        ax.scatter(
            center[0],
            center[1],
            center[2],
            c="red",
            s=50,
            alpha=0.9,
            label="障碍物" if index == 0 else None,
        )
        if obstacle.kind == "sphere":
            _draw_sphere(ax, center, obstacle.radius or 0.05)
        elif obstacle.kind == "box":
            _draw_box(ax, obstacle)


def _draw_target_nodes(ax, target_pose: Pose6D, pregrasp_pose: Pose6D | None) -> None:
    # 绿色点是最终抓取目标；蓝色点是 RRT 先规划到的预抓取点。
    ax.scatter(target_pose.x, target_pose.y, target_pose.z, c="green", s=90, label="目标点")
    target_transform = pose_to_transform(target_pose)
    _draw_pose_frame(ax, target_transform, axis_length=0.07, label_prefix="目标")

    if pregrasp_pose is not None:
        ax.scatter(pregrasp_pose.x, pregrasp_pose.y, pregrasp_pose.z, c="dodgerblue", s=90, label="预抓取点")
        ax.plot(
            [pregrasp_pose.x, target_pose.x],
            [pregrasp_pose.y, target_pose.y],
            [pregrasp_pose.z, target_pose.z],
            "--",
            color="gray",
            linewidth=1.5,
            label="最后靠近段",
        )


def _draw_end_effector_path(ax, path: Sequence[np.ndarray]) -> None:
    # path 是一串关节角，不是空间坐标；画之前要先用正运动学转成末端点。
    positions = np.array([forward_kinematics_chain(joints)[-1][:3, 3] for joints in path], dtype=float)
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        "-",
        color="orange",
        linewidth=2,
        label="末端轨迹",
    )


def _draw_sphere(ax, center: np.ndarray, radius: float) -> None:
    u = np.linspace(0, 2 * math.pi, 32)
    v = np.linspace(0, math.pi, 16)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="red", alpha=0.18, linewidth=0)


def _draw_box(ax, obstacle: Obstacle) -> None:
    if obstacle.size is None:
        return

    corners_world = _box_corners_world(obstacle)
    faces = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]

    for face in faces:
        polygon = corners_world[face]
        collection = Poly3DCollection([polygon], facecolor="red", edgecolor="none", alpha=0.08)
        ax.add_collection3d(collection)

    for start, end in edges:
        ax.plot(
            [corners_world[start, 0], corners_world[end, 0]],
            [corners_world[start, 1], corners_world[end, 1]],
            [corners_world[start, 2], corners_world[end, 2]],
            color="red",
            alpha=0.65,
            linewidth=1.2,
        )


def _box_corners_world(obstacle: Obstacle) -> np.ndarray:
    sx, sy, sz = obstacle.size or (0.0, 0.0, 0.0)
    half = np.array([sx, sy, sz], dtype=float) * 0.5
    corners_local = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], half[1], half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], -half[2]],
            [half[0], half[1], half[2]],
        ],
        dtype=float,
    )
    transform = pose_to_transform(obstacle.pose)
    return (transform[:3, :3] @ corners_local.T).T + transform[:3, 3]


def _draw_pose_frame(ax, transform: np.ndarray, axis_length: float, label_prefix: str) -> None:
    origin = transform[:3, 3]
    axes = transform[:3, :3]
    colors = ("red", "green", "blue")
    labels = (f"{label_prefix}-x", f"{label_prefix}-y", f"{label_prefix}-z")
    for axis_index, color in enumerate(colors):
        direction = axes[:, axis_index] * axis_length
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            color=color,
            linewidth=1.2,
            label=labels[axis_index] if axis_index == 0 else None,
        )


def _set_axes_equal(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def _scene_points_for_limits(
    path: Sequence[np.ndarray],
    obstacles: Sequence[Obstacle],
    target_pose: Pose6D,
    pregrasp_pose: Pose6D | None,
) -> np.ndarray:
    # 动画每帧都要保持同一套坐标轴范围，否则画面会抖动。
    points: list[np.ndarray] = []
    for joints in path:
        chain = forward_kinematics_chain(joints)
        points.extend(transform[:3, 3] for transform in chain)

    points.append(np.array([target_pose.x, target_pose.y, target_pose.z], dtype=float))
    if pregrasp_pose is not None:
        points.append(np.array([pregrasp_pose.x, pregrasp_pose.y, pregrasp_pose.z], dtype=float))

    for obstacle in obstacles:
        points.append(np.array([obstacle.pose.x, obstacle.pose.y, obstacle.pose.z], dtype=float))
        if obstacle.kind == "sphere" and obstacle.radius is not None:
            radius = obstacle.radius
            center = np.array([obstacle.pose.x, obstacle.pose.y, obstacle.pose.z], dtype=float)
            points.extend(
                [
                    center + np.array([radius, 0.0, 0.0]),
                    center - np.array([radius, 0.0, 0.0]),
                    center + np.array([0.0, radius, 0.0]),
                    center - np.array([0.0, radius, 0.0]),
                    center + np.array([0.0, 0.0, radius]),
                    center - np.array([0.0, 0.0, radius]),
                ]
            )
        elif obstacle.kind == "box" and obstacle.size is not None:
            points.extend(_box_corners_world(obstacle))

    return np.array(points, dtype=float)


def _equal_limits_from_points(points: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    if radius <= 0.0:
        radius = 0.5
    return (
        [float(centers[0] - radius), float(centers[0] + radius)],
        [float(centers[1] - radius), float(centers[1] + radius)],
        [float(centers[2] - radius), float(centers[2] + radius)],
    )
