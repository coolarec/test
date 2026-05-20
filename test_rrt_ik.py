from __future__ import annotations

import math
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import (
    ALLOW_APPROXIMATE_IK,
    ANIMATION_FPS,
    DEFAULT_ANIMATION_PATH,
    DEFAULT_PATH_IMAGE_PATH,
    DEFAULT_PREGRASP_DISTANCE_MM,
    DEFAULT_REQUEST_JSON,
    DEFAULT_SCENE_IMAGE_PATH,
    DEFAULT_SCENE_JSON,
    DEFAULT_TRAJECTORY_JSON_PATH,
    RRT_GOAL_SAMPLE_RATE,
    RRT_MAX_ITERATIONS,
    RRT_RNG_SEED,
    RRT_SMOOTHING_ITERATIONS,
    RRT_STEP_SIZE,
)
from ik import Pose6D, end_effector_position, forward_kinematics, inverse_kinematics, pose_to_transform, transform_to_pose
from json_io import load_planning_request, load_scene_model
from obstacles import Obstacle
from rrt import RRTPlanner, make_pregrasp_pose
from trajectory_output import write_rrt_trajectory_json
from visualize_3d import animate_planning_scene_3d, plot_planning_scene_3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test IK/RRT and render the JSON scene.")
    parser.add_argument(
        "--pregrasp-distance-mm",
        type=float,
        default=DEFAULT_PREGRASP_DISTANCE_MM,
        help="Distance from target pose back along tool Z axis, in millimeters. Ignored if request JSON has pre_grasp_pose_6d.",
    )
    parser.add_argument(
        "--allow-approximate-ik",
        action="store_true",
        default=ALLOW_APPROXIMATE_IK,
        help="Let RRT plan to the best IK result even when the current model cannot exactly reach the requested 6D pose.",
    )
    parser.add_argument("--goal-sample-rate", type=float, default=RRT_GOAL_SAMPLE_RATE, help="Probability of sampling the IK goal in RRT.")
    parser.add_argument("--max-iterations", type=int, default=RRT_MAX_ITERATIONS, help="Maximum RRT expansion iterations.")
    parser.add_argument("--smooth-iterations", type=int, default=RRT_SMOOTHING_ITERATIONS, help="Shortcut smoothing iterations after RRT succeeds.")
    return parser.parse_args()


def make_reachable_target_pose(pregrasp_joint_angles: np.ndarray, pregrasp_distance: float = 0.0) -> Pose6D:
    # 用 CS612A 正运动学生成一个确定可达的目标点。
    # pregrasp_distance 默认为 0，表示预抓取点和目标点重合。
    # 这样测试用例不会因为随手写了一个不可达目标而失败。
    pregrasp_transform = forward_kinematics(pregrasp_joint_angles)
    target_transform = pregrasp_transform.copy()
    target_transform[:3, 3] = pregrasp_transform[:3, 3] + pregrasp_distance * pregrasp_transform[:3, 2]
    return transform_to_pose(target_transform)


def plot_path(
    path: list[np.ndarray],
    target_pose: Pose6D,
    pregrasp_pose: Pose6D,
    obstacles: list[Obstacle],
    output_path: Path,
) -> None:
    # path 中保存的是关节角；画图时先用正运动学转成末端三维坐标。
    # 所以图上的蓝线是“末端轨迹”，不是整条机械臂的骨架。
    positions = np.array([end_effector_position(angles) for angles in path])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    # 蓝色折线表示 RRT 规划出的末端运动轨迹。
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "-o", linewidth=2, markersize=3, label="RRT path")
    ax.scatter(0, 0, 0, c="black", s=50, label="base")
    ax.scatter(target_pose.x, target_pose.y, target_pose.z, c="green", s=80, label="grasp target")
    ax.scatter(pregrasp_pose.x, pregrasp_pose.y, pregrasp_pose.z, c="blue", s=80, label="pregrasp")

    for index, obstacle in enumerate(obstacles):
        center = np.array([obstacle.pose.x, obstacle.pose.y, obstacle.pose.z], dtype=float)
        ax.scatter(center[0], center[1], center[2], c="red", s=120, alpha=0.8, label="obstacle" if index == 0 else None)
        if obstacle.kind == "sphere":
            draw_sphere(ax, center, obstacle.radius or 0.08)
        elif obstacle.kind == "box":
            draw_box(ax, obstacle)

    ax.set_title("6D Pose IK + RRT Path to Pregrasp Point")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.legend(loc="upper left")
    set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_sphere(ax, center: np.ndarray, radius: float) -> None:
    # 这里只是为了可视化障碍物球，不参与规划计算。
    u = np.linspace(0, 2 * math.pi, 24)
    v = np.linspace(0, math.pi, 12)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="red", alpha=0.18, linewidth=0)


def draw_box(ax, obstacle: Obstacle) -> None:
    # box 障碍物允许有姿态，因此先在局部坐标系生成 8 个顶点，再变换到世界坐标系。
    if obstacle.size is None:
        return

    sx, sy, sz = obstacle.size
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
    corners_world = (transform[:3, :3] @ corners_local.T).T + transform[:3, 3]
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]
    for start, end in edges:
        ax.plot(
            [corners_world[start, 0], corners_world[end, 0]],
            [corners_world[start, 1], corners_world[end, 1]],
            [corners_world[start, 2], corners_world[end, 2]],
            color="red",
            alpha=0.6,
        )


def set_axes_equal(ax) -> None:
    # 让 x/y/z 三个方向使用相同比例，避免球体在图中被压扁。
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def main() -> None:
    # main() 就是整个示例的主流程：
    # 1. 从 path/inputs 读取起点和目标点
    # 2. 从 path/scenes 读取障碍物环境
    # 3. 计算预抓取点和 IK
    # 4. 用 RRT 规划
    # 5. 把结果画出来

    args = parse_args()
    pregrasp_distance_m = args.pregrasp_distance_mm / 1000.0
    request = load_planning_request(DEFAULT_REQUEST_JSON)
    scene = load_scene_model(DEFAULT_SCENE_JSON)
    target_pose = request.goal_pose
    start_angles = request.start_joint_angles
    obstacles = scene.obstacles
    pregrasp_pose = request.pre_grasp_pose or make_pregrasp_pose(target_pose, distance=pregrasp_distance_m)

    print("Request source:", DEFAULT_REQUEST_JSON)
    print("Start joints(rad):", np.round(start_angles, 4))
    print(
        "Target pose(m/rad):",
        f"x={target_pose.x:.4f}, y={target_pose.y:.4f}, z={target_pose.z:.4f},",
        f"roll={target_pose.roll:.4f}, pitch={target_pose.pitch:.4f}, yaw={target_pose.yaw:.4f}",
    )
    print("Scene source:", scene.source_path)
    print("Scene obstacles:", len(obstacles))
    print(f"Pregrasp distance(mm): {args.pregrasp_distance_mm:.3f}")
    animation_output_path = DEFAULT_ANIMATION_PATH

    # 先画真实 JSON 环境：机械臂起点、scene 障碍物、inputs 目标点。
    # 这样即使后续 IK/RRT 因模型或姿态没收敛失败，也能先检查位置建模是否对齐。
    scene_output_path = DEFAULT_SCENE_IMAGE_PATH
    plot_planning_scene_3d(
        joint_angles=start_angles,
        obstacles=obstacles,
        target_pose=target_pose,
        pregrasp_pose=pregrasp_pose,
        path=None,
        output_path=scene_output_path,
    )
    print("Saved 3D scene:", scene_output_path)

    # 当前把预抓取点和目标点的距离设为 0，用 IK 转成目标关节角。
    ik_result = inverse_kinematics(pregrasp_pose, initial_angles=start_angles, obstacles=obstacles)
    # 这里打印的是 IK 是否成功，以及最终误差是否足够小。
    print("IK success:", ik_result.success)
    print("IK joint angles(rad):", np.round(ik_result.joint_angles, 4))
    print("IK position error:", round(ik_result.position_error, 6))
    print("IK orientation error:", round(ik_result.orientation_error, 6))
    if not ik_result.success and not args.allow_approximate_ik:
        animation_output_path.unlink(missing_ok=True)
        print("IK failed for the JSON target. No GIF was generated because there is no verified RRT path.")
        return

    # RRT 类只负责规划；画图和测试逻辑都留在这个文件里。
    planner = RRTPlanner(
        max_iterations=args.max_iterations,
        step_size=RRT_STEP_SIZE,
        goal_sample_rate=args.goal_sample_rate,
        smoothing_iterations=args.smooth_iterations,
        rng_seed=RRT_RNG_SEED,
    )
    try:
        path = planner.plan(
            target_pose,
            obstacles,
            start_angles=start_angles,
            pregrasp_distance=pregrasp_distance_m,
            pregrasp_pose=pregrasp_pose,
            allow_approximate_ik=args.allow_approximate_ik,
        )
    except RuntimeError as error:
        animation_output_path.unlink(missing_ok=True)
        print(f"RRT failed for the JSON target: {error}")
        print("No GIF was generated because there is no collision-verified RRT path.")
        return
    # path 是一串关节角节点，从起点配置一路连到预抓取点配置。
    print("RRT path nodes:", len(path))
    print("Final joint angles(rad):", np.round(path[-1], 4))

    output_path = DEFAULT_PATH_IMAGE_PATH
    plot_path(path, target_pose, pregrasp_pose, obstacles, output_path)
    print("Saved figure:", output_path)

    trajectory_json_path = DEFAULT_TRAJECTORY_JSON_PATH
    write_rrt_trajectory_json(
        output_path=trajectory_json_path,
        joint_path=path,
        goal_pose=target_pose,
        pregrasp_pose=pregrasp_pose,
    )
    print("Saved trajectory JSON:", trajectory_json_path)

    # RRT 成功后再覆盖生成一张带末端轨迹、终点机械臂姿态的完整环境图。
    plot_planning_scene_3d(
        joint_angles=path[-1],
        obstacles=obstacles,
        target_pose=target_pose,
        pregrasp_pose=pregrasp_pose,
        path=path,
        output_path=scene_output_path,
    )
    print("Saved 3D scene:", scene_output_path)

    animate_planning_scene_3d(
        path=path,
        obstacles=obstacles,
        target_pose=target_pose,
        pregrasp_pose=pregrasp_pose,
        output_path=animation_output_path,
        fps=ANIMATION_FPS,
    )
    print("Saved 3D animation:", animation_output_path)


if __name__ == "__main__":
    main()
