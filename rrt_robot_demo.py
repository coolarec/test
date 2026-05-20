from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import (
    ALLOW_APPROXIMATE_IK,
    ANIMATION_FPS,
    DEFAULT_ANIMATION_PATH,
    DEFAULT_PREGRASP_DISTANCE_MM,
    DEFAULT_REQUEST_JSON,
    DEFAULT_SCENE_IMAGE_PATH,
    DEFAULT_SCENE_JSON,
    DEFAULT_SCRIPT_PATH,
    DEFAULT_TRAJECTORY_JSON_PATH,
    MOVEJ_ACCEL_DEG_S2,
    MOVEJ_SPEED_DEG_S,
    ROBOT_HOST,
    RRT_GOAL_SAMPLE_RATE,
    RRT_MAX_ITERATIONS,
    RRT_RNG_SEED,
    RRT_SMOOTHING_ITERATIONS,
    RRT_STEP_SIZE,
)
from cs612a_driver import CS612ADriver
from ik import Pose6D
from json_io import PlanningRequest, SceneModel, load_planning_request, load_scene_model
from rrt import RRTPlanner, make_pregrasp_pose
from trajectory_output import write_rrt_trajectory_json
from visualize_3d import animate_planning_scene_3d, plot_planning_scene_3d


def plan_joint_path(
    request: PlanningRequest,
    scene: SceneModel,
    pregrasp_distance_m: float,
    allow_approximate_ik: bool,
    goal_sample_rate: float,
    max_iterations: int,
    smooth_iterations: int,
) -> tuple[list[np.ndarray], Pose6D]:
    # 驱动演示入口不再手写起点/终点/障碍物，而是全部从 inputs 和 scenes 读入。
    # JSON 中 x/y/z 和障碍物尺寸通常是 mm；load_* 已经统一转换成 m。
    planner = RRTPlanner(
        max_iterations=max_iterations,
        step_size=RRT_STEP_SIZE,
        goal_sample_rate=goal_sample_rate,
        smoothing_iterations=smooth_iterations,
        rng_seed=RRT_RNG_SEED,
    )
    path = planner.plan(
        request.goal_pose,
        scene.obstacles,
        start_angles=request.start_joint_angles,
        pregrasp_distance=pregrasp_distance_m,
        pregrasp_pose=request.pre_grasp_pose,
        allow_approximate_ik=allow_approximate_ik,
    )
    pregrasp_pose = request.pre_grasp_pose or make_pregrasp_pose(request.goal_pose, distance=pregrasp_distance_m)
    return path, pregrasp_pose


def format_joint_list(joint_angles: np.ndarray) -> str:
    return ", ".join(f"{value:.6f}" for value in joint_angles)


def build_rrt_motion_script(
    path: list[np.ndarray],
    speed_deg_s: float = MOVEJ_SPEED_DEG_S,
    accel_deg_s2: float = MOVEJ_ACCEL_DEG_S2,
) -> str:
    # 这里生成的是演示脚本模板：核心是把 RRT 路径中的每个关节节点串成连续 MoveJ。
    # 若后续拿到更完整的脚本手册，可只替换这一层的脚本语法模板。
    lines = [
        "def RRTMotionDemo():",
        "    # Generated from IK + RRT joint path",
        f"    speed_deg = {speed_deg_s:.3f}",
        f"    accel_deg = {accel_deg_s2:.3f}",
    ]

    for index, joint_angles in enumerate(path):
        lines.append(f"    # waypoint {index}")
        lines.append(f"    movej([{format_joint_list(joint_angles)}], speed_deg, accel_deg)")

    lines.append("end")
    return "\n".join(lines) + "\n"


def write_script_file(script_text: str, output_path: Path) -> None:
    output_path.write_text(script_text, encoding="utf-8")


def preview_summary(
    path: list[np.ndarray],
    pregrasp_pose: Pose6D,
    output_path: Path,
    trajectory_json_path: Path,
    scene_image_path: Path | None,
    animation_path: Path | None,
    request_json: Path,
    scene: SceneModel,
) -> str:
    return "\n".join(
        [
            "RRT demo prepared.",
            f"request json: {request_json}",
            f"scene json: {scene.source_path}",
            f"obstacles: {len(scene.obstacles)} (JSON unit: {scene.position_unit}, internally converted to meters)",
            f"path nodes: {len(path)}",
            f"final joints(rad): [{format_joint_list(path[-1])}]",
            (
                "pregrasp pose: "
                f"x={pregrasp_pose.x:.4f}, y={pregrasp_pose.y:.4f}, z={pregrasp_pose.z:.4f}, "
                f"roll={pregrasp_pose.roll:.4f}, pitch={pregrasp_pose.pitch:.4f}, yaw={pregrasp_pose.yaw:.4f}"
            ),
            f"script file: {output_path}",
            f"trajectory json: {trajectory_json_path}",
            f"3D scene image: {scene_image_path}" if scene_image_path else "3D scene image: disabled",
            f"3D animation: {animation_path}" if animation_path else "3D animation: disabled",
        ]
    )


def execute_script(host: str, script_text: str) -> None:
    driver = CS612ADriver(host)
    driver.script.send_script(script_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or execute a CS612A RRT motion demo script.")
    parser.add_argument(
        "--mode",
        choices=("preview", "execute"),
        default="preview",
        help="preview: only generate the script file; execute: send it to the robot through port 30001.",
    )
    parser.add_argument("--host", default=ROBOT_HOST, help="Robot controller IP address.")
    parser.add_argument("--output", type=Path, default=DEFAULT_SCRIPT_PATH, help="Output script file path.")
    parser.add_argument(
        "--trajectory-json",
        type=Path,
        default=DEFAULT_TRAJECTORY_JSON_PATH,
        help="Output JSON file containing index + 6D pose waypoints and joint angles.",
    )
    parser.add_argument(
        "--scene-image",
        type=Path,
        default=DEFAULT_SCENE_IMAGE_PATH,
        help="Output 3D scene image showing obstacles, robot arm, target, and pregrasp.",
    )
    parser.add_argument(
        "--no-scene-image",
        action="store_true",
        help="Skip writing the 3D scene image.",
    )
    parser.add_argument(
        "--animation",
        type=Path,
        default=DEFAULT_ANIMATION_PATH,
        help="Output GIF animation showing the robot moving along the RRT path.",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Skip writing the 3D animation.",
    )
    parser.add_argument("--animation-fps", type=int, default=ANIMATION_FPS, help="Animation frames per second.")
    parser.add_argument("--goal-sample-rate", type=float, default=RRT_GOAL_SAMPLE_RATE, help="Probability of sampling the IK goal in RRT.")
    parser.add_argument("--max-iterations", type=int, default=RRT_MAX_ITERATIONS, help="Maximum RRT expansion iterations.")
    parser.add_argument("--smooth-iterations", type=int, default=RRT_SMOOTHING_ITERATIONS, help="Shortcut smoothing iterations after RRT succeeds.")
    parser.add_argument(
        "--allow-approximate-ik",
        action="store_true",
        default=ALLOW_APPROXIMATE_IK,
        help="Let RRT plan to the best IK result even when exact IK fails.",
    )
    parser.add_argument(
        "--pregrasp-distance-mm",
        type=float,
        default=DEFAULT_PREGRASP_DISTANCE_MM,
        help="Distance from target pose back along the default flange tool axis, in millimeters. Ignored if request JSON has pre_grasp_pose_6d.",
    )
    parser.add_argument(
        "--request-json",
        type=Path,
        default=DEFAULT_REQUEST_JSON,
        help="Planning request JSON, normally under ./inputs.",
    )
    parser.add_argument(
        "--scene-json",
        type=Path,
        default=DEFAULT_SCENE_JSON,
        help="Obstacle scene JSON, normally under ./scenes.",
    )
    parser.add_argument("--speed-deg", type=float, default=MOVEJ_SPEED_DEG_S, help="MoveJ demo speed in deg/s.")
    parser.add_argument("--accel-deg", type=float, default=MOVEJ_ACCEL_DEG_S2, help="MoveJ demo acceleration in deg/s^2.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = load_planning_request(args.request_json)
    scene = load_scene_model(args.scene_json)
    try:
        path, pregrasp_pose = plan_joint_path(
            request,
            scene,
            pregrasp_distance_m=args.pregrasp_distance_mm / 1000.0,
            allow_approximate_ik=args.allow_approximate_ik,
            goal_sample_rate=args.goal_sample_rate,
            max_iterations=args.max_iterations,
            smooth_iterations=args.smooth_iterations,
        )
    except RuntimeError as error:
        # 读 JSON 成功后，IK/RRT 仍可能因为目标位姿不可达、姿态不匹配或碰撞约束过紧而失败。
        # 这里把已经读入的关键量打印出来，方便判断问题出在建模还是算法求解。
        goal = request.goal_pose
        print("RRT demo failed after loading JSON inputs.")
        print(f"request json: {args.request_json}")
        print(f"scene json: {scene.source_path}")
        print(f"start joints(rad): [{format_joint_list(request.start_joint_angles)}]")
        print(
            "goal pose(m/rad): "
            f"x={goal.x:.4f}, y={goal.y:.4f}, z={goal.z:.4f}, "
            f"roll={goal.roll:.4f}, pitch={goal.pitch:.4f}, yaw={goal.yaw:.4f}"
        )
        print(f"obstacles: {len(scene.obstacles)} (JSON unit: {scene.position_unit}, internally converted to meters)")
        raise error
    script_text = build_rrt_motion_script(path, speed_deg_s=args.speed_deg, accel_deg_s2=args.accel_deg)
    write_script_file(script_text, args.output)
    write_rrt_trajectory_json(
        output_path=args.trajectory_json,
        joint_path=path,
        goal_pose=request.goal_pose,
        pregrasp_pose=pregrasp_pose,
    )
    scene_image_path = None if args.no_scene_image else args.scene_image
    if scene_image_path is not None:
        plot_planning_scene_3d(
            joint_angles=path[-1],
            obstacles=scene.obstacles,
            target_pose=request.goal_pose,
            pregrasp_pose=pregrasp_pose,
            path=path,
            output_path=scene_image_path,
        )
    animation_path = None if args.no_animation else args.animation
    if animation_path is not None:
        animate_planning_scene_3d(
            path=path,
            obstacles=scene.obstacles,
            target_pose=request.goal_pose,
            pregrasp_pose=pregrasp_pose,
            output_path=animation_path,
            fps=args.animation_fps,
        )

    print(
        preview_summary(
            path,
            pregrasp_pose,
            args.output,
            args.trajectory_json,
            scene_image_path,
            animation_path,
            args.request_json,
            scene,
        )
    )

    if args.mode == "execute":
        execute_script(args.host, script_text)
        print(f"script sent to robot at {args.host}")
    else:
        print("preview mode: script was generated only, no robot command was sent")


if __name__ == "__main__":
    main()
