from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Pose6D:
    """6D pose in the base frame: position in meters, orientation in radians."""

    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
class DHLink:
    # 一节连杆对应一组 DH 参数。6 轴机械臂就会有 6 个 DHLink。
    a: float
    # a: 连杆长度，表示沿前一个坐标系 x 轴平移多少。
    alpha: float
    # alpha: 连杆扭角，表示绕前一个坐标系 x 轴旋转多少。
    d: float
    # d: 连杆偏移，表示沿当前关节 z 轴平移多少。
    theta_offset: float = 0.0
    # theta_offset: 关节零位补偿，实际参与计算的角度是 joint_angle + theta_offset。
    min_angle: float = -math.pi
    max_angle: float = math.pi
    # min_angle / max_angle: 该关节允许转动的最小值和最大值。


@dataclass(frozen=True)
class IKResult:
    # IK 的输出：是否求解成功、解出来的关节角、以及残余误差大小。
    success: bool
    joint_angles: np.ndarray
    position_error: float
    orientation_error: float
    iterations: int


@dataclass(frozen=True)
class RobotSpec:
    # 机器人手册中的关键参数集中放在这里，方便后续替换型号或做校验。
    model: str
    dh_links: tuple[DHLink, ...]
    payload_kg: float
    work_radius_m: float
    repeatability_m: float
    joint_speed_limits_deg_s: tuple[float, ...]
    max_tcp_speed_m_s: float
    force_range_n: float
    torque_range_nm: float


CS612A_JOINT_LIMIT = 2 * math.pi

CS612A_DH_LINKS = (
    # CS612A 用户手册 Ver2.15.0 表 2-1：theta 默认为 0，单位已按手册使用 m/rad。
    DHLink(a=0.0, alpha=0.0, d=0.193, min_angle=-CS612A_JOINT_LIMIT, max_angle=CS612A_JOINT_LIMIT),
    DHLink(a=0.0, alpha=1.5708, d=0.0, min_angle=-CS612A_JOINT_LIMIT, max_angle=CS612A_JOINT_LIMIT),
    DHLink(a=-0.615, alpha=0.0, d=0.0395, min_angle=-CS612A_JOINT_LIMIT, max_angle=CS612A_JOINT_LIMIT),
    DHLink(a=0.0, alpha=-1.5708, d=0.5787, min_angle=-CS612A_JOINT_LIMIT, max_angle=CS612A_JOINT_LIMIT),
    DHLink(a=0.0, alpha=1.5708, d=0.118, min_angle=-CS612A_JOINT_LIMIT, max_angle=CS612A_JOINT_LIMIT),
    DHLink(a=0.0, alpha=-1.5708, d=0.4288, min_angle=-CS612A_JOINT_LIMIT, max_angle=CS612A_JOINT_LIMIT),
)

CS612A_SPEC = RobotSpec(
    model="CS612A",
    dh_links=CS612A_DH_LINKS,
    payload_kg=12.0,
    work_radius_m=1.3225,
    repeatability_m=0.00003,
    joint_speed_limits_deg_s=(180.0, 180.0, 210.0, 410.0, 300.0, 470.0),
    max_tcp_speed_m_s=4.0,
    force_range_n=300.0,
    torque_range_nm=30.0,
)

DEFAULT_ROBOT_SPEC = CS612A_SPEC
DEFAULT_DH_LINKS = DEFAULT_ROBOT_SPEC.dh_links


def pose_to_transform(pose: Pose6D) -> np.ndarray:
    # 把 6D 位姿 (x, y, z, roll, pitch, yaw) 转成 4x4 齐次变换矩阵。
    # 前 3x3 是旋转，最后一列是平移，后面正运动学和预抓取点都会用这个形式。
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [pose.roll, pose.pitch, pose.yaw]).as_matrix()
    transform[:3, 3] = [pose.x, pose.y, pose.z]
    return transform


def transform_to_pose(transform: np.ndarray) -> Pose6D:
    # 把齐次变换矩阵转回 6D 位姿，方便测试中由正运动学生成可达目标。
    # 也就是说：既能从位姿进矩阵，也能从矩阵还原回位姿。
    roll, pitch, yaw = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")
    return Pose6D(
        x=float(transform[0, 3]),
        y=float(transform[1, 3]),
        z=float(transform[2, 3]),
        roll=float(roll),
        pitch=float(pitch),
        yaw=float(yaw),
    )


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    # Modified DH 变换矩阵：描述相邻两个关节坐标系之间的位姿关系。
    # CS612A 手册表 2-1 给出的是 theta/a/d/alpha 这些数值；
    # 经目标点校验，这组参数需要按 Modified DH 约定连乘，才能和控制器位姿一致。
    # 可以把它理解成四步动作合成后的结果：
    # 1. 绕当前 z 轴转 theta
    # 2. 沿当前 x 轴移 a
    # 3. 绕当前 x 轴转 alpha
    # 4. 沿当前 z 轴移 d
    # 最后把这四步统一写成一个 4x4 矩阵，方便连续连乘。
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -sa * d],
            [st * sa, ct * sa, ca, ca * d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def forward_kinematics(
    joint_angles: Sequence[float],
    dh_links: Sequence[DHLink] = DEFAULT_DH_LINKS,
) -> np.ndarray:
    # 输入一组关节角，输出末端在基坐标系中的位姿。
    # 这就是“正运动学”：关节角 -> 末端位姿。
    transform = np.eye(4)
    for angle, link in zip(joint_angles, dh_links):
        # 逐个关节累乘 DH 矩阵，得到末端相对基坐标系的位姿。
        transform = transform @ dh_transform(
            link.a,
            link.alpha,
            link.d,
            angle + link.theta_offset,
        )
    return transform


def forward_kinematics_chain(
    joint_angles: Sequence[float],
    dh_links: Sequence[DHLink] = DEFAULT_DH_LINKS,
) -> list[np.ndarray]:
    # 返回从基座到末端的每一级齐次变换矩阵，便于做关节级碰撞检查。
    transforms = [np.eye(4)]
    current = np.eye(4)
    for angle, link in zip(joint_angles, dh_links):
        current = current @ dh_transform(
            link.a,
            link.alpha,
            link.d,
            angle + link.theta_offset,
        )
        transforms.append(current.copy())
    return transforms


def end_effector_position(
    joint_angles: Sequence[float],
    dh_links: Sequence[DHLink] = DEFAULT_DH_LINKS,
) -> np.ndarray:
    # 只取末端位置 (x, y, z)，画图和简化碰撞检测只需要这个结果。
    return forward_kinematics(joint_angles, dh_links)[:3, 3]


def _pose_residual(
    joint_angles: np.ndarray,
    target_transform: np.ndarray,
    dh_links: Sequence[DHLink],
    orientation_weight: float,
    obstacles: Sequence[object] | None,
    obstacle_margin: float,
    obstacle_weight: float,
) -> np.ndarray:
    # 这是 IK 优化器真正看到的“误差函数”。
    # 当前关节角如果对应的末端位姿离目标越远，这里返回的 6 维误差向量就越大。
    current = forward_kinematics(joint_angles, dh_links)
    position_error = target_transform[:3, 3] - current[:3, 3]

    # 姿态误差用旋转向量表示，比直接比较欧拉角更稳定。
    rotation_delta = target_transform[:3, :3] @ current[:3, :3].T
    orientation_error = Rotation.from_matrix(rotation_delta).as_rotvec()

    residual = np.r_[position_error, orientation_weight * orientation_error]
    if not obstacles:
        return residual

    # IK 避障采用软约束：当关节链上的点接近障碍物时，给优化器增加惩罚项。
    chain_transforms = forward_kinematics_chain(joint_angles, dh_links)
    chain_points = [transform[:3, 3] for transform in chain_transforms]
    obstacle_penalties: list[float] = []
    for point in chain_points:
        for obstacle in obstacles:
            signed_distance = obstacle.signed_distance(point)
            obstacle_penalties.append(obstacle_weight * max(0.0, obstacle_margin - signed_distance))
    return np.r_[residual, np.asarray(obstacle_penalties, dtype=float)]


def inverse_kinematics(
    target_pose: Pose6D,
    initial_angles: Sequence[float] | None = None,
    dh_links: Sequence[DHLink] = DEFAULT_DH_LINKS,
    obstacles: Sequence[object] | None = None,
    max_iterations: int = 300,
    position_tolerance: float = 1e-3,
    orientation_tolerance: float = 1e-2,
    orientation_weight: float = 0.35,
    obstacle_margin: float = 0.03,
    obstacle_weight: float = 0.02,
) -> IKResult:
    # IK 主流程：目标 6D 位姿 -> 数值优化 -> 目标关节角。
    # 这里没有手写解析解，而是把问题交给 scipy 的 least_squares 来迭代求解。

    joint_count = len(dh_links)
    start = np.zeros(joint_count, dtype=float) if initial_angles is None else np.asarray(initial_angles, dtype=float)
    if start.shape != (joint_count,):
        raise ValueError(f"initial_angles must contain {joint_count} values")

    # 上下界来自每个关节的限位，避免 IK 解跑到机械臂不允许的角度。
    target_transform = pose_to_transform(target_pose)
    lower_bounds = np.array([link.min_angle for link in dh_links], dtype=float)
    upper_bounds = np.array([link.max_angle for link in dh_links], dtype=float)

    # IK 被写成最小二乘问题：让“当前位置/姿态”和“目标位置/姿态”的误差尽量小。
    result = least_squares(
        _pose_residual,
        x0=np.clip(start, lower_bounds, upper_bounds),
        bounds=(lower_bounds, upper_bounds),
        args=(target_transform, dh_links, orientation_weight, obstacles, obstacle_margin, obstacle_weight),
        max_nfev=max_iterations,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )

    # 求解完成后，用未加权误差判断是否真正到达目标。
    raw_error = _pose_residual(
        result.x,
        target_transform,
        dh_links,
        orientation_weight=1.0,
        obstacles=None,
        obstacle_margin=obstacle_margin,
        obstacle_weight=obstacle_weight,
    )
    position_error = float(np.linalg.norm(raw_error[:3]))
    orientation_error = float(np.linalg.norm(raw_error[3:]))
    success = bool(
        result.success
        and position_error <= position_tolerance
        and orientation_error <= orientation_tolerance
    )

    return IKResult(
        success=success,
        joint_angles=np.asarray(result.x, dtype=float),
        position_error=position_error,
        orientation_error=orientation_error,
        iterations=result.nfev,
    )
