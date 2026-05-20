from __future__ import annotations

from pathlib import Path


# -----------------------------
# Project paths
# -----------------------------

MODULE_DIR = Path(__file__).resolve().parent

# 规划输入 JSON。
# 里面放 start_joint_angles_rad、goal_pose_6d、可选 pre_grasp_pose_6d。
# x/y/z 单位是 mm，roll/pitch/yaw 单位是 rad。
DEFAULT_REQUEST_JSON = MODULE_DIR / "inputs" / "planning_request.example.json"

# 障碍物场景 JSON。
# 里面放 box/sphere 等障碍物，位置和尺寸通常是 mm，会在读取时转换成 m。
DEFAULT_SCENE_JSON = MODULE_DIR / "scenes" / "obstacle_scene.example.json"


# -----------------------------
# Output files
# -----------------------------

# 生成给机器人执行/预览的脚本文本。
DEFAULT_SCRIPT_PATH = MODULE_DIR / "outputs" / "generated_rrt_motion.script"

# 规划结果 JSON：包含每个路径点的末端 6D 位姿和 6 个关节角。
DEFAULT_TRAJECTORY_JSON_PATH = MODULE_DIR / "outputs" / "rrt_trajectory_output.json"

# 静态 3D 场景图：显示障碍物、机械臂、目标点、预抓取点。
DEFAULT_SCENE_IMAGE_PATH = MODULE_DIR / "outputs" / "rrt_3d_scene.png"

# 3D 动画 GIF：只有 RRT 找到碰撞检测通过的路径时才生成。
DEFAULT_ANIMATION_PATH = MODULE_DIR / "outputs" / "rrt_3d_animation.gif"

# 旧版路径图：只画末端轨迹和障碍物。
DEFAULT_PATH_IMAGE_PATH = MODULE_DIR / "outputs" / "rrt_ik_path.png"


# -----------------------------
# RRT / RRT-Connect parameters
# -----------------------------

# RRT 每次扩展的关节空间步长，单位 rad。
# 值越大，搜索更快但路径更粗；值越小，搜索更细但更慢。
RRT_STEP_SIZE = 0.16

# 采样时直接采目标关节角的概率，范围 0-1。
# 值越大越倾向于朝终点连；障碍复杂时过大可能缺少探索。
RRT_GOAL_SAMPLE_RATE = 0.7

# RRT-Connect 最大扩展次数。
# 找不到路径时可以增大，但更推荐先检查终点是否碰撞。
RRT_MAX_ITERATIONS = 8000

# 碰撞检测时沿关节空间边插值的分辨率，单位 rad。
# 值越小越严格、越慢；值越大越快但可能漏检。
RRT_COLLISION_CHECK_RESOLUTION = 0.04

# 全局安全距离，单位 m。
# 碰撞判定阈值 = 这个全局距离 + 每个障碍物 JSON 里的 safety_margin。
RRT_SAFETY_MARGIN = 0.03

# RRT 找到路径后的 shortcut 平滑次数。
# 值越大，节点通常越少，但会多做碰撞检测。
RRT_SMOOTHING_ITERATIONS = 800

# 随机种子。固定后同一输入更容易复现同一条路径。
RRT_RNG_SEED = 4


# -----------------------------
# Pre-grasp parameters
# -----------------------------

# 自动生成预抓取点时使用的距离，单位 mm。
# 如果 request JSON 已经给出 pre_grasp_pose_6d，这个距离会被忽略。
DEFAULT_PREGRASP_DISTANCE_MM = 0.0


# -----------------------------
# IK parameters
# -----------------------------

# 是否允许 IK 不满足误差阈值时继续规划到“近似解”。
# 真实避障/执行建议保持 False。
ALLOW_APPROXIMATE_IK = False


# -----------------------------
# Visualization parameters
# -----------------------------

# GIF 动画帧率。
ANIMATION_FPS = 4


# -----------------------------
# Robot driver/script parameters
# -----------------------------

# 机器人控制器 IP。
ROBOT_HOST = "192.168.1.200"

# CS612A Dashboard 端口，用于文本命令。
DASHBOARD_PORT = 29999

# CS612A 主端口/状态端口，当前脚本发送也用这个端口。
STATE_PORT = 30001

# socket 连接超时时间，单位秒。
SOCKET_TIMEOUT_S = 3.0

# 生成脚本里的 MoveJ 速度，单位 deg/s。
MOVEJ_SPEED_DEG_S = 20.0

# 生成脚本里的 MoveJ 加速度，单位 deg/s^2。
MOVEJ_ACCEL_DEG_S2 = 30.0
