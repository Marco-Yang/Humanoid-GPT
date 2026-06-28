# PICO VR 遥操迁移文档

> 适用仓库：Humanoid-GPT（本仓库）+ GR00T-WholeBodyControl（SONIC，参考）  
> 硬件：Unitree G1 机器人 + PICO 4 Enterprise 头显 + XRoboToolkit Motion Tracker  
> 作者：基于 SONIC（GR00T-WholeBodyControl）遥操逻辑迁移实现

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [原始动捕方案 vs PICO 方案对比](#2-原始动捕方案-vs-pico-方案对比)
3. [SONIC 架构分析](#3-sonic-架构分析)
4. [数据格式对齐](#4-数据格式对齐)
5. [坐标系映射](#5-坐标系映射)
6. [非镜像（Egocentric）约定](#6-非镜像egocentric约定)
7. [全身体 IK 重定向逻辑](#7-全身体-ik-重定向逻辑)
8. [SONIC 风格臂部校准](#8-sonic-风格臂部校准)
9. [三个关键 Bug 修复](#9-三个关键-bug-修复)
10. [最终架构总结](#10-最终架构总结)
11. [参数调优指南](#11-参数调优指南)
12. [与 SONIC 的差异说明](#12-与-sonic-的差异说明)

---

## 1. 背景与目标

### 1.1 原始方案（诺伊腾动捕）

Humanoid-GPT 原始版本使用**诺伊腾光学动捕系统**采集人体姿态数据，流程为：

```
光学动捕 → 全身关节角度 → SMPL 参数 → ONNX 神经网络 → G1 关节目标
```

该方案依赖专业动捕工作室，无法在机器人旁随时使用，且对空间、标记点要求严格。

### 1.2 目标：迁移至 PICO VR

目标是以**消费级 VR 硬件**替代光学动捕，实现可在任意空间使用的遥操系统：

```
PICO 头显 + 控制器 + 脚踝 Tracker → 全身关节目标 → ONNX 神经网络 → G1
```

参考对象：**SONIC（GR00T-WholeBodyControl）**，诺伊腾内部基于 PICO + XRoboToolkit 的遥操实现。

---

## 2. 原始动捕方案 vs PICO 方案对比

| 维度 | 原始动捕 | PICO VR（本实现） |
|------|----------|------------------|
| 硬件 | 专业光学动捕 | PICO 4E + 足部 Tracker |
| 数据接入 | 专有 SDK | TCP JSON（Robotoolkit 1.1.1） |
| 腿部跟踪 | 全身光学关节 | 脚踝 Tracker（joints 10/11）+ Jacobian IK |
| 臂部跟踪 | 光学腕关节 | Body joints 22/23（SMPL 估算）或手柄 fallback |
| 校准 | 静态 T 姿势 | 按钮触发校准（双 Grip 键）|
| 空间限制 | 动捕范围内 | 任意空间 |

---

## 3. SONIC 架构分析

SONIC（`GR00T-WholeBodyControl/gear_sonic/`）是诺伊腾的完整 PICO 遥操方案，包含三条数据流水线：

### 3.1 SONIC 的三个工作模式

```
PLANNER 模式：左摇杆 → 速度指令 → 行走控制
POSE 模式：全身 SMPL 关节 → 神经网络 Policy → G1 关节角
PLANNER_VR_3PT：速度指令 + 3点VR（L腕/R腕/颈）→ 上身 IK + 行走
```

### 3.2 SONIC 的数据链路

```
PICO 运行 XRoboToolkit App
    ↓  xrobotoolkit_sdk（Python pybind11 绑定）
PicoReader 线程（pico_manager_thread_server.py）
    ↓  get_body_joints_pose() → 24 个 SMPL 关节（Unity 坐标系）
compute_from_body_poses()：SMPL 关节 → SMPL 参数 + VR 3 点姿态
    ↓  ZMQ publisher
deploy.sh（神经网络推理端）
    ↓  SMPL 参数 → 神经网络 → G1 qpos
```

### 3.3 本项目与 SONIC 的关键差异

SONIC 使用 **xrobotoolkit_sdk（Python SDK）** 获取数据；  
本项目使用 **TCP JSON 流**（Robotoolkit PC Service 在 63901 端口广播）。

两者底层数据来自同一个 Robotoolkit App，格式基本相同，但：

- SDK 返回的 body joints 可能在 **Unity 坐标系** 下
- TCP JSON 返回的 body joints 在 **OpenXR stage 坐标系** 下

这一差异直接影响坐标变换矩阵的设计（见第 5 节）。

---

## 4. 数据格式对齐

### 4.1 Robotoolkit 1.1.1 TCP 协议

PICO 作为 **TCP 客户端**连接到 PC 的 63901 端口。PC 侧需运行 TCP 服务器。

**帧格式**（二进制）：

```
[0x3F][type: u8][body_len: u32 LE][body: N bytes][ts: u32 LE][dev_id: u32 LE][0xA5]
```

- `type = 0x6D`：Tracking 帧
- body：`{"functionName": "Tracking", "value": "<inner-json-string>"}` （双层 JSON）

**Inner JSON 结构**（`value` 字段解码后）：

```json
{
  "Head":       {"pose": "x,y,z,qw,qx,qy,qz", "status": 1},
  "Controller": {
    "left":  {"pose": "...", "trigger": 0.0, "grip": 0.0, "axisX": 0.0, "axisY": 0.0},
    "right": {"pose": "..."}
  },
  "Body": {
    "joints": [
      {"p": "x,y,z,qw,qx,qy,qz"},   // joint 0: 骨盆
      ...                              // joint 1-23: SMPL 全身
    ]
  },
  "predictTime": 1234567.89
}
```

> **注意**：`Body.joints` 仅在 PICO 佩戴 XRoboToolkit Motion Tracker 并开启 Body Tracking 时才出现。

### 4.2 Body.joints 关键关节索引（实测确认）

| 索引 | 身体部位 | 用途 |
|------|----------|------|
| 0 | 骨盆/腰部 | 根节点位置（`waist_pos`）|
| 10 | 左脚踝 | 左腿 IK 目标（实测确认）|
| 11 | 右脚踝 | 右腿 IK 目标（实测确认）|
| 12 | 颈部 | 上身方向参考（SONIC 风格）|
| 22 | 左手腕/手 | 左臂 IK 目标（SONIC 惯例）|
| 23 | 右手腕/手 | 右臂 IK 目标（SONIC 惯例）|

> **与标准 SMPL 的差异**：标准 SMPL 中脚踝是 joint 7/8，脚是 joint 10/11。  
> Robotoolkit 的 Body.joints 似乎使用了 SMPL+H 或变体排列，实测 joint 10/11 对应脚踝位置。

### 4.3 位姿字符串格式

所有位姿以逗号分隔字符串传输：

```
"x,y,z,qw,qx,qy,qz"
```

- 位置：米
- 四元数：**标量优先**（w,x,y,z），与 SONIC SDK 的标量末尾（x,y,z,qw）**相反**

解析函数（`client.py::_pose_str`）：

```python
def _pose_str(s: str) -> tuple[np.ndarray, np.ndarray]:
    v = [float(x) for x in s.split(",")]
    pos = np.array(v[:3], dtype=np.float32)   # (x, y, z)
    rot = np.array(v[3:7], dtype=np.float32)  # (qw, qx, qy, qz)
    return pos, rot
```

---

## 5. 坐标系映射

这是整个迁移中最关键也最容易出错的部分。

### 5.1 三个坐标系定义

**PICO 世界坐标系（OpenXR stage space）**：

```
+Y 轴：向上
-Z 轴：用户正对方向（前方）
+X 轴：用户右侧方向
原点：用户站立位置地面
```

用户面朝 -Z，因此：
- 用户的**左侧** = PICO **+X**
- 用户的**右侧** = PICO **-X**
- 用户的**前方** = PICO **-Z**

**MuJoCo/G1 机器人世界坐标系**：

```
+Z 轴：向上
+X 轴：机器人前方
+Y 轴：机器人左侧
```

**Unity 坐标系（SONIC SDK 返回）**：

```
+Y 轴：向上
+Z 轴：前方（用户面朝 +Z）
+X 轴：右侧（左手系）
```

用户面朝 +Z，因此 Unity 中用户的**左侧** = **-X**。

### 5.2 本项目的旋转矩阵：R_PICO2ROBOT

将 OpenXR 坐标转为 Robot 坐标：

```python
R_PICO2ROBOT = np.array([
    [0., 0., -1.],   # robot +X (前) = -PICO Z (用户前方)
    [+1., 0., 0.],   # robot +Y (左) = +PICO X (用户左侧)
    [0., 1., 0.],    # robot +Z (上) = PICO Y
], dtype=np.float32)
```

**推导过程**：

| Robot 轴 | 对应 PICO | 原因 |
|----------|----------|------|
| +X（前）| -Z | 用户前方是 PICO -Z |
| +Y（左）| +X | 用户左侧是 PICO +X |
| +Z（上）| +Y | 两者 Y/Z 都是"上" |

### 5.3 SONIC 的旋转矩阵（不可直接复用）

SONIC 的 `pico_manager_thread_server.py` 中：

```python
Q = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
# Unity [x, y, z] → Robot [-x, z, y]
```

**不能直接用于本项目**，原因：
- SONIC SDK 返回 Unity 坐标（用户左侧 = -X）
- 本项目 TCP JSON 返回 OpenXR 坐标（用户左侧 = +X）

若误用 SONIC 的 Q 矩阵，左右方向会镜像翻转（这是早期测试中左腿/左臂运动方向反转的根本原因）。

### 5.4 另一个混淆源：pico_streamer.py 中的 R

SONIC 的 `pico_streamer.py` 中使用：

```python
R_HEADSET_TO_WORLD = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
```

注意 `R[1,0] = -1`，与本项目 `R_PICO2ROBOT[1,0] = +1` 相反。

这是因为 `pico_streamer.py` 是**手柄控制器**的 streamer，使用**镜像（Mirror）约定**：操作员面向机器人，手柄位置做镜像映射。  
本项目（和 SONIC 的 POSE 模式）使用**非镜像（Egocentric）约定**，因此需要 `+1`。

---

## 6. 非镜像（Egocentric）约定

### 6.1 两种遥操约定

**镜像（Mirror）约定**：
- 操作员面对机器人，举起右手 → 机器人左臂抬起（如同镜中映像）
- 适合正面操作，常见于传统工业遥操

**非镜像（Egocentric）约定**：
- 操作员"穿进"机器人，举起右手 → 机器人右臂抬起
- 更自然，适合第一视角 VR 遥操

### 6.2 本项目选择 Egocentric

SONIC 的 POSE/PLANNER_VR_3PT 模式采用 egocentric，本项目保持一致。

**验证方法**：用户抬起左脚，MuJoCo 中机器人左脚抬起 = Egocentric ✓

---

## 7. 全身体 IK 重定向逻辑

### 7.1 输入数据流

```
PicoFrame
├── head_pos, head_rot        → 根节点朝向（偏航角）
├── waist_pos (joint 0)       → 骨盆高度 → 机器人根节点 Z
├── left_foot_pos (joint 10)  → 左脚踝 → 左腿 IK 目标
├── right_foot_pos (joint 11) → 右脚踝 → 右腿 IK 目标
├── left_wrist_pos (joint 22) → 左手腕 → 左臂 IK 目标（SONIC 风格）
├── right_wrist_pos (joint 23)→ 右手腕 → 右臂 IK 目标（SONIC 风格）
└── left_pos, right_pos       → 手柄位置（臂部 fallback）
```

### 7.2 qpos_full 布局（36 维）

```
qpos[0:3]   → 根节点位置 XYZ（固定 X=0, Y=0, Z=root_z）
qpos[3:7]   → 根节点四元数（来自 head_yaw）
qpos[7]     → 腰部俯仰（waist pitch，跟随头部俯仰）
qpos[8:13]  → 腰部其他自由度（默认）
qpos[7:13]  → 左腿 6 关节（Jacobian IK 求解）
qpos[13:19] → 右腿 6 关节（Jacobian IK 求解）
qpos[19:22] → 腰部（部分由 waist_pitch 控制）
qpos[22:29] → 左臂 7 关节（Jacobian IK 求解）
qpos[29:36] → 右臂 7 关节（Jacobian IK 求解）
```

### 7.3 腿部 IK：脚踝目标计算

**公式**：

```
ankle_tgt = root_robot - R_PICO2ROBOT @ (root_pico - foot_pico) * leg_scale
```

**含义**：脚踝目标 = 根节点位置 - （根到脚向量在 PICO 系下，转到 Robot 系，乘以缩放）

**`root_pico` 的计算**：使用 `waist_pos`（joint 0）高度 + 首帧脚踝 XZ 均值（固定在地面位置）：

```python
root_pico = np.array([
    (calib_lf[0] + calib_rf[0]) / 2.0,  # X：校准时脚的中点
    current_root_y,                        # Y：当前骨盆高度
    (calib_lf[2] + calib_rf[2]) / 2.0,  # Z：校准时脚的中点
])
```

**`leg_scale` 计算**：

```python
pico_vleg = waist_y_calib - ankle_avg_y_calib  # PICO 中腰到踝距离（≈0.75m）
leg_scale = robot_vleg / pico_vleg             # 机器人/人类腿长比（≈1.04）
```

### 7.4 根节点高度计算

```python
# 优先使用 joint 0（骨盆 tracker，最准确）
if frame.waist_pos is not None:
    current_root_y = frame.waist_pos[1]  # PICO Y 轴 = 当前骨盆高度
else:
    # fallback：用头部位移估计骨盆位移
    dh = frame.head_pos[1] - calib_head_pos[1]
    current_root_y = calib_root_y + dh

root_z = DEFAULT_ROOT_Z + (current_root_y - calib_root_y) * leg_scale
root_z = max(root_z, 0.50)  # 安全下限
```

### 7.5 IK 求解器：Damped Least-Squares Jacobian

对手臂和腿部均使用 MuJoCo 的 `mj_jacBody` 计算 Jacobian，配合 DLS 阻尼伪逆：

```
dq_task = Jᵀ (JJᵀ + λ²I)⁻¹ Δpos        # 任务空间步长
dq_null = (I - J⁺J) · α(q_default - q)  # 零空间：拉向默认姿势
dq      = step_size × (dq_task + dq_null)
```

参数：`λ=0.05`（阻尼），`step=0.4`，`null_gain=0.1`，`max_iter=30`，`tol=3mm`

---

## 8. SONIC 风格臂部校准

### 8.1 问题：为什么需要专门校准

SONIC 的 `ThreePointPose` 类解决了两个核心问题：
1. 人体腕部位置与机器人默认腕位置不匹配——需要在 T 姿势时对齐
2. 校准时机不对——第一帧时用户可能还未进入站立姿势，导致参考位置错误

早期实现采用**首帧自动校准**，实测出现了"一侧手臂镜像、一侧同步"的问题：  
原因是首帧捕获时两侧手腕位置不对称（用户尚未完全站好），  
导致一侧的 `calib_lw_rel` 恰好接近 T 姿势参考，另一侧则偏差很大，IK 求解到错误位置。

### 8.2 当前实现：按钮触发校准（SONIC ThreePointPose 风格）

**触发方式**：左右 Grip 键同时按下保持约 5 帧（≈50ms），站在 T 姿势中完成校准。  
打印 `[PicoRetargeter] Arm calibration done.` 后即可放开。

**校准捕获（`_capture_arm_calibration`，对应 SONIC `_capture_calibration`）**：

```python
# 1. 颈部方向转换到机器人坐标系，取逆（补偿校准时的身体倾斜）
neck_robot = _SRot.from_matrix(R @ neck_pico.as_matrix() @ R.T)
calib_neck_inv = neck_robot.inv()

# 2. 腕部相对骨盆的向量，转换到机器人坐标系
lw_rel = R @ (left_wrist_pos - waist_pos)   # robot frame

# 3. 应用颈部逆旋转（与 SONIC _capture_calibration 步骤 2 一致）
lw_corrected = calib_neck_inv.apply(lw_rel)

# 4. 计算偏移量（相对 G1 FK 默认腕位）
calib_lw_offset = lw_corrected - default_l_wrist_rel
# 其中 default_l_wrist_rel = MuJoCo FK 默认腕位 - 默认根节点位置
```

**实时跟踪（每帧，对应 SONIC `_apply_calibration`）**：

```python
lw_rel = R @ (left_wrist_pos - waist_pos)   # 当前腕-骨盆向量（robot frame）
lw_calibrated = calib_neck_inv.apply(lw_rel) - calib_lw_offset
# 展开：= default_l_wrist_rel + calib_neck_inv.apply(Δ相对校准时刻的位移)

# 加上当前根节点 XYZ → 世界坐标系下的 IK 目标
l_target = lw_calibrated + qpos[:3]
```

**数学等价关系**（与 SONIC 的 offset 公式完全对应）：

| 时刻 | `lw_calibrated` |
|------|----------------|
| T 姿势（校准时）| `default_l_wrist_rel`（G1 FK 默认位置）|
| 手臂向前伸 Δ | `default_l_wrist_rel + neck_inv.apply(ΔR)` |
| 下蹲 | `lw_calibrated + [0,0,Δz]`（通过 `qpos[:3]` 自动补偿）|

### 8.3 三个校准变量的含义

| 变量 | 类型 | 含义 |
|------|------|------|
| `_calib_neck_inv` | `scipy.Rotation` | 校准时颈部旋转的逆；补偿用户站立时的头/颈偏转 |
| `_calib_lw_offset` | `np.ndarray (3,)` | 左腕偏移量；校准时如果完全在 T 姿势，理论值接近零 |
| `_calib_rw_offset` | `np.ndarray (3,)` | 右腕偏移量（同上） |

校准打印的 `L-off / R-off` 数值越接近 `[0, 0, 0]`，说明校准时 T 姿势越标准。  
若偏移超过 0.1m，建议重新校准。

### 8.4 与 SONIC ThreePointPose 的对比

| 特性 | SONIC ThreePointPose | 本项目 |
|------|---------------------|--------|
| 校准触发 | `calibrate_now()` 外部调用 | 双 Grip 键保持 5 帧 |
| 颈部偏航补偿 | 完整实现（`calib_neck_inv`）| 完整实现（相同逻辑）|
| FK 参考对齐 | Pinocchio/Pink G1 FK | MuJoCo FK（`mj_forward` 默认 qpos）|
| 根节点局部化 | `inv(root_rot).apply(pos - root_pos)` | 仅平移（`wrist - waist`），无旋转 |
| 手腕朝向 | SE3 完整 6D | 仅位置 3D（IK 由 null-space 控制朝向）|
| 臂部缩放 | 无（直接用 FK 对齐）| 无（body joint 模式不用 `arm_scale`）|
| 依赖 | Pinocchio/Pink | MuJoCo + scipy |

> **未实现的根节点局部化**：SONIC 在计算 `wrist - waist` 后，还会应用  
> `inv(root_rot).apply(...)` 将向量转换到身体局部坐标系，以补偿用户转身。  
> 本项目省略此步骤；若用户在使用过程中大幅转动身体，手臂跟踪可能出现偏差。

---

## 9. 三个关键 Bug 修复

在对 MuJoCo 仿真测试时发现并修复了三个映射错误。

### Bug 1：站立时机器人在 MuJoCo 中坐着

**现象**：用户站立，MuJoCo 中 G1 机器人膝盖弯曲、坐姿。

**根本原因**：`_calib_root_y` 初始化为 `1.7 × 0.52 = 0.884`（人体学估计），  
但 PICO 世界坐标原点在地面，`waist_pos[1]`（joint 0 骨盆高度）实测约为 **-1.25m**。

差值：`-1.25 - 0.884 = -2.134`，乘以 `leg_scale ≈ 2.14`（也因此算错了），  
导致 `root_z = 0.78 + (-2.134 × 2.14) = -3.79m` → 触发安全下限 `0.50m` = 坐姿。

**修复**：首帧从 `frame.waist_pos[1]` 直接读取 `_calib_root_y`：

```python
if frame.waist_pos is not None:
    self._calib_root_y = float(frame.waist_pos[1])  # ≈ -1.25m（实测）
```

同时修正 `pico_vleg` 计算：

```python
pico_vleg = self._calib_root_y - feet_avg_y  # ≈ -1.25 - (-2.0) = 0.75m ✓
leg_scale = robot_vleg / pico_vleg           # ≈ 0.78/0.75 = 1.04 ✓
```

修复后，站立时 `current_root_y = _calib_root_y` → `root_z = DEFAULT_ROOT_Z` ✓

### Bug 2：左腿向左运动，机器人左腿向右旋转

**现象**：用户左脚向左移动，MuJoCo 中机器人左腿反方向运动。

**根本原因**：`R_PICO2ROBOT[1, 0] = -1`（原始值），导致：

```
lf2root[0] = root_pico_x - lf_pico_x
# 用户左脚向左（PICO +X 增大）→ lf_pico_x 增大 → lf2root[0] 减小（负）
# ankle_tgt[1] = root_robot[1] - R[1,0] * lf2root[0] * scale
#              = 0 - (-1) * (负) * scale = 负 = 机器人右侧 ✗
```

**修复**：`R_PICO2ROBOT[1, 0] = -1` → `+1`：

```python
R_PICO2ROBOT = np.array([
    [0.,  0., -1.],
    [+1., 0.,  0.],   # 修复前为 -1
    [0.,  1.,  0.],
])
```

验证：用户左脚向左（PICO +X 增大）→ lf2root[0] 减小（负）→  
`ankle_tgt[1] = 0 - (+1) * (负) * scale = 正 = 机器人左侧 ✓`

### Bug 3：向内弯曲左臂，MuJoCo 中向外伸展

**现象**：用户左臂向身体弯曲，机器人左臂反方向运动。

**根本原因**：同 Bug 2，`R_PICO2ROBOT[1, 0] = -1` 同样用于臂部偏移计算：

```python
l_off = R_PICO2ROBOT @ (controller_pos - head_pos)
# 手柄在头部左侧 → PICO X 分量为正
# R[1,0] = -1 → robot Y 分量为负 = 机器人右侧 ✗
```

**修复**：同 Bug 2，改为 `R[1, 0] = +1` 后臂部映射也正确。

---

## 10. 最终架构总结

### 10.1 代码文件结构

```
deploy/pico/
├── client.py          # TCP 服务器 + PicoFrame 解析
└── retarget_pico.py   # PicoFrame → G1 qpos_full 重定向

scripts/
└── pico_mujoco.py     # MuJoCo 仿真测试入口
```

### 10.2 client.py：PicoFrame 数据结构

```python
@dataclass
class PicoFrame:
    timestamp: float
    # HMD
    head_pos, head_rot          # 头部位置和朝向
    # 手柄
    left_pos, left_rot          # 左手柄（臂部 fallback）
    right_pos, right_rot
    left_trigger, left_grip, left_joystick
    right_trigger, right_grip, right_joystick
    # Body joints (可选，需要 Motion Tracker)
    left_foot_pos, left_foot_rot    # joint 10：左脚踝
    right_foot_pos, right_foot_rot  # joint 11：右脚踝
    waist_pos, waist_rot            # joint 0：骨盆
    neck_pos, neck_rot              # joint 12：颈部
    left_wrist_pos, left_wrist_rot  # joint 22：左手腕
    right_wrist_pos, right_wrist_rot # joint 23：右手腕
```

### 10.3 retarget_pico.py：PicoRetargeter 处理流程

```
每帧 retarget(frame) 调用：

1. 首帧校准（仅腿部）
   ├── calib_head_pos     ← head_pos
   ├── calib_lf_pico      ← left_foot_pos（joint 10）
   ├── calib_rf_pico      ← right_foot_pos（joint 11）
   ├── calib_root_y       ← waist_pos[1]（joint 0 骨盆高度）
   └── leg_scale          ← robot_vleg / pico_vleg
   注：臂部校准已移除，改由按钮触发（见步骤 7）

2. 基础 qpos（默认站姿）

3. 根节点朝向
   └── head_yaw → quat_robot（仅偏航，忽略横滚俯仰）

4. 腰部俯仰
   └── head_pitch × 0.5 → waist_pitch（关节 14）

5. 根节点高度
   └── waist_pos[1] → leg_scale → root_z（含安全下限 0.50m）

6. 腿部 Jacobian IK（需要 foot trackers）
   ├── 左脚踝目标 = root - R_PICO2ROBOT @ (root_pico - lf_pico) * scale
   └── 右脚踝目标（同上）

7. 臂部 Jacobian IK（SONIC ThreePointPose 风格）
   ├── 按钮检测：left_grip > 0.5 且 right_grip > 0.5 保持 5 帧
   │   └── 触发 _capture_arm_calibration(frame)
   │       ├── calib_neck_inv ← inv(neck_rot 在 robot 坐标系下)
   │       ├── calib_lw_offset ← neck_inv.apply(R @ (lw-waist)) - default_lw_rel
   │       └── calib_rw_offset ← 同上
   │
   ├── 优先（已校准 + body joints 可用）：SONIC ThreePointPose 风格
   │   ├── lw_rel = R_PICO2ROBOT @ (lw_pos - waist_pos)
   │   ├── lw_cal = calib_neck_inv.apply(lw_rel) - calib_lw_offset
   │   └── l_target = lw_cal + qpos[:3]   # 加当前根节点位置 → 世界坐标
   │
   └── fallback（未校准 或 无 body tracking）：手柄位置相对头部
       └── l_target = chest + shoulder_offset + R_PICO2ROBOT @ (ctrl - head) * arm_scale
```

### 10.4 两种臂部模式自动切换

```python
has_body_arms = (
    frame.left_wrist_pos is not None     # joint 22 可用
    and frame.waist_pos is not None      # joint 0 可用（计算相对位置）
    and self._calib_neck_inv is not None # 按钮校准已完成
)
```

条件满足时自动使用 body joint 模式；否则退化到手柄模式。  
**未校准前机器人手臂保持 G1 默认站姿**，不会因为首帧随机位置导致奇异姿势。

---

## 11. 参数调优指南

### 11.1 臂部缩放 `_ARM_SCALE`

```python
_ARM_SCALE = 0.50 / 0.65  # ≈ 0.769
```

- 分子 `0.50`：G1 机器人手臂从肩到腕的有效伸展距离（米）
- 分母 `0.65`：标准成人手臂从肩到腕的距离（米）

若臂部动作幅度不够：增大分子（如 `0.55 / 0.65`）  
若臂部频繁触发关节限位：减小分子

### 11.2 腿部缩放 `leg_scale`

自动从首帧数据计算，无需手动调整。如需强制设置：

```python
self._leg_scale = 0.78 / 0.75  # robot_vleg / pico_vleg
```

### 11.3 IK 参数

| 参数 | 默认值 | 调大效果 | 调小效果 |
|------|-------|---------|---------|
| `_IK_DAMPING` | 0.05 | 更平滑，精度稍低 | 更精确，可能震荡 |
| `_IK_ALPHA` | 0.4 | 收敛更快 | 更稳定，收敛慢 |
| `_IK_MAX_ITER` | 30 | 精度更高，CPU 消耗更大 | 可能不收敛 |
| `_IK_TOL` | 3mm | — | 精度更低 |

---

## 12. 与 SONIC 的差异说明

### 12.1 主要差异

| 特性 | SONIC | 本项目 |
|------|-------|--------|
| 数据接入 | xrobotoolkit_sdk（Python binding）| TCP JSON 流解析 |
| 腿部控制 | SMPL → 神经网络 Policy | 脚踝 Jacobian IK |
| 臂部控制 | Pink/Pinocchio IK + 完整 SE3 | MuJoCo Jacobian IK（仅位置）|
| 颈部补偿 | ThreePointPose 完整实现 | 实现 inv(neck_rot) 补偿（同 SONIC 逻辑）|
| 坐标系 | Unity（左手系，+Z 前）| OpenXR（+Y 上，-Z 前）|
| 变换矩阵 | Q=[[-1,0,0],[0,0,1],[0,1,0]] | R=[[0,0,-1],[+1,0,0],[0,1,0]] |
| 下游推理 | ZMQ → 专用 SMPL Policy | 直接生成 qpos → ONNX |

### 12.2 为什么不能直接复制 SONIC 的 Q 矩阵

SONIC 的 Q 矩阵假设 body joints 在 Unity 坐标系下（用户左侧 = -X）。  
本项目 TCP JSON 的 body joints 在 OpenXR 坐标系下（用户左侧 = +X）。

直接使用 Q 会导致 X 轴符号反转，即左右镜像错误。

**验证方式**：用户抬起左脚时检查 `joints[10]["p"]` 的 X 分量变化方向。  
若 X 增大 → OpenXR 坐标，需要 `R[1,0] = +1`；  
若 X 减小 → Unity 坐标，需要 `R[1,0] = -1`。

### 12.3 本项目未迁移的 SONIC 功能

以下 SONIC 特性未迁移（超出当前需求范围）：

- **SMPL 全身参数**：SONIC 将 24 个 body joints 转为 SMPL 参数（β, θ）送入专用神经网络
- **Pink/Pinocchio IK**：完整的 SE3 IK，含手腕朝向控制
- **ZMQ 通信架构**：SONIC 将 streamer/policy/sim 解耦为独立进程
- **PLANNER 模式**：摇杆速度指令控制行走（locomotion control）
- **A/B/X/Y 按键状态机**：SONIC 的多模式切换逻辑

---

*最后更新：2026-06-29*
