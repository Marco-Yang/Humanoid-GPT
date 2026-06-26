# Humanoid-GPT 环境配置教程

> 适用仓库：[Marco-Yang/Humanoid-GPT](https://github.com/Marco-Yang/Humanoid-GPT)
> 硬件配置：Unitree G1 机器人 + PICO VR 头显
> 目标平台：Ubuntu 20.04 / 22.04，NVIDIA GPU（CUDA 12.x）

---

## 目录

1. [前提条件](#1-前提条件)
2. [克隆仓库](#2-克隆仓库)
3. [创建 Conda 环境](#3-创建-conda-环境)
4. [安装基础依赖](#4-安装基础依赖)
5. [下载第三方库](#5-下载第三方库)
6. [安装真机依赖](#6-安装真机依赖)
7. [下载模型权重](#7-下载模型权重)
8. [验证安装](#8-验证安装)
9. [仿真运行](#9-仿真运行)
10. [PICO 遥操配置](#10-pico-遥操配置)
11. [G1 真机部署](#11-g1-真机部署)
12. [Jetson 板载部署](#12-jetson-板载部署)
13. [常见问题](#13-常见问题)

---

## 1. 前提条件

| 项目     | 要求                                            |
| -------- | ----------------------------------------------- |
| 操作系统 | Ubuntu 20.04 / 22.04                            |
| GPU      | NVIDIA GPU，CUDA 12.x                           |
| Python   | 3.11 或 3.12（不支持 3.10 及以下，不支持 3.13） |
| Conda    | Miniconda 或 Anaconda                           |
| 存储     | 至少 10 GB 可用空间（含模型权重）               |

---

## 2. 克隆仓库

```bash
git clone https://github.com/Marco-Yang/Humanoid-GPT.git
cd Humanoid-GPT

# 添加上游原始仓库（用于同步更新）
git remote add upstream https://github.com/GalaxyGeneralRobotics/Humanoid-GPT.git
```

目录结构：

```
Humanoid-GPT/
├── tracking/          # 推理核心：ONNX 策略、关键点转换、追踪指标
├── scripts/           # 推理脚本、评估脚本、Gradio demo
├── deploy/            # 部署代码
│   ├── pico/              # PICO VR 遥操（fork 新增）
│   ├── onboard_deploy/    # Jetson 板载部署
│   └── play_track.py      # 统一入口（仿真 + 真机）
├── utils/             # MuJoCo/MJX 仿真、变换工具
├── storage/           # 模型权重、示例轨迹、G1 资产文件
└── notes/             # 教程文档
```

---

## 3. 创建 Conda 环境

```bash
conda create -n h-gpt python=3.12 -y
conda activate h-gpt
```

---

## 4. 安装基础依赖

```bash
pip install -e ".[cuda]"
```

包含：MuJoCo 3.3.7、MuJoCo-MJX、JAX（CUDA12）、PyTorch 2.8、ONNX Runtime GPU、Gradio 等。

> **网络 SSL 问题**：如果 pip 报 `SSLEOFError`，加上 `--trusted-host` 参数：
>
> ```bash
> pip install -e ".[cuda]" \
>     --trusted-host pypi.org \
>     --trusted-host files.pythonhosted.org
> ```

---

## 5. 下载第三方库

thirdparty.zip 包含真机必需的 CycloneDDS 和 Unitree SDK。

```bash
pip install gdown

# 下载（如遇 SSL 错误加 PYTHONHTTPSVERIFY=0）
PYTHONHTTPSVERIFY=0 gdown https://drive.google.com/uc?id=1ArtgwKxVHXTO4KXsKXPLdhy1yAtKKnz9 -O thirdparty.zip

# 备用链接
PYTHONHTTPSVERIFY=0 gdown https://drive.google.com/uc?id=1bfgFhrv6tfuDOkt11AOJAO2IHTRXlYey -O thirdparty.zip

unzip thirdparty.zip && rm thirdparty.zip
```

> 如果 gdown 下载失败，可手动从浏览器下载后 `scp` 传入：
> `https://drive.google.com/file/d/1bfgFhrv6tfuDOkt11AOJAO2IHTRXlYey/view`

解压后结构：

```
thirdparty/
├── GMR-galbot/          # 通用运动重定向（本项目不使用，可忽略）
├── noitom/              # 诺伊腾客户端（本项目不使用，可忽略）
├── cyclonedds/          # DDS 中间件（G1 真机必需）— 仅含 build/install 目录，源码需单独 clone
└── unitree_sdk2_python/ # Unitree G1 SDK（G1 真机必需）
```

> **注意**：thirdparty.zip 里的 `cyclonedds/` 目录可能只有 `build/` 和 `install/` 空文件夹，**不含源码**。编译前必须先 clone 源码（见第 6.1 节）。

---

## 6. 安装真机依赖

> **PICO 遥操不需要 GMR 和 noitom**，跳过它们，只安装以下两项。

### 6.1 编译 CycloneDDS

Unitree SDK 需要 `cyclonedds==0.10.2`，对应 `releases/0.10.x` 分支。

```bash
sudo apt-get install -y cmake build-essential

# 进入 thirdparty 目录，删掉空壳，clone 源码
cd thirdparty
rm -rf cyclonedds
git -c http.sslVerify=false clone \
    https://github.com/eclipse-cyclonedds/cyclonedds \
    -b releases/0.10.x cyclonedds

# 编译安装
cd cyclonedds
mkdir -p build install && cd build

# 注意：conda 环境的 libstdc++ 版本较旧，cmake 需要使用系统库
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
    cmake .. -DCMAKE_INSTALL_PREFIX=../install
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
    cmake --build . --target install -j$(nproc)

cd ../../..   # 回到项目根目录
```

> **为什么要设 `LD_LIBRARY_PATH`**：conda 激活后会把自身的 `lib/` 加入搜索路径，其中的 `libstdc++.so.6` 版本较旧（缺少 `GLIBCXX_3.4.30`），导致 cmake 报错退出。前缀 `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu` 让 cmake 优先使用系统库，仅对当次命令生效，不影响 conda 环境。

### 6.2 安装 Unitree SDK

```bash
export CYCLONEDDS_HOME="$PWD/thirdparty/cyclonedds/install"
pip install -e thirdparty/unitree_sdk2_python

# 永久写入环境变量
echo "export CYCLONEDDS_HOME=\"$HOME/Humanoid-GPT/thirdparty/cyclonedds/install\"" >> ~/.bashrc
source ~/.bashrc
```

### 6.3 安装 TensorRT（真机推理加速）

```bash
pip uninstall onnxruntime -y
pip install onnxruntime-gpu tensorrt-cu12

# 暴露 TensorRT 动态库
cat >> ~/.bashrc << 'EOF'
for _d in "$HOME/miniconda3/envs/h-gpt/lib"/python*/site-packages/{tensorrt_libs,nvidia/*/lib}; do
  [ -d "$_d" ] && export LD_LIBRARY_PATH="$_d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
unset _d
EOF
source ~/.bashrc

# 验证
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# 输出应包含 TensorrtExecutionProvider
```

---

## 7. 下载模型权重

将以下两个 ONNX 文件放到 `storage/ckpts/` 对应路径：

| 权重     | 路径                                                            | 说明         |
| -------- | --------------------------------------------------------------- | ------------ |
| 追踪策略 | `storage/ckpts/pns_wo_priv216.onnx`                           | 全身运动追踪 |
| 行走策略 | `storage/ckpts/G1-Walk/07140632_G1-Walk_v2.0.0_baseline.onnx` | 速度控制行走 |

从 [GitHub Releases](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT/releases) 或论文项目页下载。

---

## 8. 验证安装

```bash
conda activate h-gpt
cd ~/Humanoid-GPT

# 验证 MuJoCo 和 JAX
python -c "import mujoco; import jax; print('MuJoCo:', mujoco.__version__); print('JAX:', jax.devices())"

# 验证 ONNX Runtime
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# 可视化示例轨迹（不需要权重）
python -m scripts.vis --mocap_path storage/test

# 离线推理测试（需要权重）
python -m scripts.inference \
    --load_path storage/ckpts/pns_wo_priv216.onnx \
    --mocap_path storage/test
```

---

## 9. 仿真运行

### 9.1 常用命令

```bash
# 最轻量：不启动 mocap，纯离线轨迹追踪
python -m deploy.play_track --no-mocap

# 离线追踪示例轨迹（keyboard 按 2 进入）
python -m deploy.play_track --track-dir storage/test

# PICO 遥操仿真（见第 10 节）
python -m deploy.play_track --mocap_type pico --pico_port 9864
```

### 9.2 键盘控制

| 按键         | 功能                         |
| ------------ | ---------------------------- |
| `0`        | 行走模式（walk policy）      |
| `1`        | 在线遥操模式（PICO）         |
| `2`–`9` | 离线轨迹追踪（按文件名排序） |
| `W / S`    | 前进 / 后退速度              |
| `A / D`    | 左平移 / 右平移速度          |
| `Q / E`    | 偏航角速度（左转 / 右转）    |
| `R`        | 重置仿真                     |
| `` ` ``      | 退出仿真                     |

---

## 10. PICO 遥操配置

### 10.1 工作原理

```
PICO 头显 + 控制器
    │ UDP JSON（局域网）
    ▼
PicoClient（Linux PC，port 9864）
    │
PicoRetargeter（MuJoCo IK）
    │ qpos_full (36D)
    ▼
追踪策略 ONNX ──▶ G1 机器人
```

| 部位     | 控制来源                                                                       |
| -------- | ------------------------------------------------------------------------------ |
| 双臂     | 控制器位置 → MuJoCo IK → 7 DOF 关节角                                        |
| 躯干朝向 | 头显 yaw 角                                                                    |
| 腿部     | 脚部 tracker → MuJoCo IK → 6 DOF/侧（有 tracker 时）；无 tracker 保持默认站姿 |
| 根节点高度 | 蹲下时由头部/腰部 tracker 估算，驱动机器人跟随                              |
| 行走     | 切换到模式 `0`，用键盘 W/A/S/D 控制                                           |
| 手部开合 | trigger > 0.3 = 握拳，< 0.3 = 张开                                            |

### 10.2 PICO 端 Unity 数据格式

PICO 上的 Unity 应用需每帧向 Linux PC 的 `9864` 端口发送 UDP JSON 包：

```json
{
  "t":    1234567890.123,
  "head": {"p": [x, y, z], "q": [w, x, y, z]},
  "lc":   {"p": [x, y, z], "q": [w, x, y, z], "trig": 0.0, "grip": 0.0, "joy": [0.0, 0.0]},
  "rc":   {"p": [x, y, z], "q": [w, x, y, z], "trig": 0.0, "grip": 0.0, "joy": [0.0, 0.0]},
  "lf":   {"p": [x, y, z], "q": [w, x, y, z]},
  "rf":   {"p": [x, y, z], "q": [w, x, y, z]},
  "wp":   {"p": [x, y, z], "q": [w, x, y, z]}
}
```

| key    | 设备           | 是否必须 |
| ------ | -------------- | -------- |
| `head` | HMD            | 必须     |
| `lc`   | 左手控制器     | 必须     |
| `rc`   | 右手控制器     | 必须     |
| `lf`   | 左脚 tracker   | 可选     |
| `rf`   | 右脚 tracker   | 可选     |
| `wp`   | 腰部 tracker   | 可选     |

`lf`/`rf`/`wp` 缺失时腿部保持默认站姿，程序不报错。

**坐标系（OpenXR stage space）**：+Y 朝上，−Z 朝前，+X 朝右；位置单位米；四元数 `[w, x, y, z]`；原点为用户初始化时头部正下方地面。

### 10.3 网络配置

PICO 和 Linux PC 连同一局域网（建议同一 Wi-Fi 或有线直连）：

```bash
# 查找 Linux PC 局域网 IP
ip addr show
# 在 PICO Unity 应用中填入此 IP，目标端口 9864
```

### 10.4 启动 sim2sim 验证（不接真机）

先在仿真里验证 PICO 遥操效果再上真机：

```bash
conda activate h-gpt
cd ~/Humanoid-GPT

python -m deploy.play_track --mocap_type pico --pico_port 9864
```

MuJoCo viewer 打开后按 **`1`** 进入 PICO 遥操模式。

### 10.5 参数说明

| 参数               | 默认值      | 说明                        |
| ------------------ | ----------- | --------------------------- |
| `--mocap_type`   | `pnlink`  | 设为`pico` 启用 PICO 后端 |
| `--pico_host`    | `0.0.0.0` | 本机监听地址                |
| `--pico_port`    | `9864`    | UDP 监听端口                |
| `--human_height` | `1.7`     | 用户身高，影响手臂缩放比例  |

---

## 11. G1 真机部署

> 详细流程见 [deploy/DEPLOY.md](../deploy/DEPLOY.md)

### 11.1 机器人上电

1. 短按电池键，再长按约 2 秒上电
2. 等待头部指示灯稳定（约 30 秒）
3. 遥控器按 `L2 + R2` 进入 debug 模式

### 11.2 网络连接

```bash
# 主机与 G1 网线直连，配置同子网静态 IP
# G1 默认 IP：192.168.123.164
ping 192.168.123.164

# 查网卡名（传给 --net）
ip addr
```

### 11.3 启动命令

```bash
# PICO 遥操真机
python -m deploy.play_track --real --net <网卡名> \
    --mocap_type pico --pico_port 9864

# 仅行走（不遥操）
python -m deploy.play_track --real --net <网卡名> --no-mocap
```

### 11.4 真机启动序列

| 遥控器按键 | 功能                     |
| ---------- | ------------------------ |
| `start`  | 阻尼模式 → 默认站立姿态 |
| `A`      | 进入运动 / 追踪循环      |
| `select` | 紧急停止，返回阻尼模式   |

---

## 12. Jetson 板载部署

> 详细流程见 [deploy/onboard_deploy/DEPLOY_ONBOARD.md](../deploy/onboard_deploy/DEPLOY_ONBOARD.md)

### 12.1 SSH 连接 G1 Jetson

```bash
ssh unitree@192.168.123.164
# 密码：123
# ROS 版本选择提示时选 1（foxy）
```

### 12.2 Jetson 上安装环境

```bash
# 安装 Miniconda（aarch64）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash && source ~/.bashrc

# 克隆 fork 仓库
git clone https://github.com/Marco-Yang/Humanoid-GPT.git
cd Humanoid-GPT

# 创建环境并安装
conda create -n h-gpt python=3.12 -y
conda activate h-gpt
pip install -e .

# 安装真机依赖（同第 6 节：CycloneDDS + Unitree SDK）
```

### 12.3 每次运行前

```bash
# 锁定 Jetson 时钟，保证控制频率稳定
sudo jetson_clocks

# 启动板载追踪
python -m deploy.onboard_deploy.play_track_onboard \
    --onnx_track storage/ckpts/pns_wo_priv216.onnx
```

---

## 13. 常见问题

### Q: pip 安装报 SSLEOFError

当前网络 HTTPS 被中间设备截断，两种方法：

```bash
# 方法一：跳过 SSL 验证
pip install -e ".[cuda]" \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org

# 方法二：关闭验证（全局，慎用）
pip config set global.trusted-host "pypi.org files.pythonhosted.org"
```

### Q: gdown 下载 Google Drive 失败

```bash
# 关闭 Python SSL 验证
PYTHONHTTPSVERIFY=0 gdown <url> -O thirdparty.zip

# 或用 curl（-k 跳过证书）
curl -k -L "https://drive.google.com/uc?export=download&id=1bfgFhrv6tfuDOkt11AOJAO2IHTRXlYey&confirm=t" -o thirdparty.zip
```

如果完全无法访问 Google Drive，在有网络的机器下载后 scp 传入：

```bash
scp thirdparty.zip adam@<linux-pc-ip>:~/Humanoid-GPT/
```

### Q: git push 报 TLS 错误

```bash
git config http.sslVerify false
git push origin main
```

### Q: MuJoCo viewer 无法打开

```bash
# 检查 OpenGL
python -c "import OpenGL; print(OpenGL.__version__)"

# 无显示器 / headless 环境用 EGL
MUJOCO_GL=egl python -m scripts.inference ...
```

### Q: JAX 看不到 GPU

```bash
python -c "import jax; print(jax.devices())"
nvidia-smi   # 检查 CUDA 版本是否为 12.x
```

### Q: PICO 数据收不到

```bash
# 确认端口在监听
ss -ulnp | grep 9864

# 用本机模拟 PICO 发包测试
python3 -c "
import socket, json, time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
pkt = json.dumps({
    't': time.time(),
    'head': {'p': [0, 1.7, 0],    'q': [1, 0, 0, 0]},
    'lc':   {'p': [-0.3, 1.4, 0.3], 'q': [1, 0, 0, 0], 'trig': 0, 'grip': 0, 'joy': [0, 0]},
    'rc':   {'p': [0.3,  1.4, 0.3], 'q': [1, 0, 0, 0], 'trig': 0, 'grip': 0, 'joy': [0, 0]},
})
s.sendto(pkt.encode(), ('127.0.0.1', 9864))
print('已发送测试包')
"
```

### Q: cmake 报 `GLIBCXX_3.4.30' not found`

conda 激活后其旧版 `libstdc++` 被 cmake 优先加载，在所有 cmake 命令前加前缀即可：

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu cmake .. -DCMAKE_INSTALL_PREFIX=../install
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu cmake --build . --target install -j$(nproc)
```

或者直接在 conda 环境里安装兼容版本：

```bash
conda install -c conda-forge cmake
```

### Q: cyclonedds 目录下只有 build/install，没有源码

thirdparty.zip 可能不包含 cyclonedds 源码，需要手动 clone：

```bash
cd thirdparty
rm -rf cyclonedds
git -c http.sslVerify=false clone \
    https://github.com/eclipse-cyclonedds/cyclonedds \
    -b releases/0.10.x cyclonedds
```

### Q: conda 环境名不显示在终端

```bash
conda config --set changeps1 True
source ~/.zshrc   # 或 source ~/.bashrc
```

---

## 快速参考

```bash
# 激活环境
conda activate h-gpt
cd ~/Humanoid-GPT

# 仿真验证（无设备）
python -m deploy.play_track --no-mocap

# sim2sim：PICO 遥操仿真
python -m deploy.play_track --mocap_type pico --pico_port 9864

# 真机：PICO 遥操
python -m deploy.play_track --real --net <网卡名> --mocap_type pico --pico_port 9864

# 推送代码到 fork
git config http.sslVerify false
git push origin main
```
