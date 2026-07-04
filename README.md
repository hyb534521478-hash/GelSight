# GelSight 视触觉传感器三维力标定工程（Ubuntu22.04）

## 1. 这套工程解决什么问题

当前数据有两套时间轴：

- 图片：`force/new_output/1..109`，每张图片文件名里带绝对时间戳；
- 拉力计：`force/1.txt..109.txt`，第一列是相对时间`s`，第二列是力值`N`。

本工程先做时间对齐，再按8:2划分训练/验证集，最后分别训练四种模型：

1. `[Img_0, Img_n] -> [F, theta, alpha]`，ResNet18回归；
2. `光流点位偏移 -> [F, theta, alpha]`，Farneback光流 + MLP；
3. `[Img_n-k+1 ... Img_n] -> [F, theta, alpha]`，时间窗口长度支持`3/5/7`；
4. `[Img_0, Img_n-k+1 ... Img_n] -> [F, theta, alpha]`，初始图 + 时间窗口。

实时软件支持按 `r` 复位当前帧为新的 `Img_0`。

---

## 2. 目标目录

用户机器上的目标目录：

```bash
~/桌面/视触觉传感器标定/Gelsight标定
```

数据目录应保持为：

```bash
~/桌面/视触觉传感器标定/Gelsight标定/force/new_output/1
~/桌面/视触觉传感器标定/Gelsight标定/force/new_output/2
...
~/桌面/视触觉传感器标定/Gelsight标定/force/new_output/109

~/桌面/视触觉传感器标定/Gelsight标定/force/1.txt
~/桌面/视触觉传感器标定/Gelsight标定/force/2.txt
...
~/桌面/视触觉传感器标定/Gelsight标定/force/109.txt
```

工程目录放在：

```bash
~/桌面/视触觉传感器标定/Gelsight标定/gelsight_force_calibration_project
```

解压后可以执行：

```bash
cd 解压出来的/gelsight_force_calibration_project
bash install_to_desktop.sh
cd ~/桌面/视触觉传感器标定/Gelsight标定/gelsight_force_calibration_project
```

---

## 3. Conda环境

建议环境名：`gelsight_force`

```bash
cd ~/桌面/视触觉传感器标定/Gelsight标定/gelsight_force_calibration_project
bash scripts/setup_conda_env.sh gelsight_force
conda activate gelsight_force
```

如果PyTorch GPU安装失败，不要硬改代码，先按本机CUDA版本重新安装PyTorch GPU版，再执行：

```bash
pip install -r requirements.txt
```

检查GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 4. 先检查数据

```bash
python scripts/00_check_dataset.py
```

正常情况：能发现接近109个片段，每个片段有图片，对应的 `1.txt..109.txt` 存在。

---

## 5. 时间对齐预处理

```bash
python scripts/01_preprocess_align.py
```

输出：

```bash
outputs/aligned/alignment_table.csv
outputs/aligned/unmatched_table.csv
outputs/debug_alignment/episode_001_contact_debug.png
```

核心逻辑：

- 图像侧：第一张图作为 `Img_0`，计算 `d_t=max(|I_t-I_0|)`，检测首次突变点；
- 力侧：检测力值首次明显离开零点的位置；
- 两个首次接触点视为同一物理事件；
- 起点对齐后，只保留图片和力曲线共同覆盖的时间区间。

---

## 6. 训练/验证8:2划分

```bash
python scripts/02_split_dataset.py
```

输出：

```bash
outputs/splits/dataset_split.csv
outputs/splits/train_episodes.txt
outputs/splits/val_episodes.txt
```

这里按片段划分，不按单张图片随机划分。这样可以避免同一次按压过程同时进入训练集和验证集，验证结果更真实。

---

## 7. 四种训练方式

### 方法1：初始图 + 当前图

```bash
python scripts/03_train_method1_img0_imgn.py
```

输入：`[Img_0, Img_n]`

输出：`[F, theta, alpha]`

适合先跑通基线。

### 方法2：光流点位偏移

```bash
python scripts/04_train_method2_optical_flow.py
```

输入：`Img_0 -> Img_n` 的光流点阵位移特征。

输出：`[F, theta, alpha]`

优点是可解释性更强，能看到形变位移和力方向的对应关系；缺点是精度可能不如深度网络端到端。

### 方法3：时间窗口

```bash
python scripts/05_train_method3_temporal.py --window 3
python scripts/05_train_method3_temporal.py --window 5
python scripts/05_train_method3_temporal.py --window 7
```

输入：`[Img_n-k+1 ... Img_n]`

输出：`[F, theta, alpha]`

适合判断方向存在滞后或单帧方向信息不足的情况。

### 方法4：初始图 + 时间窗口

```bash
python scripts/06_train_method4_ref_temporal.py --window 3
python scripts/06_train_method4_ref_temporal.py --window 5
python scripts/06_train_method4_ref_temporal.py --window 7
```

输入：`[Img_0, Img_n-k+1 ... Img_n]`

输出：`[F, theta, alpha]`

这是最推荐重点比较的方法。它同时有无受力参考图和动态变化窗口，理论上比方法1和方法3信息更完整。

---

## 8. 一键跑完整训练

```bash
bash scripts/run_all_train.sh
```

训练结果统一保存在：

```bash
outputs/runs/
```

每个run目录包含：

```bash
best.pt
training_log.csv
loss_curve.png
val_predictions_best.csv
run_meta.json
```

---

## 9. 复评某个模型

```bash
python scripts/07_evaluate_checkpoint.py \
  --checkpoint outputs/runs/某个run目录/best.pt \
  --split val
```

输出：

```bash
outputs/runs/某个run目录/val_predictions_eval.csv
```

看三个核心指标：

```text
MAE_F
MAE_theta
MAE_alpha
```

---

## 10. 实时运行软件

先训练出一个 `best.pt`，再运行：

```bash
python scripts/08_realtime_run.py \
  --checkpoint outputs/runs/某个run目录/best.pt \
  --camera 0
```

窗口按键：

```text
r：把当前帧设为新的Img_0，相当于复位/重新标零
q：退出
```

如果更换了GelSight初始接触状态、光照、装置位置，必须按 `r` 复位，否则预测会漂移。

---

## 11. 常见参数调整

配置文件：

```bash
config/default.yaml
```

显存不够：

```yaml
training:
  batch_size: 1
  image_size: 160
```

训练太慢：

```yaml
training:
  epochs: 30
  image_size: 160
```

接触点检测不准：

```yaml
alignment:
  image_threshold_k: 6.0
  image_min_jump: 5.0
  force_threshold_k: 6.0
  force_min_jump_n: 0.02
```

---

## 12. GitHub上传

第一次上传新仓库：

```bash
cd ~/桌面/视触觉传感器标定/Gelsight标定/gelsight_force_calibration_project

git init
git add .
git commit -m "init gelsight force calibration project"
git branch -M main
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

后续更新：

```bash
git add .
git commit -m "update training and realtime pipeline"
git push
```

从GitHub下载到本地桌面：

```bash
cd ~/桌面/视触觉传感器标定/Gelsight标定
git clone https://github.com/你的用户名/你的仓库名.git gelsight_force_calibration_project
cd gelsight_force_calibration_project
```

---

## 13. GPT5.5 + Codex协同建议

详细流程见：

```bash
docs/GPT55_Codex_调试流程.md
```

最务实的做法：

1. GPT5.5负责读日志、判断问题、给出修改策略；
2. Codex只做单文件、单目标修改；
3. 每次修改后立刻运行对应脚本验证；
4. 四种方法全部跑完后，用 `training_log.csv` 和 `val_predictions_best.csv` 横向比较。

不要让Codex一次性重写整个工程，容易破坏已经跑通的数据路径、CSV字段和checkpoint结构。
