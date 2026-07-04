# GPT5.5 + Codex 工程调试流程

## 目标
把标定工程变成可复现流水线，而不是一次性脚本。每次修改只围绕一个问题：路径、时间对齐、数据划分、单个训练方法、实时推理。

## 建议工作流

1. 先跑数据检查

```bash
python scripts/00_check_dataset.py
```

把终端输出和目录截图发给GPT5.5，确认 `new_output/1..109` 与 `1.txt..109.txt` 是否匹配。

2. 跑时间对齐

```bash
python scripts/01_preprocess_align.py
```

重点看：

- `outputs/aligned/alignment_table.csv` 样本数是否合理；
- `outputs/aligned/unmatched_table.csv` 是否出现大量解析失败；
- `outputs/debug_alignment/episode_xxx_contact_debug.png` 中图像接触点和力接触点是否落在合理位置。

3. 跑8:2划分

```bash
python scripts/02_split_dataset.py
```

要求：同一个片段只能出现在训练集或验证集其中之一，不能混到两边。

4. 按四种方法分别训练

```bash
python scripts/03_train_method1_img0_imgn.py
python scripts/04_train_method2_optical_flow.py
python scripts/05_train_method3_temporal.py --window 3
python scripts/05_train_method3_temporal.py --window 5
python scripts/05_train_method3_temporal.py --window 7
python scripts/06_train_method4_ref_temporal.py --window 3
python scripts/06_train_method4_ref_temporal.py --window 5
python scripts/06_train_method4_ref_temporal.py --window 7
```

每个run目录重点看：

- `training_log.csv`：训练/验证loss是否下降；
- `loss_curve.png`：是否过拟合；
- `val_predictions_best.csv`：F、theta、alpha分别错在哪里。

5. 用GPT5.5做结果汇总

把所有 `training_log.csv` 和 `val_predictions_best.csv` 的末几行发给GPT5.5，要求按下面格式输出：

```text
方法 | 输入 | MAE_F | MAE_theta | MAE_alpha | 训练稳定性 | 是否适合实时部署 | 下一步优化
```

6. 用Codex做局部代码修改

只给Codex明确的小任务，例如：

```text
只修改 src/gelsight_force_calib/preprocess.py，增强图像接触点检测鲁棒性。
要求：不要改训练接口；保留alignment_table.csv字段；新增参数写入config/default.yaml。
```

不要一次性让Codex重写整个工程，否则容易破坏已经能跑通的路径和接口。

## 快速定位问题

- 图片文件名解析失败：检查是否符合 `gelsight_F_theta_alpha_YYYYMMDD_HHMMSS_mmm.png`。
- 对齐样本太少：检查接触点曲线图，可能阈值过高或力txt时间范围太短。
- CUDA显存不足：把 `config/default.yaml` 里的 `batch_size` 改成1，或把 `image_size` 改成160。
- 实时预测漂移：按 `r` 重新设置Img0；如果仍漂移，优先训练方法4。
