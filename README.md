# 题目 A：让 AI 预测下一阵风

本仓库实现二维矩形断面绕流速度场的短时预报。模型先用 POD 提取空间模态，再用带岭回归的 AR 模型递推模态系数。给定预测起点及之前的历史帧，模型预测随后 20 帧的 $(w,u)$ 速度场。

## 提交入口

- 答辩稿：`deliverables/A题_POD-AR_答辩汇报.pptx`
- GitHub 源码：https://github.com/tuofeilun711/wind-prediction

## 数据边界

- `train`：计算归一化统计量、POD 模态和 AR 参数。
- `validation`：唯一调参数据。模型选择主扫描使用 `99/199/299` 三个窗口；冻结配置的验证参考结果再报告 `99/199/299/379` 四窗结果。
- `test`：配置冻结后只运行官方 8 窗口共同参照评价。
- `transition`：只用于第三问跨工况压力测试。

候选和评价窗口只使用起点及之前的速度场；起点之后的真值只参与误差计算。

原始 HDF5 文件不进入代码仓库：

```text
文件名：bridge_a_flow_light.h5
SHA-256：b7253fa216b932a851d5a8d8213894f580cda8d03b120c6b90f08f3b715358ff
```

## 方法

1. 只保留 `mask == 1` 的 18,282 个流体网格，把两通道速度拼成 36,564 维状态。
2. 对训练快照做去均值 SVD，保留 50 个 POD 模态，解释 81.5% 的训练能量。
3. 在全部训练轨迹的模态系数上联合拟合 AR(3)（pooled），但每个训练样本都限制在单条轨迹内，不构造跨轨迹边界样本。
4. 用岭回归稳定模态递推；最终配置为 `ridge_alpha=3.0`。α=0.01 或 0.1 时，AR(3) 在 k=50/80 已明显发散；α=3.0 后所测 k=50/80 均可稳定递推。
5. 递推 20 帧后重构完整流场，并在流体网格上计算双通道 RMSE。
6. 第二问额外保存训练 AR 单步残差，在 validation 四个窗口上使用固定种子生成 20 个残差自助集成成员，报告离散度以及 90%/95% 经验覆盖率。

模型配置保存在 `configs/model_config.json`。AR 的 `R²=0.9999` 是**训练集单步系数拟合 R²**，不等价于 20 帧递推精度。

## 核心结果

| 指标 | 结果 |
| --- | ---: |
| validation 四窗平均 RMSE | 0.102480 |
| test 八窗平均 RMSE | 0.101645 |
| test 持续性基线 RMSE | 0.177252 |
| 相对持续性基线改善 | 42.7% |
| validation 集成平均 RMSE | 0.103278 |
| validation 集成平均离散度 | 0.031769 |
| validation 90%/95% 经验覆盖率 | 47.3% / 52.6% |
| 集成离散度与实际 RMSE 相关系数 | 0.648 |
| POD 能量占比（50 模态） | 81.5% |
| AR 谱半径 | 1.000062 |

谱半径略大于 1，严格来说属于弱不稳定而不是渐近稳定；在已测试的 20 帧时域内没有出现明显发散，但不能据此宣称模型可无限长时外推。集成区间明显低估误差，仅可作为相对风险信号，不能当作已校准的 90%/95% 置信区间。

## 项目结构

```text
wind-prediction/
├── main.py                  # 冻结配置拟合 + validation/test 评价 + 写 results/
├── data/h5_loader.py        # HDF5 读取、归一化、流体点
├── models/pod_ar.py         # POD(SVD) + 轨迹内样本池化 AR(3) 岭回归
├── eval/metrics.py          # 固定窗口 RMSE、validation 选择协议
├── experiments/
│   ├── q1_baseline.py       # 仅 validation 超参数选择
│   ├── q2_reliability.py    # validation 误差增长与分量误差
│   ├── q3_transfer.py       # transition 留出测试 + 排除 reverse_* 严格对照
│   └── sweep_stable.py      # k / AR 阶 / ridge α 的 validation 扫描
├── configs/model_config.json
├── results/                 # 汇总、逐时刻 RMSE、集成可靠性及第三问对照结果
├── tests/test_core.py       # 5 个单元测试
├── deliverables/A题_POD-AR_答辩汇报.pptx
└── requirements.txt         # numpy + h5py
```

## 复现

建议 Python 3.12。**基础复现只需要 numpy + h5py**：

```bash
python -m pip install numpy h5py
python main.py                 # 训练 + 评价 + 重写 results/
python -m unittest discover -s tests -v
```

`main.py` 会按 `configs/model_config.json` 拟合（50 模态 / AR(3) / 岭 α=3.0），
并把 `validation_reference.csv`、`common_reference.csv`、`common_reference_per_step.csv`、
`test_error_growth.csv`、`ensemble_validation.csv`、`transition_eval.csv`、
`transition_summary.json`、`reliability_error_growth.csv`、
`reliability_summary.json`、`summary.json`、`model_diagnostics.json` 写进 `results/`；
加 `--no-write` 则只打印不落盘。

`deliverables/A题_POD-AR_答辩汇报.pptx` 与本次冻结配置（50 模态 / AR(3) / 岭 α=3.0）口径一致。
演示图表已嵌入最终答辩稿，仓库不额外保存重复的导出图片或自动摘要文件。

超参数扫描与可靠性实验（只需 numpy + h5py，各写自己的 csv）：

```bash
python experiments/q1_baseline.py
python experiments/q2_reliability.py
python experiments/q3_transfer.py
python experiments/sweep_stable.py
```

注意：`main.py`、`q2_reliability.py` 和 `q3_transfer.py` 会写入 `results/` 中的对应结果文件。
想重新生成第一、二问的基准状态，重跑 `python main.py` 即可。

数据文件放在仓库根目录时，上述命令无需额外参数。也可以使用：

```bash
python main.py --h5 /path/to/bridge_a_flow_light.h5
```

Windows 上同样直接用 `python main.py` 即可，脚本按自身位置定位仓库，不依赖固定磁盘路径。

## 第二问与第三问

第二问只在 validation 上分析：

- 误差从第 1 帧 `0.08957` 增长到第 20 帧 `0.11975`；
- 持续性基线同窗平均 `0.17579`；
- 20 成员集成平均 RMSE `0.10328`，平均离散度 `0.03177`，离散度与实际误差相关系数 `0.648`；
- 名义 90%/95% 区间的经验覆盖率仅为 `47.3%/52.6%`，说明该集成区间明显偏窄，只能作为相对风险信号；
- 输出 `results/reliability_error_growth.csv`、`ensemble_validation.csv` 与 `reliability_summary.json`。

第三问不把 transition 与验证集作简单百分比“退化”比较，因为轨迹长度和工况不同。模型分别报告 transition 两条轨迹的绝对误差：

- `reverse_a0`：15 窗，平均 RMSE `0.10210`；
- `reverse_a3`：11 窗，平均 RMSE `0.10281`；
- 同期持续性基线分别约 `0.17524`、`0.18845`。

注意 `reverse_a1` 本身属于 train，因此全训练模型在 transition 上的结果只是“两条留出轨迹”测试，不是零样本反向工况测试。为补足这一点，`q3_transfer.py` 增加严格对照：排除所有 `reverse_*` 训练轨迹后重新训练，得到的 `reverse_a0/reverse_a3` 平均 RMSE 分别为 `0.10375/0.10492`，总体由 `0.10246` 小幅升到 `0.10433`。这支持同一几何和运行工况族内的有限短时迁移，但仍不足以证明物理普适性。

## 局限

- 只使用最后 10 帧历史；没有比较更长历史能否进一步降低误差。
- POD 只保留 81.5% 训练能量，小尺度结构会被截断。
- 谱半径略大于 1，属于弱不稳定；长期预测仍需额外阻尼或约束。
- 残差自助集成的经验覆盖率明显低于名义水平，当前区间未校准。
- transition 样本较短，第三问只说明有限跨工况迁移能力。
- RMSE 不能单独反映涡结构是否合理，应结合速度场和空间误差图一起解释。

## AI 使用说明

AI 用于辅助代码检查、重构建议、图表与文档排版。数据划分、调参边界、模型公式、误差指标和最终配置均由仓库脚本执行，结果文件由同一冻结配置生成。
