# MoEBench

面向 Linux 环境下 **基准测试（Benchmark）与系统状态** 的研究项目：从「专家池」式的大型套件（如 UnixBench、Phoronix Test Suite）出发，结合系统特征与智能方法，探索按需执行子集、缩短总耗时的可能路径。

本仓库包含：

| 内容 | 说明 |
|------|------|
| `moebench/` | **系统状态特征采集**：静态特征（CPU/内存/存储/内核/调度等）与动态特征（短时 warmup + `perf` / `/proc` / 可选 eBPF） |
| `byte-unixbench/` | UnixBench 测试套件 |
| `phoronix-test-suite/` | Phoronix Test Suite |

## 环境要求

- **Python**：3.9+（标准库即可，无额外 pip 依赖）
- **可选**：`perf`（动态特征中的 PMU/软件事件，需内核允许使用性能事件）
- **可选**：`bpftrace`（动态特征中的调度相关 tracepoint，通常需 root）
- **常用系统工具**：`lscpu`、`numactl`、`lsblk`、`findmnt`、`sysctl` 等（部分缺失时对应字段会为空或跳过）

### 一键安装依赖

仓库提供了跨发行版安装脚本（Ubuntu/Debian、Fedora/RHEL、Arch、openSUSE）：

```bash
cd /path/to/MoEBench
./scripts/install_dependencies.sh
```

常用可选参数：

```bash
# 不安装 bpftrace
./scripts/install_dependencies.sh --no-bpftrace

# 只装依赖，不编译 UnixBench
./scripts/install_dependencies.sh --no-build
```

## 运行方式

在仓库根目录执行，使 `moebench` 包可被解析：

```bash
cd /path/to/MoEBench
python3 -m moebench all -o moebench-features.json
# 需要更高权限采集动态特征时
python3 -m moebench --sudo all -o moebench-features.json
```

不写 `-o` 时，JSON 会输出到**标准输出**；使用 `-o` 时默认只写文件，并在**标准错误**打印一行 `Wrote <路径>`。

## 命令行用法

```text
python3 -m moebench [all|static|dynamic] [选项]
```

- **`all`**（默认）：采集静态 + 动态（会执行短时 warmup，耗时相对较长）
- **`static`**：仅静态特征（较快）
- **`dynamic`**：仅动态特征（含 warmup 与 perf/回退逻辑）

### 常用选项

| 选项 | 说明 |
|------|------|
| `-o FILE`, `--output FILE` | 将结果写入单个 **UTF-8** JSON 文件，默认 **缩进美化**（`--indent`，默认 2） |
| `--warmup-s SEC` | 动态 warmup 时长（秒），默认 `3.0` |
| `--proc-sample-s SEC` | `/proc` 等采样间隔（秒），默认 `0.5` |
| `--mem-mb N` | warmup 工作集大小（MiB），默认 `64` |
| `--no-ebpf` | 跳过 bpftrace 探针 |
| `--indent N` | JSON 缩进；`0` 为单行 |
| `--print` | 与 `-o` 同时使用时，再向 stdout 打印同一份 JSON |
| `--envelope` | 无 `-o` 时也在 stdout 输出带 `meta` 的统一结构 |
| `--raw` | 不包 `meta` 信封，仅原始采集结构 |

### 常用命令示例

```bash
# 静态 + 动态，写入美化后的单一 JSON
python3 -m moebench all -o moebench-features.json

# 仅静态（快速，适合频繁快照）
python3 -m moebench static -o static.json

# 仅动态，缩短 warmup
python3 -m moebench dynamic --warmup-s 1 --no-ebpf -o dynamic.json

# 紧凑单行 JSON（便于脚本管道）
python3 -m moebench static --indent 0 -o static.min.json

# 查看帮助（若系统注册了 python3 的 argparse 帮助）
python3 -m moebench --help
```

## 输出 JSON 结构（使用 `-o` 或 `--envelope` 时）

默认会包一层 **meta**，便于实验记录与版本对齐：

```json
{
  "meta": {
    "moebench_version": "0.1.0",
    "collected_at_utc": "2026-03-22T12:00:00+00:00",
    "mode": "all"
  },
  "static": { ... },
  "dynamic": { ... }
}
```

- `mode` 为 `static` 时仅有 `static` 字段；为 `dynamic` 时仅有 `dynamic` 字段。
- 使用 `--raw` 时无 `meta`，形状与 `collect_*()` 返回值一致。

## Python API

```python
from moebench import collect_all, collect_static, collect_dynamic

static = collect_static()
dynamic = collect_dynamic(warmup_s=3.0, proc_sample_s=0.5, enable_ebpf=True, mem_mb=64)
both = collect_all()
```

## UnixBench 实验流水线（xi / yi / ti 与专家元数据）

默认与官方 `perl Run` 一致，跑 **system index** 子项（`dhry2reg`、`whetstone-double`、文件与管道/进程/系统调用、shell 脚本等，见 `INDEX_SUITE_TEST_IDS`）。**终端会完整显示** UnixBench 的标准输出（子进程继承当前终端，不经管道截获）。

1. 采集 **xi**：`collect_all()`（静态 + 动态特征，可用 `--no-features` 跳过）。
2. 在 `byte-unixbench/UnixBench` 下执行 **`perl Run`**，并通过 `UB_OUTPUT_FILE_NAME` 固定本次结果文件名。
3. 解析生成的纯文本报告，得到 **yi**（各子项分数、Index、System Benchmarks Index Score）与 **ti**（各子项耗时秒数，按并行副本数分组）。
4. 合并 **专家集合 E**：每个子项对应 `e_001…`，含类别（CPU / IO / syscall / thread 等）、占位字段（历史均值/方差、权重、相关性、跨硬件稳定性），并在本次运行填入 `observed` 与 `execution_cost`（以耗时为代理）。

### 数据集目录（`dataset/`）

所有 UnixBench 数据集 JSON 默认落在仓库根下 **`dataset/`** 内；每次实验使用一个 **会话子目录**（默认可读标签：`主机名_UTC时间戳`，可用 `--session` 自定义），避免混淆。

- **单轮**：默认写入 `dataset/<session>/run-01.json`；仍可用 `-o` 指定任意路径。
- **多轮**：`-n N`（如 **5 轮**）生成 `run-01.json` … `run-NN.json`，并在同目录写入 **`manifest.json`**（轮次、路径、时间戳列表）。默认 **每一轮都采集 xi**；若需省时间可 **`--reuse-xi`**（仅第 1 轮采集，后续轮复用）。

### 命令

```bash
cd /path/to/MoEBench

# 仅导出专家目录 → dataset/unixbench-expert-catalog.json（也可用 -o 指定路径）
python3 -m moebench.unixbench --catalog-only

# 单轮：dataset/<hostname>_<UTC>/run-01.json
python3 -m moebench.unixbench

# 单轮：自定义输出文件（与旧用法兼容）
python3 -m moebench.unixbench -o /tmp/once.json

# 五轮：dataset/<session>/run-01.json … run-05.json + manifest.json
python3 -m moebench.unixbench -n 5
# 若要以 sudo 权限采集每轮 xi
python3 -m moebench.unixbench --sudo -n 5

# 五轮 + 自定义会话文件夹名
python3 -m moebench.unixbench -n 5 --session my_i9_exp_20260322

# 五轮且复用第 1 轮的 xi（更快）
python3 -m moebench.unixbench -n 5 --reuse-xi

# 指定数据集根目录（默认仍为仓库下 dataset/）
python3 -m moebench.unixbench -n 5 --dataset-root /data/moebench_datasets

# 把额外参数交给 UnixBench 的 Run（须使用 -- 分隔）
python3 -m moebench.unixbench -n 1 -- -v -i 3
```

输出 JSON 含 `schema: moebench.unixbench.dataset.v1` 字段：`xi`、`yi`、`ti`、`experts`、`session`（`tag` / `round_index` / `xi_reused_from_previous_round`）、`unixbench.result_files`（报告 / `.log` / `.html` 路径）。多轮结果可在会话目录内批量做 **方差与相关** 分析。

## Phoronix Test Suite 实验流水线（xi / yi / ti）

需已安装 **Phoronix Test Suite**（可在仓库旁克隆 `phoronix-test-suite/`，或系统 PATH 中有 `phoronix-test-suite`）。首次无人值守批量测试前建议执行一次 **`phoronix-test-suite batch-setup`** 完成交互配置。

1. 采集 **xi**：`collect_all()`（`--no-features` 可跳过）。
2. 运行套件：默认 **`phoronix-test-suite batch-run <suite>`**（使用套件内默认选项，不逐项提问）；若需与命令行一致使用 **`phoronix-test-suite run cpu`**，请指定 **`--pts-mode run --suite cpu`**。
3. 通过环境变量 **`TEST_RESULTS_NAME`** 固定保存的结果文件名，便于后续导出。
4. 调用 **`phoronix-test-suite result-file-to-json <结果名>`** 得到结构化结果，写入 **`yi.pts_export`**；**`ti`** 由各 profile 结果缓冲中的 **`test_run_times`**（秒）汇总得到；**`experts`** 由导出结果中的各测试 profile 生成。

### 命令

```bash
cd /path/to/MoEBench

# 推荐：batch-run + cpu 套件，单轮数据 → dataset/<session>/run-01.json（另存同目录 run-01_pts_raw.json）
python3 -m moebench.phoronix --suite cpu --pts-mode batch-run

# 与「phoronix-test-suite run cpu」等价
python3 -m moebench.phoronix --suite cpu --pts-mode run

# 需要 root/完整 perf 等时（MoEBench 会把 dataset 会话目录 chown 给 SUDO_USER，便于 PTS 以该用户写导出 JSON）
python3 -m moebench.phoronix --suite cpu --pts-mode batch-run --sudo

# 指定本机 PTS 可执行文件路径
python3 -m moebench.phoronix --pts-bin /opt/phoronix-test-suite/phoronix-test-suite --suite cpu

# 多轮：dataset/<session>/run-01.json … run-NN.json + manifest.json（勿与 -o 同用）
python3 -m moebench.phoronix --suite cpu --pts-mode batch-run -n 5

# 多轮且仅第 1 轮采 xi，后续轮复用（更快）
python3 -m moebench.phoronix --suite cpu --pts-mode batch-run -n 5 --reuse-xi

# 自定义会话目录名
python3 -m moebench.phoronix --session my_pts_exp --suite cpu -n 3

# 快速单测（默认只跑 pts/ctx-clock，覆盖 --suite；可先 install 该 profile）
python3 -m moebench.phoronix --pts-smoke --pts-mode run
python3 -m moebench.phoronix --pts-smoke --pts-smoke-suite pts/smallpt --pts-mode run
```

输出 JSON 含 `schema: moebench.phoronix.dataset.v1`：`xi`、`yi`（含 `pts_export`）、`ti`、`experts`、`phoronix`（命令、结果文件名、`result-file-to-json` 原始导出路径）。多轮时另有 `schema: moebench.phoronix.batch_manifest.v1` 的 `manifest.json`。

### PTS 全套实验（与 UnixBench 流程对齐：专家 → GNN 路由 → XGBoost 重建 → 对比）

**数据**：仅使用 **PTS** 会话目录（本仓库示例为 `dataset/aces-*`，且每个会话至少 `run-01.json`）。**不要**把 UnixBench 会话混进来（例如 `aces-System-Product-Name_20260324T043804Z` 仅 UnixBench）；加载器默认会排除该目录名。训练时请用 **`aces-*/run-01.json`** 这类 glob，避免扫到无关 JSON。

**权限**：**只有采集系统特征（xi）需要 `sudo` 时**再使用 `python3 -m moebench.phoronix ... --sudo` 或 `python3 -m moebench ... --sudo`。下面的分析、训练、对比实验均在**普通用户**下执行即可（与 Phoronix 安装在用户目录一致，也避免权限问题）。

#### 1）专家建模（相关 / 冗余分析）

```bash
cd /home/cxc/MoEBench

python3 scripts/phoronix_expert_analyze.py \
  --dataset-root dataset \
  --glob-pattern 'aces-*/run-01.json'
```

生成 `phoronix_expert_model_global.json`、相关系数 CSV 等（输出在 `dataset/` 或 `--out-dir`）。

#### 2）路由模型（Expert GNN）

```bash
mkdir -p dataset/pts_router

python3 scripts/router_train.py \
  --benchmark phoronix \
  --dataset-root dataset \
  --glob-pattern 'aces-*/run-01.json' \
  --model-type gnn_expert \
  --model-out dataset/pts_router/router_gnn.pt \
  --gnn-emb-dim 16 \
  --mlp-epochs 200 \
  --auto-install
```

#### 3）结果重建模型（XGBoost，无补测 / 主动采样）

```bash
mkdir -p dataset/pts_models

python3 scripts/reconstruct_train_eval.py \
  --benchmark phoronix \
  --dataset-root dataset \
  --glob-pattern 'aces-*/run-01.json' \
  --model-type xgboost \
  --skip-cv \
  --no-uncertainty \
  --export-model dataset/pts_models/reconstruct_xgb.pkl \
  --train-aug 20 \
  --train-k-min 2 \
  --train-k-max 12 \
  --auto-install
```

（`--train-k-max` 勿大于当前 cpu 套件中 profile 个数；若报维度错误可调小。）

#### 4）完整实验：Top-K 子集 + 重建 vs 全量 `run cpu`

```bash
# 默认不启用 eBPF、不以 root 跑 PTS；若需与采集时相同权限采 xi，再加 --sudo-for-xi
python3 scripts/experiment_router_reconstruct_vs_full_pts.py \
  --router-model dataset/pts_router/router_gnn.pt \
  --reconstruct-model dataset/pts_models/reconstruct_xgb.pkl \
  --top-k 5 \
  --pts-mode run \
  --suite-full cpu \
  --dataset-root dataset
```

输出 JSON 含 **部分运行时间**、**全量 `cpu` 时间**、**suite 均值（各子项主 `value` 的算术平均）** 的预测误差等。全量 `run cpu` 耗时较长，请预留时间。

输出目录形如 `dataset/experiments/pts_pts_nvidia-gpu-compute_<会话>/`，与不同 PTS 套件区分。

### PTS `pts/nvidia-gpu-compute`（与 `run pts/nvidia-gpu-compute` / GPU 套件对齐）

全流程与上文相同，区别在于 **套件 id**、**默认数据目录名带套件标记**（未指定 `--session` 时为 `<主机>_pts_nvidia-gpu-compute_<UTC>/`）、以及下游脚本用 **`--pts-suite pts/nvidia-gpu-compute`** 只读取该套件的会话（可与 `cpu` 数据混放在 `dataset/` 下）。

**xi 中的 GPU 相关采集**：与套件无关，`collect_all()` 在 **`xi.static.gpu`** 中记录 **每块 NVIDIA 显卡**的驱动/vBIOS、显存总量、PCIe 代数与位宽、最大 GPU/SM/显存频率、功耗墙、持久化与 compute mode，以及 **`clinfo -l`** 得到的 **OpenCL 平台/设备数量**（原文在 `opencl.text`）；在 **`xi.dynamic.gpu`** 中再采一次 **瞬时**显存占用、GPU/显存利用率、功耗、温度、当前频率等。PTS 跑分结果里的「Graphics / OpenCL 字符串」仍只在导出的 **`yi.pts_export`**（Phodevi）里，与上述 **可机读数值特征**互补。路由/重建用的 `XiVectorizer` 已将上述字段展开为 **`gpu_*` / `opencl_*` 数值列**；旧 JSON 缺少 `gpu` 时对应列为 0。**更换 `XiVectorizer` 后须重新训练**路由与重建模型。

**1）数据采集**（等价于 `phoronix-test-suite run pts/nvidia-gpu-compute`，交互式可选用 `--pts-mode run`）

```bash
python3 -m moebench.phoronix --suite pts/nvidia-gpu-compute --pts-mode run
# 或 batch-run：
# python3 -m moebench.phoronix --suite pts/nvidia-gpu-compute --pts-mode batch-run
```

多轮请将 `--suite` 固定为同上；会话目录名已含 `pts_nvidia-gpu-compute`，便于与 CPU 采集区分。

**2）专家建模**

```bash
python3 scripts/phoronix_expert_analyze.py \
  --dataset-root dataset \
  --glob-pattern '*pts_nvidia-gpu-compute*/run-*.json' \
  --pts-suite pts/nvidia-gpu-compute \
  --out-dir dataset
```

生成 `dataset/phoronix_pts_nvidia-gpu-compute_expert_model_global.json` 及对应 CSV（文件名含套件标签）。

**3）路由（GNN）**

```bash
mkdir -p dataset/pts_nvidia_gpu_router

python3 scripts/router_train.py \
  --benchmark phoronix \
  --pts-suite pts/nvidia-gpu-compute \
  --dataset-root dataset \
  --glob-pattern '*pts_nvidia-gpu-compute*/run-*.json' \
  --model-type gnn_expert \
  --model-out dataset/pts_nvidia_gpu_router/router_gnn.pt \
  --auto-install
```

**4）重建模型（XGBoost 示例）**

```bash
mkdir -p dataset/pts_nvidia_gpu_models

python3 scripts/reconstruct_train_eval.py \
  --benchmark phoronix \
  --pts-suite pts/nvidia-gpu-compute \
  --dataset-root dataset \
  --glob-pattern '*pts_nvidia-gpu-compute*/run-*.json' \
  --model-type xgboost \
  --skip-cv \
  --no-uncertainty \
  --export-model dataset/pts_nvidia_gpu_models/reconstruct_xgb.pkl \
  --train-aug 20 \
  --train-k-min 2 \
  --train-k-max 12 \
  --auto-install
```

（`train-k-max` 勿大于当前套件 profile 个数。）

**5）完整对比实验**

```bash
python3 scripts/experiment_router_reconstruct_vs_full_pts.py \
  --router-model dataset/pts_nvidia_gpu_router/router_gnn.pt \
  --reconstruct-model dataset/pts_nvidia_gpu_models/reconstruct_xgb.pkl \
  --top-k 3 \
  --pts-mode run \
  --suite-full pts/nvidia-gpu-compute \
  --dataset-root dataset \
  --sudo
```

需已安装 **`pts/nvidia-gpu-compute`** 套件内测试与驱动/NVIDIA 依赖；仍以 **创建 PTS 数据集时的同一用户** 运行，勿 `sudo python3`（除非按前述文档仅为 xi 提权）。

## 关于 `perf` 与权限

若 `kernel.perf_event_paranoid` 较高（如 `4`），普通用户可能无法使用 `perf` 的 PMU 计数，动态特征会退化为 **`/proc` 等代理指标**，并在 `dynamic` 中给出 `perf_degraded` 说明。需要完整 PMU 时，请按系统策略调整该参数或使用具备 **`CAP_PERFMON`** 的方式运行采集进程（参见内核文档 [Perf events and tool security](https://www.kernel.org/doc/html/latest/admin-guide/perf-security.html)）。

## 许可证

各子目录可能自带许可证（如 UnixBench、PTS）；`moebench` 包请与项目整体约定保持一致。

## UnixBench 路由器建模（Router）

目标：输入 `xi`，输出每个 expert 的选择概率；根据概率选择 `Top-K` 个 expert 运行 UnixBench 子集，并把选择/执行结果写入 JSON。

### 训练

```bash
cd /home/cxc/MoEBench

# 方式 1：LightGBM Ranker（推荐）
python3 scripts/router_train.py \
  --dataset-root dataset \
  --glob-pattern '*/run-*.json' \
  --model-type lightgbm \
  --model-out dataset/unixbench_router/router_model.pkl \
  --auto-install

# 方式 2：小型 MLP
python3 scripts/router_train.py \
  --dataset-root dataset \
  --glob-pattern '*/run-*.json' \
  --model-type mlp \
  --model-out dataset/unixbench_router/router_model.pt \
  --auto-install

# 方式 3：Subset selection network（仅 xi → 各 expert 的 logits；按 relevance 分布做软交叉熵）
python3 scripts/router_train.py \
  --dataset-root dataset \
  --glob-pattern '*/run-*.json' \
  --model-type subset_sel \
  --model-out dataset/unixbench_router/router_subset.pt \
  --auto-install

# 方式 4：简单 Expert GNN（固定全连接邻接 + 2 层消息传递；无 torch_geometric 依赖）
python3 scripts/router_train.py \
  --dataset-root dataset \
  --glob-pattern '*/run-*.json' \
  --model-type gnn_expert \
  --model-out dataset/unixbench_router/router_gnn.pt \
  --gnn-emb-dim 12 \
  --auto-install
```

说明：

- **`lightgbm`**：`xi || expert_onehot`，`LGBMRanker(lambdarank)`。
- **`mlp`**：逐样本回归 relevance（与旧版一致）。
- **`subset_sel` / `gnn_expert`**：对每个系统把 12 个 expert 的 relevance 归一化为目标分布，对 `softmax(logits)` 做交叉熵式训练（列表项级别、无外部 GNN 库）。

### 一次性训练 + 完整实验对比（推荐）

在**同一台机器、同一重建模型**下，依次训练多种 Router 并各跑一遍「Router + Reconstruction vs Full」，汇总到 `ablation_summary.json`：

```bash
cd /home/cxc/MoEBench

# 先导出重建模型（若还没有）
python3 scripts/reconstruct_train_eval.py \
  --dataset-root dataset \
  --skip-cv \
  --export-model dataset/models/reconstruct_lgbm.pkl \
  --model-type lightgbm

# 训练 lightgbm / mlp / subset_sel / gnn_expert 并各跑完整实验（可加 sudo）
# 推荐：整行复制（避免续行符断行导致 `--auto-install` 被当成 shell 命令）
python3 scripts/run_router_model_ablation.py --dataset-root dataset --reconstruct-model dataset/models/reconstruct_lgbm.pkl --sudo --auto-install

# 多行时：每行末尾必须是 \ 且后面不能有空格；下一行要紧贴接上，不要单独一行只写 --auto-install
python3 scripts/run_router_model_ablation.py \
  --dataset-root dataset \
  --reconstruct-model dataset/models/reconstruct_lgbm.pkl \
  --sudo \
  --auto-install
```

常用参数：

- **`--skip-train`**：跳过训练，仅对已存在的 `dataset/router_models/<时间戳>/` 下检查点跑实验。
- **`--sudo`**：把 `--sudo` 传给完整实验脚本（采集 `xi` 需要 root 时用）。不要写裸的 `--experiment-extra --sudo`（后者会被 argparse 拆错）；若必须用 `--experiment-extra`，请写成 `--experiment-extra='--sudo'` 或 `--experiment-extra "--sudo"`。
- **`--models lightgbm,mlp`**：只跑子集。
- **`--models-dir path`**：指定模型输出目录（默认 `dataset/router_models/<UTC时间戳>/`）。
- **`--experiments-parent`**：实验输出放在 `<dataset-root>/<该子目录>/router_ablation_<UTC>/`（默认 `experiments`）。若该目录曾被 `sudo` 写成 root 所有导致无法创建子目录，脚本会自动改写到 `dataset/ablation_runs/router_ablation_<UTC>/`，或你可先执行 `sudo chown -R $USER:$USER dataset/experiments` 恢复权限。

产物：

- 各模型一份完整实验 JSON：`dataset/experiments/router_ablation_<UTC>/experiment_<model>.json`（若触发回退则为 `dataset/ablation_runs/router_ablation_<UTC>/…`）
- 汇总：同目录下的 `ablation_summary.json`（含各模型的 `suite_abs_err`、`partial_ub_s`、`full_ub_s` 等）

### 运行（推断 + 执行 Top-K 子测试）

```bash
cd /home/cxc/MoEBench
python3 scripts/router_run_unixbench.py \
  --model dataset/unixbench_router/router_model.pkl \
  --top-k 3
# 若需要 sudo 权限采集 xi
python3 scripts/router_run_unixbench.py --sudo \
  --model dataset/unixbench_router/router_model.pkl \
  --top-k 3
```

输出：会在 `dataset/unixbench_router/<session>/` 下生成 router-run 的 JSON，并且 UnixBench 的标准输出会直接显示在终端。

> 建议：在 miniconda 环境中运行，不要用 `sudo python ...`，否则会切到系统 Python 导致 `ModuleNotFoundError: lightgbm`。

## 结果重建（Reconstruction）

目标：输入 **系统特征 `xi` + 已执行子测试结果**（由 Router 选出的子项），预测 **完整 12 个子测试 Index + 完整 suite 总分**（`system_benchmarks_index_score`）。

### 交叉验证评估（MAE / RMSE / Spearman / Kendall）

```bash
cd /home/cxc/MoEBench
python3 scripts/reconstruct_train_eval.py \
  --dataset-root dataset \
  --model-type lightgbm \
  --folds 5 \
  --eval-partial-k 3 \
  --train-aug 10 \
  --report-json dataset/reconstruct_cv.json
```

输出 JSON 会包含：

- `oof_metrics.mae_suite_index` / `rmse_suite_index`
- `oof_metrics.spearman_suite` / `kendall_tau_suite`
- `time_savings.mean_fraction_wall_time_saved_vs_full_suite`（基于 `ti` 的模拟节省比例）

### 导出可推理的重建模型

```bash
cd /home/cxc/MoEBench
python3 scripts/reconstruct_train_eval.py \
  --dataset-root dataset \
  --skip-cv \
  --model-type lightgbm \
  --export-model dataset/models/reconstruct_lgbm.pkl
```

默认会导出 **v2**（带不确定性：树模型为各目标训练 `expected |残差|` 估计器，MLP 为均值+log-var 双头）。主动补跑实验需要 v2；若只要旧版 v1，请加 **`--no-uncertainty`**。

如果使用 MLP，可改为：

```bash
python3 scripts/reconstruct_train_eval.py \
  --dataset-root dataset \
  --skip-cv \
  --model-type mlp \
  --export-model dataset/models/reconstruct_mlp.pt
```

## 论文补充实验（离线 CV：UnixBench + PTS CPU + PTS GPU）

面向投稿实证：`scripts/paper_reconstruct_cv_extras.py` 在**已有数据集 JSON**上做离线交叉验证（**不**重新跑 UnixBench / PTS）。默认 **`--suites unixbench,phoronix_cpu,phoronix_gpu`**，一次输出三套件的汇总 JSON（schema **`moebench.paper_reconstruct_cv_extras.v2`**，顶层 `suite_results[]` 每项对应一套 benchmark）。

| 能力 | 说明 |
|------|------|
| **三套件** | **UnixBench**（`moebench.unixbench.dataset.v1`）；**PTS CPU**（`yi.suite == cpu`）；**PTS GPU**（`yi.suite == pts/nvidia-gpu-compute`）。各自可用独立 glob 收集 `run-*.json`。 |
| **评估子集策略** | `random`、`fixed_first_k`、`fixed_cpu_mix` / `fixed_io_mix`（UnixBench 优先 CPU/IO 子项；PTS 上无匹配 id 时退化为「canonical profile 顺序前缀」）、`greedy_slowest` / `greedy_fastest`（UB：`ti` parallel 32；PTS：`ti.by_test_id.time_s_total`）、`router`（见下） |
| **router** | 三套套件需 **分别** 训练路由：`--router-model-unixbench`、`--router-model-pts-cpu`、`--router-model-pts-gpu`。也可用 **`--router-model`**（仅等价于 UnixBench，兼容旧用法）。`policies` 含 `router` 时，**凡在本次 `--suites` 中选中的套件都必须提供对应 checkpoint**，否则会报错退出。 |
| **xi 消融** | `full`、`static_hw_only`、`no_perf_pmu`、`no_dynamic_proc`、`no_gpu`（三套共用同一向量化与消融逻辑） |
| **PTS suite 标量** | `--pts-suite-target logmean`（默认，与导出重建模型常用设定一致）或 `arithmetic_mean` |
| **CV 划分** | `leave_one_session_out`（按会话目录名留出）；仅有一条会话时会 **按套件** 回退 `random_fold` |
| **K 扫描** | `--k-sweep 1,2,3,4,5` |
| **额外指标** | `median_bucket_accuracy_suite`（suite 中位数分桶一致性） |

**一键跑三套件**（可通过环境变量改 glob / CV / 输出路径）：

```bash
cd /path/to/MoEBench
./scripts/run_paper_cv_three_suites.sh
# 或例如：
OUT=dataset/paper_cv_full.json CV_MODE=random_fold FOLDS=5 ./scripts/run_paper_cv_three_suites.sh
```

**等价 Python（明确三套 glob）**：

```bash
cd /path/to/MoEBench

python3 scripts/paper_reconstruct_cv_extras.py \
  --dataset-root dataset \
  --suites unixbench,phoronix_cpu,phoronix_gpu \
  --glob-unixbench '*/run-*.json' \
  --glob-pts-cpu 'aces-*/run-*.json' \
  --glob-pts-gpu '*pts_nvidia-gpu-compute*/run-*.json' \
  --cv-mode leave_one_session_out \
  --folds 5 \
  --policies random,fixed_first_k,fixed_cpu_mix,greedy_slowest,greedy_fastest \
  --xi-ablations full \
  --pts-suite-target logmean \
  --eval-partial-k 3 \
  --report-json dataset/paper_cv_three_suites.json
```

**仅跑其中一类**：

```bash
python3 scripts/paper_reconstruct_cv_extras.py --dataset-root dataset --suites unixbench \
  --glob-unixbench '*/run-*.json' --report-json dataset/paper_cv_ub_only.json

python3 scripts/paper_reconstruct_cv_extras.py --dataset-root dataset --suites phoronix_cpu \
  --glob-pts-cpu 'aces-*/run-*.json' --report-json dataset/paper_cv_pts_cpu_only.json

python3 scripts/paper_reconstruct_cv_extras.py --dataset-root dataset --suites phoronix_gpu \
  --glob-pts-gpu '*pts_nvidia-gpu-compute*/run-*.json' --report-json dataset/paper_cv_pts_gpu_only.json
```

**含 router 的三套件对比**（示例路径请换成你训练生成的文件）：

```bash
python3 scripts/paper_reconstruct_cv_extras.py \
  --dataset-root dataset \
  --suites unixbench,phoronix_cpu,phoronix_gpu \
  --policies random,router \
  --router-model-unixbench dataset/unixbench_router/router_gnn.pt \
  --router-model-pts-cpu dataset/pts_router/router_gnn.pt \
  --router-model-pts-gpu dataset/pts_nvidia_gpu_router/router_gnn.pt \
  --eval-partial-k 5 \
  --report-json dataset/paper_cv_three_suites_router.json
```

**Pareto（多 K）**：

```bash
python3 scripts/paper_reconstruct_cv_extras.py \
  --dataset-root dataset \
  --suites unixbench,phoronix_cpu,phoronix_gpu \
  --cv-mode random_fold \
  --folds 5 \
  --k-sweep 1,2,3,4,5,6 \
  --policies random,fixed_first_k \
  --report-json dataset/paper_cv_k_sweep_three_suites.json
```

说明：某一套件有效样本 **&lt; 2** 或 glob 无匹配时，该套件会记入 `suite_errors`，其余套件仍写入 `suite_results`。若三套件全部失败则退出码非 0。实现模块：`moebench/paper_eval/`（`subset_policies.py`、`xi_ablation.py`）。

## 完整实验：Router + Reconstruction vs Full

目标：一次实验同时得到：

1. Router 子集执行时间与重建预测总分；
2. 全量 UnixBench 执行时间与真实总分；
3. 两者时间差与总分误差。

```bash
cd /home/cxc/MoEBench
python3 scripts/experiment_router_reconstruct_vs_full.py \
  --router-model dataset/unixbench_router/router_model.pkl \
  --reconstruct-model dataset/models/reconstruct_lgbm.pkl \
  --top-k 3 \
  --sudo
```

输出文件：`dataset/experiments/<session>/experiment_router_reconstruct_vs_full.json`

### 三种重建模型对比 + GNN 路由 + 低置信度补跑子测试

使用 **GNN 路由**（需先训练 `router_gnn.pt`）与三种 **v2 重建模型**（LightGBM / XGBoost / MLP）。共享一次部分跑与一次全量跑得到真值；各重建模型在相同初始子集上按 **预测 σ 最大** 的未执行子测试依次补跑、重预测，统计额外时间与总分误差下降。

```bash
cd /home/cxc/MoEBench
# 可选：先训练 GNN 路由
python3 scripts/router_train.py --dataset-root dataset --model-type gnn_expert \
  --model-out dataset/unixbench_router/router_gnn.pt --auto-install

python3 scripts/experiment_reconstruct_active_compare.py \
  --dataset-root dataset \
  --router-model dataset/router_models/20260402T045114Z/router_gnn.pt \
  --train-reconstruct-models \
  --no-auto-install \
  --sudo
```

`--train-reconstruct-models` 会导出 `dataset/models/reconstruct_{lgbm,xgb}_v2.pkl` 与 `reconstruct_mlp_v2.pt`，**默认**对子进程开启 `pip install`（缺 xgboost 等时会自动装）；若环境离线或禁止联网，请加 **`--no-auto-install`** 并事先装好依赖。若已导出可去掉 `--train-reconstruct-models` 并用 `--reconstruct-lightgbm` 等指定路径。汇总 JSON：`dataset/experiments/reconstruct_active_<session>/reconstruct_active_three.json`，查看 `per_model.*` 的 `suite_absolute_error_initial` / `final`、`suite_error_reduction`、`extra_subtests_wall_seconds`。

### Top-K 扫描（GNN 路由 + XGBoost 重建）

固定一次 `xi` 和一次 full UnixBench（作为共享真值），对多个 `K` 分别跑 partial + reconstruct，输出每个 K 的误差与节省时间，并给出推荐 K。

```bash
cd /home/cxc/MoEBench
python3 scripts/experiment_topk_sweep.py \
  --dataset-root dataset \
  --router-model dataset/router_models/20260402T045114Z/router_gnn.pt \
  --reconstruct-model dataset/models/reconstruct_xgb_v2.pkl \
  --k-values 1,2,3,4,5,6 \
  --objective balanced \
  --sudo
```

输出：`dataset/experiments/topk_sweep_<session>/topk_sweep.json`，重点看 `per_k` 与 `recommended.k`。

重点字段说明：

- `timing_seconds.partial_unixbench`：Router 子集 benchmark 时间
- `timing_seconds.full_unixbench`：全量 benchmark 时间
- `scores.predicted_full_suite_benchmarks_index`：重建预测总分
- `scores.actual_full_suite_benchmarks_index`：全量真实总分
- `comparison.suite_absolute_error` / `suite_relative_error`：总分误差
- `comparison.benchmark_time_saved_seconds_vs_full`：相对全量节省的 benchmark 秒数

## 常见问题（FAQ）

### 1) `FileNotFoundError: /path/to/router_model.pkl`

这是占位路径，不是实际文件。请改成真实模型路径（通常是）：

```text
dataset/unixbench_router/router_model.pkl
```

可先确认文件是否存在：

```bash
ls dataset/unixbench_router/
```

### 2) 没有 `reconstruct_lgbm.pkl`

先执行“导出可推理的重建模型”命令，生成：

```text
dataset/models/reconstruct_lgbm.pkl
```

### 可选：安装 Python ML 依赖

如果你已使用 miniconda：

```bash
conda activate <your_env>
./scripts/install_ml_python_deps.sh --no-torch
```

```bash
cd /home/cxc/MoEBench
./scripts/install_ml_python_deps.sh
```
