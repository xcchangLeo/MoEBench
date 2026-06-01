# 实验机器对照表

文件夹名中的 **hostname slug**（主机代码）与硬件配置、论文 Host ID 的对应关系。
数值来自各 session 的 `run-01.json` → `xi.static`（`lscpu` / `MemTotal`）；可用脚本重新生成：

```bash
python3 scripts/build_machines_registry.py
```

机器可读副本：`dataset/machines_registry.json`。

## 总览

| 论文 Host | 配置标签 | hostname slug | vCPU | 标称内存 | 实测 MemTotal | GPU | 已采集套件 |
|-----------|----------|---------------|------|----------|---------------|-----|------------|
| **H1** | **32U128G** | `aces-System-Product-Name` | 32 | 128 GiB | ~125.6 GiB | 有 (RTX 3090 Ti) | UnixBench, PTS-CPU, PTS-GPU |
| **H2** | **2U8G** | `iZbp1glgt48i9a8d49embxZ` | 2 | 8 GiB | ~7.2 GiB | 无 | UnixBench, PTS-CPU |
| **H3** | **4U8G** | `iZbp15n87643uk1sqjrdvdZ` | 4 | 8 GiB | ~7.1 GiB | 无 | UnixBench, PTS-CPU |
| **H4** | **4U16G** | `iZbp16krl0yc7euw7sb6slZ` | 4 | 16 GiB | ~15.0 GiB | 无 | UnixBench, PTS-CPU |
| **H5** | **8U8G** | `iZbp1acaw5wdllhz47922rZ` | 8 | 8 GiB | ~7.1 GiB | 无 | UnixBench, PTS-CPU |

> 标称内存按云实例规格取整（8 / 16 / 128 GiB）；实测值略低于标称，属正常现象。

## 目录命名规则

| 类型 | 路径模式 | 示例 |
|------|----------|------|
| 原始采集 session | `dataset/<hostname>_<UTC>/` | `dataset/iZbp15n87643uk1sqjrdvdZ_20260525T050535Z/` |
| PTS session | `dataset/<hostname>_<suite>_<UTC>/` | `dataset/iZbp15n87643uk1sqjrdvdZ_cpu_20260526T174649Z/` |
| 模型产物 | `dataset/models/<hostname>/` | `dataset/models/iZbp16krl0yc7euw7sb6slZ/` |
| 实验产物 | `dataset/experiments/<hostname>/` | `dataset/experiments/aces-System-Product-Name/` |
| 带时间戳的 grid 实验 | `dataset/experiments/router_recon_grid_<suite>_<hostname>_<UTC>/` | `router_recon_grid_unixbench_iZbp15n87643uk1sqjrdvdZ_20260525T073914Z` |

从任意 session 或实验目录名解析 hostname：

- UnixBench：`iZbp15n87643uk1sqjrdvdZ_20260525T050535Z` → `iZbp15n87643uk1sqjrdvdZ`
- PTS-CPU：`iZbp15n87643uk1sqjrdvdZ_cpu_20260526T174649Z` → `iZbp15n87643uk1sqjrdvdZ`
- PTS-GPU：`aces-System-Product-Name_pts_nvidia-gpu-compute_20260522T041557Z` → `aces-System-Product-Name`

## 各机器 session 目录

### H1 — `aces-System-Product-Name` (32U128G)

- `aces-System-Product-Name_20260524T102558Z` — UnixBench
- `aces-System-Product-Name_cpu_20260520T033403Z` — PTS-CPU
- `aces-System-Product-Name_pts_nvidia-gpu-compute_20260522T041557Z` — PTS-GPU

### H2 — `iZbp1glgt48i9a8d49embxZ` (2U8G)

- `iZbp1glgt48i9a8d49embxZ_20260524T141144Z` — UnixBench
- `iZbp1glgt48i9a8d49embxZ_cpu_20260525T050416Z` — PTS-CPU

### H3 — `iZbp15n87643uk1sqjrdvdZ` (4U8G)

- `iZbp15n87643uk1sqjrdvdZ_20260525T050535Z` — UnixBench
- `iZbp15n87643uk1sqjrdvdZ_cpu_20260526T174649Z` — PTS-CPU

> 注：H3 已有采集数据与 grid 实验，但尚无 `dataset/models/iZbp15n87643uk1sqjrdvdZ/` 目录。

### H4 — `iZbp16krl0yc7euw7sb6slZ` (4U16G)

- `iZbp16krl0yc7euw7sb6slZ_20260524T103905Z` — UnixBench
- `iZbp16krl0yc7euw7sb6slZ_cpu_20260525T045459Z` — PTS-CPU

### H5 — `iZbp1acaw5wdllhz47922rZ` (8U8G)

- `iZbp1acaw5wdllhz47922rZ_20260524T103902Z` — UnixBench
- `iZbp1acaw5wdllhz47922rZ_cpu_20260525T045422Z` — PTS-CPU

## 脚本 / API 用法

```python
from moebench.dataset_machines import load_machines_registry, machine_config_label

reg = load_machines_registry()
print(reg["by_slug"]["iZbp15n87643uk1sqjrdvdZ"]["config_label"])  # 4U8G
print(machine_config_label("iZbp15n87643uk1sqjrdvdZ"))            # 4U8G
```

命令行训练 / 实验时指定 `--machine <hostname_slug>`，例如：

```bash
--machine iZbp16krl0yc7euw7sb6slZ   # H4 / 4U16G
```
