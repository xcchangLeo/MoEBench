# BenchScout WWW 2026 修改计划（Round 0）

## 审查范围与证据边界

- 检查了 `BenchScout_WWW2026_submission.zip`：压缩包只有 `main.tex`、`main.pdf`、BibTeX/ACM 模板和 8 个 PDF 图，**不包含代码、数据、训练模型、运行脚本或复现实验说明**。因此本计划把“论文源文件修订”和“仓库实验/代码修订”分开；不能把仓库中已有结果直接假定为压缩包内结果。
- 检查了 `审稿意见.md`：两位审稿人总体为 weak accept，但共同指出评估覆盖窄、基线不足、方法新颖性不清楚、训练/测试划分和同主机泄漏风险需要解释，并希望看到更多平台、系统变化和噪声条件。
- 检查了 `dataset/`、`dataset/paper_reviewer_baselines/`、`dataset/paper_supplementary/` 及 `scripts/audit_paper_protocol.py`。当前 raw 数据有 5 个 host 身份；每个 host-suite 只有一个 session、5 个 run。仓库审计结论是：没有 headline 结果对应的机器可读 train/test manifest，无法证明 session-level holdout；现有 reviewer baseline 也明确写着 `run-level five-fold CV`、`No row is a session-level temporal holdout`。
- 不能凭空补写新结果。以下标为“需真实重跑”的项目在取得实际运行数据前，只能改协议、代码、图表占位或限制结论，不能填入预计数值。

## P0：投稿前必须修复的可信度问题

### P0.1 统一真实数据覆盖与主张

**发现**：正文 `main.tex` 同时写了“five hosts”和“six heterogeneous Linux hosts”，实验表列出 M6/T4，但 raw dataset 没有 M6 的 run 记录；正文对 M6 只给了 full-suite reference，未给 BenchScout 方法结果。摘要、贡献和结论也存在 `15/18 seconds`、`0.067--1.06%`、均值约 `0.2%` 等不同口径。

**修改**：

1. 先由实验代理确认 M6 是否有可核验的原始 JSON、采集 manifest、完整套件墙钟和模型产物。
2. 若没有，删除 M6/T4 行和“six hosts”表述，明确当前论文是 M1--M5 CPU（PTS-GPU 仅 M1），并从真实 JSON 重新聚合每个表的均值、P95、最大误差、时间与节省比例。
3. 若确实要保留 M6，必须补齐 M6 的 full run、训练/测试划分、方法预测、GPU 驱动/版本和运行日志；这些是**需真实重跑/补采数据**，不能从表格或图片反推。
4. 建立单一结果清单（suite、host、run/session、split、模型 checkpoint、脚本 commit、聚合规则），由论文表格自动生成，避免摘要、正文和图表继续引用不同实验批次。

**验收**：论文每个数字都能回链到 `dataset/` 中的 JSON/manifest 和生成脚本；不存在没有数据支撑的 host、误差、时长或节省比例。

### P0.2 解决数据泄漏与划分不可审计问题

**发现**：每个 host-suite 只有一个五轮 session；`dataset/paper_reviewer_baselines/reviewer_baseline_summary.md` 明确说明是 run-level five-fold CV 且没有 session-level temporal holdout。正文却暗示“held-out test runs”及独立训练/测试配置，且没有随论文提供 train/test manifest。训练、模型选择、Top-K 选择和最终报告是否使用同一 run 集合无法从现有产物证明。

**修改**：

1. 在代码中加入严格、可序列化的 split manifest：按 session 留出；若只能使用单 session，至少预注册 run-level split，训练、调参、最终测试分别固定，禁止用测试误差挑选 router/reconstructor/K。
2. 修改所有评估脚本，输出每个 fold 的 train IDs、validation IDs、test IDs、模型选择指标和 checkpoint 路径；在最终 JSON 中写入 dataset hash、代码 commit、随机种子。
3. 重新生成 headline hybrid、router-only、probe-only、execution-free、Top-K、路由策略和 xi 消融结果。该项是**需真实重跑**；若暂时没有第二个独立 session，应在正文明确改称“within-session run-level CV”，报告这一限制，不能称 temporal/session holdout。
4. 增加独立 session 或按时间采集的 held-out run（至少每个关键 host-suite 一个训练 session + 一个测试 session）；这是**需真实采集/重跑**，并优先于增加模型数量。

**验收**：审计脚本能从 manifest 复现论文每一行的样本数和划分；测试标签不参与 K、策略、模型或超参数选择；正文准确区分 run-level CV、session holdout 和 cross-host transfer。

### P0.3 修正时间与“在线测量”表述

**发现**：主实验使用历史 full-run score 和历史墙钟做 offline replay；部分时长是 `T_x + K*T_probe` 的估计。正文部分段落把此结果叙述为在线执行或给出固定 `15/18 s`，而 reviewer baseline 中部分 baseline 时间也是估计值。

**修改**：

- 每张表拆出 `offline replay error`、`estimated cost` 和真正在线 probe/trace 的 measured cost；只在有日志的行写 measured。
- 明确 full-suite 是历史参考，不把离线取出的部分分数称为在线观测；router-only 和 hybrid 的输入差异必须在 caption 和正文中重复说明。
- 对并行 probe 的墙钟定义给出公式、串行/并行设置和调度开销；若未实际测量并行端到端时长，删除“fixed wall-clock”或“negligible overhead”的绝对表达。

## P1：直接回应 WWW 审稿意见的核心实验

### P1.1 强基线与公平比较

仓库已有 `dataset/m1_unixbench_alternative_baselines*`、`paper_reviewer_baselines`，可作为起点，但当前 reviewer 汇总显示策略结果多为 run-level CV，且 Phoronix baseline 与正文 suite 命名不一致。需要统一 suite、K、split 和时间预算后再纳入正文。

优先加入并在同一严格 test split 上比较：

1. 固定前 K、固定 CPU/IO 混合、随机 K（多种 seed，并报告均值/方差/P95）；
2. 相关性选择、最快/最慢 K、简单 correlation-based reconstruction；
3. matrix completion；
4. adaptive probing（若预算、观测和实现可与 BenchScout 公平匹配）；
5. execution-free Wang/Tousi 仅作为静态预测 baseline，明确其训练域与 suite 不匹配限制。

已有 JSON 可以先重新审计和聚合；但只要 split、K 或训练协议改变，就属于**需真实重跑**，不能直接沿用旧数字。主表应至少报告 mean、std、median、P95/max、wall-time budget，并显示相对提升而非只显示最优一行。

### P1.2 泛化与系统变化

审稿人要求更多平台、kernel/软件变化、噪声和 contention。当前五台主机且每 host-suite 单 session 无法支持这些结论。

建议按资源可行性分层执行（全部为**需真实采集/重跑**）：

- 最低要求：同一 host 的独立时间 session，改变 sysctl/频率 governor/内核负载，保留未见配置测试集；
- contention：后台 CPU、内存带宽、磁盘 I/O、网络或 co-located workload，报告误差和失败/超时率；
- 软件/内核变化：至少两个 kernel 或 benchmark/compiler 版本；
- 跨机器：保留 per-host calibration 作为主要模式，再增加 zero-shot/cross-host transfer 结果，不能把 per-host 模型结果包装成跨机泛化。

如果无法完成这些采集，正文应主动收窄贡献到“host-local calibrated acceleration”，加入 limitations 段，而不是声称 production/general deployment robustness。

### P1.3 误差分布、失败模式与不确定性

当前摘要只报均值，无法回应低误差是否由少量 run 或 suite 聚合掩盖。修改评估脚本和图表，报告每个 host-suite-run 的绝对/相对误差、P50/P95/max、timeout、缺失 PMU/eBPF、probe 失败、异常值和预测区间（若实现可信）。对最坏 run 给出 trace，说明 fallback 是否真的触发。

eBPF on/off 和 probe duration 表只有 M1 UnixBench；可以作为案例，但不能外推到所有 host/suite。扩大覆盖需真实重跑；否则把标题和结论改成“controlled M1 case study”。

## P1：论文清晰度与新颖性修改（无需新实验即可先做）

1. 在 Introduction 末尾用一个表或段落精确区分已有工作中的 subset selection、prefix/probe prediction、matrix completion 与本文的组合边界；每个差异都要有引用，不用“pieces are familiar”式笼统断言。
2. 用一张信息流图明确训练阶段和推理阶段，标出哪些信号来自 target host、哪些标签来自历史 full run、哪些值是 probe prediction、哪些是 reconstructor output。
3. 在 Methods 中给出 router 训练目标、reconstructor mask 生成、probe label、PTS log transform、官方 aggregate 的完整定义；说明 K 和模型类型如何在训练/验证集上选择。
4. 删除正文中重复的 `Overall Effectiveness` 段落、过时注释和相互冲突的五台/六台主机叙述；把“formal benchmarking”限定为加速估计，不暗示可替代 certified submission。
5. 加入明确 limitations：host-local calibration 成本、单 session 数据限制、benchmark/version 范围、GPU 仅 M1、contention/kernel/fault 未覆盖的边界、模型失败时是否回退 full run。
6. Related Work 补充 adaptive benchmark selection、early termination/prefix measurement、performance regression under workload interference、matrix completion/active learning 等直接相邻工作，并逐项说明本文差异。

## P2：代码、数据和复现包

压缩包不能让审稿人运行方法。若 WWW artifact 允许，至少提供：

- `scripts/` 中的采集、训练、评估、表格生成入口和固定版本依赖；
- 脱敏后的最小 raw JSON、manifest、模型 checkpoint 或可重训配置；
- 一键离线 replay，能从发布数据重建正文表格和主要图；
- `README` 说明硬件、内核、GPU/PTS/UnixBench 版本、权限需求、eBPF/perf fallback、预计时长和无法复现在线测量时的替代步骤；
- `PROVENANCE.md` 记录每个数字的来源文件、命令、commit、seed、split 和聚合脚本；
- 论文 zip 至少附代码仓库 URL/DOI 或 artifact appendix（取决于 WWW 匿名规则，不能泄露作者身份）。

这些是可直接改包结构的工作；但只要声称“可复现实验”，对应数据、checkpoint 和运行命令必须真实存在。

## P3：推荐执行顺序与独立审稿门槛

1. 先冻结原始 zip 和当前仓库状态，复制出带 round/commit 的工作目录；每轮修改保留 diff 和生成物清单。
2. 完成 P0.1--P0.3 的数据审计、split manifest 和数字重聚合。
3. 在独立审稿人报告后，只修复其指出且有证据的项目；修改代理不得自行把“修改后”视为提分。
4. 先跑严格 split 和已有 baseline，再决定是否投入 contention/kernel/跨机采集；任何未跑项目在论文中标记为 limitation。
5. 由独立审稿代理重新检查：问题重要性、方法新颖性、协议可审计性、baseline 公平性、统计充分性、复现性、表格数字一致性。若仍存在不可验证 claim，继续收窄 claim 或标记需实验，不得通过文字包装提升评分。

## 当前回合结论

当前材料最多支持“在五台 Linux 主机的 host-local、单 session 历史数据上，离线评估一个 bounded-probe + sparse-reconstruction 原型”的表述。它不支持 session-level 泛化、M6/T4 方法结果、production contention 鲁棒性或已可复现的公开 artifact 声明。下一轮修改应先解决数据和协议可信度，再考虑增加模型/图表和文字润色。

