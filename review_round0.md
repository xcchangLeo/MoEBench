# BenchScout WWW 2026 初审报告（Round 0）

**审稿立场：** 独立审阅稿件 ZIP 中的 `main.tex` / `main.pdf`，并只读检查仓库代码、数据、实验报告与审计脚本。此报告依据当前可见证据；未因已有实验或拟议修改而预先加分。源文件行号指 ZIP 解压后的 `main.tex`。

## 总体评价

论文提出 BenchScout：用主机上下文选择少量 benchmark 组件，做限时探测，再重建完整套件分数。问题有实际意义，端到端测量管线也可能有用；但当前稿件的核心准确性结论依赖极少、同 session 的数据，缺乏可核实的训练/测试隔离和模型选择过程。数据审计只发现 5 个 host identities；稿件提到的 M6/T4 没有对应原始 host-suite 数据。当前证据不能支撑“跨机器、可推广的高精度”结论，也不能独立复现摘要和主表结果。对 WWW Research Track，我建议 **Reject**，不是小修后可接收。

## 阻断性问题（P0）

### P0. 评估隔离不足，无法支持泛化与精度主张

稿件称每个 host-suite 的 held-out 测试配置独立于训练配置抽样（`main.tex` 第 637–641 行、附录第 1065–1068 行），并在正文多个位置报告交叉验证误差。仓库的 `scripts/audit_paper_protocol.py` 对原始 run JSON 的只读审计得到 **55 个 run、5 个 host identities、11 个 host-suite pair；每个 pair 只有 1 个 raw session、5 个 runs**，且明确记录没有 headline 结果的机器可读 train/test manifest，无法证明独立 session holdout。已有 supplementary 结果也记录 `cv_mode_effective=random_fold`、单 session、`n_samples=5`；每折只有一个验证样本。相邻轮次来自同一采集 session，且稿件所称“独立抽样配置”没有随结果保存的 split manifest 可核验。

这使误差估计对单个样本、采集时段和配置选择高度敏感。尤其稿件在多个 host/suite 和模型组合上挑选最优结果，而没有展示嵌套模型选择或锁定测试集；报告的均值和小数点精度不能替代独立评估。

**可验证修复：** 提交可审计的每个数据样本清单（host、suite、session、配置、时间戳、训练/验证/测试成员）、数据生成与 split 脚本及原始 OOF 预测；使用独立采集 session 或新配置/时间段做锁定测试集，在该测试集上不再挑 router、reconstructor、probe、K、特征或 baseline。若无法补做实验，应收窄为同一 host/session 内的探索性结果，撤掉跨 session/生产稳健性及“泛化”表述，并用真实 OOF 汇总重算所有结果。

### P0. 主实验数值缺少可复核的数据血缘，且 PTS 聚合语义被称为官方

摘要声称五台异构 host、跨 UnixBench 与 PTS 的平均 aggregate-score relative error 约 0.2%（`main.tex` 第 102–107、309 行）；主表报告多组精确误差（第 738–795 行）。仓库里确有大量 experiment JSON，但关键产物引用原作者机器的绝对路径（例如 `dataset/experiments/.../grid_summary.json` 中 `/home/cxc/MoEBench/...`），且主稿 ZIP 不包含数据、代码或结果 manifest。复核脚本 `scripts/paper_aggregate_tables.py` 在当前工作区因这些记录的绝对路径权限/位置不可访问而无法重建主表。故不能确认主表表格、摘要平均值与最终选择配置之间的关系，也不能从稿件包单独追溯 headline 结果。

另一个定义问题：方法第 547–549 行将 PTS profiles 的 primary throughput/latency 值用作者定义的 `log-mean` 聚合，同时把该目标与“official label/official suite semantics”并列陈述。稿件没有给出 PTS-CPU 或 PTS-GPU 的官方套件总分定义，且这些异量纲 profiles 的 log-mean 是作者构造的目标，不能未经论证称为官方分数。UnixBench 的官方 index score 与作者构造的 PTS 汇总目标应分开报告。

**可验证修复：** 附完整、相对路径的可复现实验包（固定代码版本、环境/依赖、原始数据或可公开取得的指针、配置、所有 fold 预测、aggregation 代码、表格生成脚本），从干净目录自动产生与稿件逐项一致的表格。明确区分 UnixBench 官方 System Benchmarks Index Score、PTS 原始 profile 指标及本文定义的 log-mean proxy；给出聚合公式、单位/方向变换、profile 纳入规则和敏感性分析，停止将 proxy 称为 PTS 官方 suite score。

### P0. M6/T4 与数据覆盖不一致，GPU 结果证据不完整

实验设置写六台 host，列出 M6 为 16-vCPU/64-GiB、NVIDIA T4，并称其提供 PTS-GPU full-suite reference（`main.tex` 第 583–605、617–621 行）。但仓库的 `dataset/MACHINES.md` 和 `scripts/audit_paper_protocol.py` 只记录 H1–H5 / 5 个 host identities；原始数据 audit 的 host-suite 清单没有 M6/T4。GPU 主表则对 M6 填入 12,951 秒 full-suite 时长，却没有 M6 预测结果（第 785–793 行）。稿件无法说明该时长的来源、环境、原始 PTS 记录或为何 M6 不能进入误差评估。文中“六台机器”因而不能由随附实验材料独立核验。

**可验证修复：** 若保留 M6 描述，提交其带时间戳的 PTS 原始记录、manifest、机器信息及运行命令，并解释其结果只用于参考的预注册原因；否则删除 M6 和该行时长，将主张限定为审计能证明的五台机器。不要把只有 full-run 参考值的机器计入“异构机器评估”。

## 主要问题（P1）

1. **方法贡献和新颖性定位不清。** 主要模块是树/MLP router、固定时间的 probe 回归器、随机 mask 的多输出重建器（`main.tex` 第 485–558 行），但目标函数、训练样本粒度、负采样/排序对、网络结构/超参数、早停与 calibration 都未充分定义。现稿较像已知回归器的系统组合；“Mixture-of-Experts”称呼没有展示通常意义上的专家训练/门控目标，也没有相对最接近的 benchmark subset selection、adaptive benchmarking 或 runtime prediction 方法进行强对比。需要具体界定算法新意，提供伪代码和复杂度，并对固定子集、相关性贪心、随机/最快/最慢等可比策略报告统一的误差—成本 frontier。

2. **主结果与 Top-K/模型选择实验协议不一致，且 baseline 对比不够公平。** 主结果宣称每次约 15 秒（3 秒 profile + K=3 每个 4 秒，`main.tex` 第 397–408 行），但 supplementary Top-K M1 文件记录 random-fold、`n_samples=5`，三个 suite 的误差远高于表格所暗示的数字（例如 Top-K 报告的 UnixBench K=3 mean relative error 为约 0.231%，PTS-CPU 约 1.700%，PTS-GPU 约 0.895%；见 `dataset/paper_supplementary/topk_*_aces-System-Product-Name.json`）。这是不同实验路径/不同指标，稿件应明确标出不可直接比较，解释从这些结果选到 K=3/2/3 后如何锁定 headline 配置。执行自由 baselines 只使用 static hardware features（第 623–627、742 行），其精度差不能证明“执行锚定”优于合理的动态上下文模型或简单历史基线；应增加更强且同 split、同目标、同调参预算基线。

3. **统计报告与风险刻画不足。** 关键主表只给单点误差和耗时，没有置信区间、跨随机种子、p95/max 在线误差或预测失败率。相比误差均值，异常配置/负载上的误差更决定该方法能否用于调优或回归检测。每个 test fold 单样本不能支持总体稳定性结论。

4. **稳健性结论超出证据。** 缺失 eBPF 实验仅在一台主机、UnixBench probe-only 条件进行（`main.tex` 第 977–1007 行）；probe duration sweep 同样是单主机单 suite、all-component probe（第 1010–1032 行）。这不能证明完整 router+probe+reconstructor 在 PMU/eBPF 不可用、真实邻居 contention、超时/失败探测下稳定。请对端到端系统注入失败与负载，报告可用率及误差尾部；否则缩窄“生产遥测稳健”表述。

5. **局限性和适用边界不足。** 论文讨论审计型 benchmark 的限制只存在于注释掉的文字（ZIP `main.tex` 第 284–300 行为注释），正文结论没有讨论每主机先验训练成本、需要重复历史 full runs、冷启动、测试集分布外变化、跑探针对系统造成的扰动、不可认证替代等重要限制。应明确把工具定位为 calibrated preview / tuning aid，不宣称替代标准 run-to-completion。

## 次要问题（P2）

- related work 仅约一页，且把预测类工作概括为“不在 target host 上执行”，较为绝对；应覆盖并讨论选择性/自适应 benchmark、early stopping、workload characterization、在线校准与测量噪声工作，避免人为设置过弱的对照。
- `main.tex` 第 269 行用 BYTE Unix benchmark 的引用支撑“固定 CPU 子集无法适用于 I/O/memory host”的论述，证据针对性不足；需直接引用已有子集/benchmark selection 研究。
- 说明关键图的样本量、误差条和采集条件。相关性图只含 pooled n=25，未展示不确定性；相关性并不自动证明在不同状态下存在条件互补的 Top-K 信息增益。
- 检查 PDF 中断行、图表可读性与术语一致性（“suite score/index/profile score”“real probe/micro-probe”“official”用法）。

## 投稿格式核查

- 提供的 PDF 为 12 页 Letter，匿名作者显示为 `Anonymous Author(s)`；正文采用 `acmart` 的 `sigconf, review, anonymous`，没有可见作者身份泄露。不能仅凭 PDF 页数断言违反 WWW 页数政策，需按当年官方 CFP 对正文/参考文献是否计页核对。
- 源文件仍带 ACM DOI、ISBN 占位符和 sample template 版权信息（`main.tex` 第 53–77 行），但通过宏覆盖隐藏 review 版权脚注，当前 PDF 文本未见这些占位符。投稿源建议清理误导性占位元数据，并用官方 WWW 2026 模板/选项重新编译确认。
- `\bibliographystyle{acm}` 与包内 `ACM-Reference-Format.bst` 看起来合理；引用/标签静态检查没有发现未定义 `\ref`，但因 ZIP 没有 `.log`/构建脚本，无法独立验证 BibTeX 警告或最终投稿系统格式。

## 评分（初审）

- **Overall recommendation：Reject（2/5）**
- **Confidence：4/5**（评估协议与覆盖结论基于稿件及仓库 audit；无法访问/复核作者机器上所有实验路径）
- **Novelty：2/5**
- **Technical quality：2/5**
- **Experimental quality：1/5**
- **Clarity：3/5**
- **Reproducibility：1/5**

**决定理由：** 目前最核心的两项主张——稀疏测量在不同运行状态下仍能准确恢复同一 benchmark 指数，以及跨多台异构主机节省运行成本——没有足够独立测试证据。高精度数字在只有每 host-suite 五个同 session 样本时特别脆弱；M6 也缺少数据血缘。即便文字和格式全部改善，这些实证缺口仍然存在，不能靠编辑性修改跨过 WWW 接收门槛。

## 未核验事项

- 没有独立重跑全部模型训练和表格生成；当前环境无法读取部分 experiment JSON 所引用的 `/home/cxc/MoEBench` 绝对路径。
- 未核验 WWW 2026 最新 CFP 的精确页数、匿名和 artifact 政策条文。
- 未对方法代码逐函数审计训练数据泄漏；因此将结果描述为“无法证明隔离”，不直接指控实际泄漏。
