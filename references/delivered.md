# MatchForge · 交付记录（delivered.md）

> 作者：**晨星** ｜ 生成日期：2026-10-10 ｜ 系统：MatchForge（图匹配 / QAP）

## 一、选定域

**图匹配 Graph Matching —— 二次分配问题（QAP）求解。**
避开了已交付域，选此因其在网络去匿名化、跨时序对齐、生物网络比对中均为核心难题，且算法谱系清晰、可确定性复现、可基准化。

## 二、算法谱系（6 方法 + 旗舰）

| 方法 | 路线 | 关键文献 |
|------|------|----------|
| random | 随机置换（下界） | — |
| degree_profile | 度降序配对 | — |
| spectral | Umeyama 特征向量 + 匈牙利 | Umeyama 1988 |
| grampa | 特征间隔加权谱相似 | Fan 2019 |
| faq | Birkhoff 多面体 Frank-Wolfe | Vogelstein 2015 |
| matchfuse | 异构热启动 FAQ 多起点 + 跨候选择优 | 本文集成 |
| **seeded** | **锚点受约束 FAQ（种子图匹配）** | **Fishkind 2012 / Lyzinski 2014** |

## 三、关键科学决策

1. **纯 ER 图无法无种子恢复**：ER/SBM 节点统计可互换 → QAP 景观平坦，所有无种子方法在噪声下退化到随机下界（已实测验证）。
2. **改用稠密 ER（p=0.5）作为 DGP**：节点邻域可区分，QAP 景观唯一可恢复。
3. **旗舰定为种子图匹配**：给定少量已知对应（anchors，默认 10 个），受约束 Frank-Wolfe 把平坦景观扳直，自由节点被高效恢复。这是该问题的 SOTA，且较无种子基线增益巨大、结果诚实。

## 四、质量门禁（全部 PASS）

- **G1** 同构(η=0)恢复率：spectral/grampa/faq/matchfuse/seeded 均 = 100% ≥ 0.99 ✅
- **G2** 头条(η=0.3)：seeded=86.2% vs 最强基线 2.2% → **Δ=+84.0pp** ≥ +10pp ✅
- **G3** 确定性：同 seed 两次运行逐位一致（max|Δrecovery|=0）✅

## 五、工程交付物

- 零依赖（numpy+scipy），scipy 缺失时 LAP 离线贪心兜底。
- 全局确定性：BLAS 线程钉死 + `np.random.default_rng` 全链路。
- 错误码体系 E100–E500（重码/重名导入期炸）。
- 测试：pytest 全绿（确定性 / 不变量 / FAQ 单调 / 锚点 / 离线兜底 / CLI 冒烟 / 配置校验）。
- ruff check + format 双绿；pyproject 钉规则。
- CI：`.github/workflows/ci.yml`（ruff + pytest + benchmark 门禁）。
- `benchmark.json` 产出 + 二次运行确定性校验 + `requirements.lock.txt` 锁版。

## 六、发布

- 仓库：https://github.com/CJX0712/matchforge （作者署名：晨星）
- Tag：v0.1.0 ｜ Release：见 GitHub
- 本文档对应任务书 §11「交付回写」。
