# MatchForge · 世界级图匹配系统

> 作者：**晨星** ｜ 开源协议：MIT ｜ 语言：Python（零下载、离线可降级、确定性可复现）

MatchForge 是一套面向**二次分配问题（QAP, Graph Matching）**的图匹配求解系统。它把学术界最扎实的几条路线——谱匹配、GRAMPA、Frank-Wolfe 在 Birkhoff 多面体上的 FAQ，以及**种子图匹配（Seeded Graph Matching）**——统一在一个确定性、可审计、可基准化的流水线里。

核心结论：**稠密相关图对上，无种子方法在噪声下退化到接近随机下界；而仅用少量已知对应（anchors）的种子图匹配即可把恢复率从 ~2% 拉回 86%+。**

---

## 方法谱系

| 方法 | 类别 | 思路 | 角色 |
|------|------|------|------|
| `random` | 基线 | 随机置换 | 下界 |
| `degree_profile` | 启发式 | 按节点度降序配对 | 结构基线 |
| `spectral` | 谱 | Umeyama 特征向量对齐 + 匈牙利 | 同构/低噪主力 |
| `grampa` | 谱 | 特征间隔加权相似度（Fan 2019） | 谱增强 |
| `faq` | QAP | Frank-Wolfe on Birkhoff polytope（Vogelstein 2015） | 精确 QAP 求解 |
| `matchfuse` | 集成 | 异构热启动 FAQ 多起点 + 跨候选对齐得分择优 | 无种子旗舰 |
| `seeded` | **种子 QAP** | 锚点受约束 Frank-Wolfe（Fishkind 2012） | **旗舰** |

## 为什么需要种子图匹配

随机图（纯 ER / SBM）的节点在统计上**可互换**：大量置换都能给出几乎一样高的对齐得分，QAP 景观极度平坦，无种子方法必然陷在局部极大。这是文献共识，不是实现缺陷。
**种子图匹配**通过给定少量已知节点对应，把平坦景观"扳直"——锚点行恒为已知对应，仅对自由节点做降维匈牙利子问题，余下节点被 FAQ 高效、确定性地恢复。这正是真实场景（跨时序网络对齐、去匿名化、生物网络比对）的 SOTA 做法。

---

## 安装

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.lock.txt
# 或：pip install numpy scipy
```

仅依赖 `numpy` + `scipy`。`scipy` 缺失时，LAP 自动降级为合法贪心指派（离线兜底）。

## 使用

### Python API

```python
from core.seed import set_all
from core.config import Config
from data.generators import generate_correlated_pair
from matching.seeded import SeededMatcher

set_all(0)
gp = generate_correlated_pair(n=120, model="er", edge_prob=0.5, noise=0.3, seed=7, n_anchors=10)
res = SeededMatcher().match(gp)
print("recovery:", (res.perm == gp.true_perm).mean())  # ≈ 0.86
```

### 命令行（benchmark + 三道门禁 + 落盘）

```bash
python -m cli run --out examples/benchmark.json
# 用环境变量覆盖默认配置：
MATCHFORGE_N_NODES=60 MATCHFORGE_NOISE_LEVELS=0.0,0.2,0.3 python -m cli run --env
```

### 端到端演示

```bash
python -m examples.run_demo      # 打印恢复率表 + G1/G2/G3 门禁 + 二次运行确定性校验
```

---

## 基准结果（默认配置：n=120, p=0.5, 稠密 ER, 5 seed）

恢复率 = 正确匹配节点比例（均值 ± 标准差，单位 %）。

| 方法 | η=0.0 | η=0.1 | η=0.2 | η=0.3 | η=0.4 |
|------|------:|------:|------:|------:|------:|
| random | 0.3 | 0.7 | 0.7 | 0.7 | 0.7 |
| degree_profile | 22.2 | 3.5 | 2.3 | 2.2 | 1.3 |
| spectral | 100.0 | 1.2 | 0.8 | 1.3 | 0.8 |
| faq | 100.0 | 2.7 | 3.3 | 1.2 | 1.2 |
| grampa | 100.0 | 1.2 | 1.7 | 0.7 | 1.0 |
| matchfuse | 100.0 | 2.5 | 2.5 | 1.3 | 0.7 |
| **seeded** | **100.0** | **100.0** | **100.0** | **86.2** | **33.7** |

### 三道质量门禁（全部 PASS）

- **G1 同构恢复率**：η=0 下 spectral/grampa/faq/matchfuse/seeded 均 = 100% ≥ 0.99 ✅
- **G2 头条增益**：η=0.3 时 `seeded=86.2%` vs 最强基线 `degree_profile=2.2%` → **Δ=+84.0pp** ≥ +10pp ✅
- **G3 确定性**：同 seed 两次运行逐位一致（max|Δrecovery|=0）✅

---

## 数学不变量（已实现并测试）

- 对齐得分与失真对偶： `‖A − PBPᵀ‖² = ‖A‖² + ‖B‖² − 2·s(P)`
- FAQ 目标序列 `s_k` **单调非减**（闭式线搜索保证；违反即抛 `E201`）
- 种子 FAQ 锚点行恒为唯一 1（数值投影兜底，无漂移）
- 全局确定性：BLAS 线程钉死 + `np.random.default_rng(seed)` 全链路受控

## 工程特性

- **零下载 / 离线兜底**：scipy 缺失时 LAP 退化为贪心合法置换。
- **确定性可复现**：`core/seed.set_all` 在 import numpy 前钉死 `OMP/MKL/OPENBLAS/..._NUM_THREADS=1`。
- **错误码体系 E100–E500**：重码/重名即在导入期 `RuntimeError`（见 `core/errors.py`）。
- **可审计流水线**：`pipeline.MatchPipeline` 统一调度、跨 seed/噪声聚合。

## 引用

- Vogelstein et al. 2015, *Fast Approximate Quadratic Programming for Graph Matching* (FAQ).
- Fishkind et al. 2012, *Seeded Graph Matching*.
- Lyzinski et al. 2014, *Seeded Graph Matching for Correlated Erdős–Rényi Graphs*.
- Umeyama 1988, *An Eigendecomposition Approach to Weighted Graph Matching*.
- Fan et al. 2019, *GRAMPA: Scalable Graph Matching*.
