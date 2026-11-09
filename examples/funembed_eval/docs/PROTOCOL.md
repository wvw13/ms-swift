# 评测协议与人工审核说明

## 固定输入

一个 JSONL 记录对应一次病例检查，不等同于一个患者。必填：

| 字段 | 含义 |
|---|---|
| case_id | 原清单病例 ID，唯一，关联向量 |
| tie_key | 原病例相对路径，唯一，用于分数并列排序 |
| patient_id / center | 原患者分组键与中心；患者排除键为 center + patient_id |
| split | internal_train / val / internal_test / external_test，保持原划分 |
| images.cfp | 一张 CFP 相对路径或 null |
| images.oct | 按 slice 7–12 升序排列的 `{slice,path}` 列表 |
| labels.cfp/oct/combined | 三组 `{ids:[有序类别ID],state:...}` |
| image_md5（可选） | CFP 加所有可用中心 OCT 的逐文件 MD5，顺序与 images 一致 |
| image_stats（随私有包提供） | 对应图像 `[size,mtime_ns]`，检查 hash 依据是否过期 |
| input_ids（私有包附加） | 保留原 C/O/B 中间输入 ID，便于追溯 |

`classes.json` 按 ID 0–21 排列，含标准 label、英文 text、已有中文名称等；ID 21 是明确无异常。`texts.json` 顺序一致，含 id/name/definition.C/O/B。类别映射和正式定义由私有包提供。B 定义沿用 `CFP features: {C} OCT features: {O}`，不生成新描述。

C 使用 CFP；O 使用中心六张 007–012；B 使用 CFP、007、008、009、010、011、012。缺图不补图、不复制切片；对应输入不可用，其他完整输入可保留。缺少实际文件会在检查阶段报错。空/未知/冲突标签仍可编码，但对应指标排除；normal 必须是 `[21]`，不能与疾病混用。山东保留外部划分。

正式清单计数：train 3,785（C），O/B 3,784；val456、internal_test1,038、external_test984。患者数依次2,717、321、644、574。modal 标签有效 train C/O/B=3617/3509/3784；val=441/419/456；内部=1009/976/1038；外部=945/863/984。

## 编码协议

图像与疾病候选分别编码；所有分支使用同一模型与 adapter。I0 是 `Represent the user's input.`；I1 原文在 `prompts.py`，影像指令说明模态和检索关注点，文本指令说明 name 或 visual-feature description。system 放指令，user 放图像占位符或候选文本。真实标签只进入计分，不进入影像消息。

原生 swift `qwen3_vl_emb` 提取最终层最后有效**输入** token，经 attention mask 排除 padding，再 L2。每图 token 上限1024；实际尺寸由 processor 按比例调整，不保证每张恰为1024×1024。记录实际 prompt token 数，不允许静默截断。

联合编码将 O/B 的六/七张放入一次请求。逐图编码各图先归一化，记 N 为 L2：C=u_C，O=N(mean(u_007…u_012))，B=N(0.5 C+0.5 O)。历史 legacy 的批处理顺序与处理器单独保留；不能把两种编码缓存混用。

编码输出 FP32 `[N,d]`，检查行数/ID、有限值、非零单位范数；跨图像与文本维度必须相同。完整/LoRA模型与 processor、模板、表征协议见 [MODEL_INPUT](MODEL_INPUT.md)。

## Query、Gallery 与标签

Query 来自所选 val/internal_test/external_test。病例 Gallery 始终为内部训练集，外部测试也用同一内部病例库；各模态按图像可用性及该标签协议是否有效筛选。固定同一清单比较模型。文本 Gallery 始终22类，无论某类在某个模态训练标签中出现与否；13类 OCT 仅为另行定义的诊断。

当前病例方向：C→C、O→O、B→B、O→C、C→O。历史额外 B→C、B→O 可显式选择。每个 Gallery 项是一例，不是单张切片。

- **modal**：C依据 CFP 标签，O依据 OCT 标签，B依据综合标签；跨模态两端各取自己输入的标签。
- **combined**：所有输入都依据综合标签。

例如某病例 CFP={a}、OCT={b}、综合={a,b}，C→O 时 modal 比较 {a} 与候选的 OCT 标签；combined 比较 {a,b} 与候选综合标签。病例综合标签包含某种改变，并不要求每种影像都显示该改变。

病例候选依次排除：同 tie_key（同病例）、同中心与患者键、共享任意 CFP/中心六张图的内容 MD5。三类计数互斥，计数单位是 query–gallery 对；重复检查使用病例全部可用七图，不局限于正在编码的模态。私有包复用已有43,834份哈希，仅补算原缓存缺失的4份；size/mtime检查保证后续变化能被发现。没有提供 hashes 时才按文件内容生成缓存。

## 排序与指标

向量转 FP64 再 L2，余弦点积降序；并列文本按 class_id 升序，病例按 tie_key 字典序。无 softmax、阈值或 reranker。

对于标签有效 query q，先排除候选，再构造有效库 Gq：

- 文本 gain=1（候选类别属于 q 的标签），否则0。
- 病例 gain=|Yq∩Yg|/|Yq∪Yg|。训练时可能排除部分重叠；**评测保留部分重叠**。
- gain>0 即 Hit/Recall 的相关项。明确正常与正常 gain=1，与疾病为0。

令 Rq 为 Gq 中全部 gain>0 的候选，Tk 为模型前K项：

```
Hit@K    = 1[至少一个相关项在 Tk 中]
Recall@K = |Tk∩Rq| / |Rq|
DCG@K    = Σ(r=1..min(K,|Gq|)) gain_r / log2(r+1)
NDCG@K   = DCG@K / IDCG@K
```

IDCG 从**整个有效库**按真实 gain 降序取前K项，不从模型返回的前K项构造；使用线性 gain，不用2^gain−1。K固定1/3/5。文本 Recall 分母是 query 已知真实类别数；病例 Recall 分母是排除后的全部相关病例数。候选不足K时只使用实际候选。

没有相关候选的 query 不参与三个指标的均值，单独计数。均值按有效 query 等权，不按患者或类别等权。一个患者多次检查会贡献多个 query。全部 query 无相关候选时省略分数字段，保留计数。

输出数量关系：

```
n_queries_available = n_queries_missing_or_invalid_label + n_queries_label_valid
n_queries_total = n_queries_label_valid = n_queries_valid + n_queries_no_relevant
coverage_valid_of_label_valid = n_queries_valid / n_queries_label_valid
coverage_valid_of_available = n_queries_valid / n_queries_available
```

分母为0时覆盖率为 null。`coverage.csv` 保留所有任务，即使无任何有效 query。例：正式外部 modal C→O 为945个标签有效query，其中294有效、651无相关候选；须同时报告294/945，而不能只报告均值。

只有同时选择 C-C/O-O/B-B 且各方向有分数时，`same_modal_ndcg5` 给出三个方向 NDCG@5 的等权均值。不会把跨模态或文本方向混入。modal、combined 分别报告；验证集用于预先指定的选模，测试结果不回改 checkpoint。此工具本身不选 checkpoint。

## 可追溯输出

summary包含生效配置、计数、排除、指标、编码指纹；每query JSONL包含标签、top5、n_relevant、n_excluded及指标。score NPZ保存FP32矩阵、query/gallery ID；排序在FP64执行，极近分数不宜用FP32导出矩阵反推并列。

`audit-results` 对JSONL记录的query指标按原规则重新聚合，检查summary。重新核验完整排名需编码缓存和同一清单，而非仅top5。结果保存在私有位置。

缓存指纹包含源代码和配置/数据版本，complete内有NPZ SHA。病例ID关联而非文件行号猜测。图像内容MD5的历史来源、数据清单SHA、正式文本SHA及配套包对应提交在私有bundle.json/audit.json登记。

## 候选库与角色（新增协议）
reference_train保持历史内部训练库；within_split明确同划分候选库，跨模态也排除同病例/患者/共享图像MD5。query_ids/gallery_ids分别控制角色，编码去重，smoke限制分别应用。schema3缓存指纹记录角色与模态集合；完整相关候选集合决定Recall分母和线性IDCG。新协议固定既有checkpoint的结果不称为新协议最优。
