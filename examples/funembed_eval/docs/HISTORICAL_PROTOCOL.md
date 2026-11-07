# 历史统一评测报告与当前配置

参考历史材料为12页《FunEmbed-临研数据集统一评测架构-实验报告》；实验设置和公式用于理解原协议，第10–11页结果表为空。数值结果来自另存的机器可读产物。历史生成式方案另提出先生成描述再取末尾表征，它是方法方案，不是当前接收者checkpoint的推理实现证明。

| PDF页码/实验 | 原协议 | 对应原入口 → 本包入口 | 当前差异和处理 |
|---|---|---|---|
| 1–3 数据流与划分 | 训练3784作为固定库、内部1038/外部984查询；全输入综合标签 | eval/dataset.py → data.py | 私有清单保留6263原记录；current C训练3785，O/B3784；historical配置complete_B恢复旧范围 |
| 3–4 指标 | Hit/Recall/NDCG@1/3/5，病例Jaccard线性gain、全文库IDCG | eval/metrics.py → metrics.py | 原文件保留；补充输入/标签/有效query数量和覆盖率，不改指标均值 |
| 4 实验1 | C/O/B→22类英文疾病名称 | eval/evaluate.py exp1 → evaluate + text_modalities | 当前I1/v2模态定义；name/definition显式配置，正式保持22候选 |
| 5 实验2 | C-C/O-C/O-O/B-C/B-O/B-B | eval/evaluate.py exp2 → case_directions | 当前常用CC/OO/BB/OC/CO；旧方向仍可选择，方向独立计分 |
| 5 实验3 | Qwen2B/8B逐图与联合，O/B文本和O-O/B-B | eval/models.py、aggregate → legacy_encoders、retrieval_core.aggregate | joint/per_image分开记录；保持原native processor及逐图展平顺序 |
| 6–8 模型/消息 | CLIP、SigLIP、Qwen各自处理器；Qwen默认I0、输入末尾表征 | 官方随权重脚本 → legacy；ms-swift → SwiftEncoder | 主线使用qwen3_vl_emb直接embedding；接收者的实际推理协议另确认 |
| 9 图像与聚合 | CFP+中心六张；逐图L2→均值→L2；B的C/O等权 | 原aggregate函数 → 原函数复用 | 当前联合六/七图一次输出；不拼GT；预算1024 image tokens |
| 9–10 排序与排除 | 余弦、类ID/路径破同分，同病例/患者/重复图排除 | 原排名和排除函数 → retrieval_core | 实际源代码排序先转FP64再L2；保留此数值口径，score导出FP32 |
| 10–11 结果 | 待填表，无实测值 | 原始运行summary → 各运行summary | 不把空白表写成完成；不同指令、文本、聚合、标签协议不混排 |
| 11–12 讨论 | 综合标签不一定单模态可见；外部检内部病例 | 原combined → 当前modal及combined并列 | 保留无相关候选统计；验证选配置、测试固定报告 |

当前判别式模型比较须固定数据清单、各模态图像顺序、图像预算、指令、文本版本、标签协议、候选库、排除规则和指标。模型自身的模板/processor/表征位置如有变化，单独登记。检查点由模型训练方在验证集预先选择，本工具不利用测试结果选checkpoint。

历史配置 `configs/historical_*.json` 保留旧默认批量和聚合。当前模板只提供协议，`init-lab` 会把正式私有输入和用户运行参数填入新配置。模型内部新增token、先生成再提取、输入mask等没有由历史PDF自动启用。
