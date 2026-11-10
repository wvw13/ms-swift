# 输入、监督与评测协议

## 私有输入
manifest.jsonl：每条含case_id、input_ids(C/O/B)、tie_key、patient_id、center、split、images.cfp、images.oct(每项slice/path)、labels.cfp/oct/combined(ids/state)、image_md5及image_stats。四个split固定internal_train/val/internal_test/external_test；患者键为center+patient_id。MD5必须与all_paths顺序一一对应；OCT选择007至012且排序固定。缺OCT的记录可用于C，不补图片。

classes.json：22项，id从0至21、label标准名称；texts.json：同序22项，id、name、definition.C/O/B；B严格为`CFP features: {C} OCT features: {O}`。无异常为显式类别，空标签state不转换。预检查保留正式词表全部22类。

图像端仅包含任务指令与图像，不拼入GT、患者、中心或split。训练输入根据模态标签选择监督，标注缺失不参与该任务。未标注类作为T1负例依赖既有标注口径。

## 任务日程
T1阳性轮换、其他阳性排除；T2/T3同标签正例、无交集负例、部分重叠跳过。三个任务共用一个编码器/InfoNCE；单行候选由训练时编码，均有梯度。关闭批内负例。采样池覆盖和排除写入pairs统计，固定seed和提前物化schedule，恢复按位置读。

## 检索
within_split两端来自同划分，编码按样本/模态去重，角色ID集合分别保留。reference_train保留历史库含义。跨模态同样排除同病例、同患者、共享MD5。

modal：C看CFP标签、O看OCT标签、B看综合标签；combined：两端都用综合标签。例如同病例CFP为A、OCT为B，C→O的modal相关性比较A与B，而combined比较各病例综合诊断。

病例gain=标签Jaccard，相交即二值相关；Hit@K为至少命中一次，Recall@K分母为排除后全部相关候选数。NDCG使用线性gain及log2折扣，IDCG由排除后完整候选相关性降序得到。图文单个疾病为二值相关。候选不足K取全部候选；无相关候选单独计数，不纳入有效query均值。指标按query平均，非按患者。相似度为归一化向量点积，稳定tie_key规则沿用评测代码。

输出summary、per_query和complete；audit-results重算逐query汇总，覆盖率为有效query/标签有效query。输入可用数与标签有效数另存。向量维度2048、有限且单位范数，cache指纹包含角色、样本、权重、输入及编码条件；batch改变不能伪装复用。

## 版本与复现
funembed_eval模块按SOURCE_PROVENANCE复制既有发布版本，指标没有分叉重写。训练模块是既有采样/框架入口的可移植封装，新增CUSTOM任务方向配置；不保证与历史物化样本逐行相同，因为迁移输入患者键表达与排序可能不同。历史配置保留方法/预算/选模口径，不把新运行称为历史样本精确重放。精确复原历史轨迹需原pairs、schedule和RNG状态。
