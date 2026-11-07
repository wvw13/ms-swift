# FunEmbed 可移植检索评测

在模型所在机器上编码，随后用 CPU 完成病例及图文检索。本目录可以独立复制；运行时只依赖明确指定的模型、框架、manifest、文本及输出路径。

## 已交付的范围

- 原始 Qwen3-VL-Embedding-2B/8B、同架构判别式 PEFT LoRA：`swift` 后端，直接编码输入。
- 历史 CLIP、SigLIP、Qwen：`legacy` 后端，保留旧原生 processor 与逐图聚合；Qwen 也支持联合输入。
- 五个当前病例方向 C→C、O→O、B→B、O→C、C→O；可配置旧报告 B→C、B→O 等方向；C/O/B→22 类疾病名称或定义。
- combined/modal 两种标签协议、病例/患者/图像重复排除、Hit/Recall/NDCG@1/3/5、覆盖率与逐查询结果。
- 已有原模型、判别式 LoRA 的源实现 GPU 证据；本次可移植版本完成 CPU 检查，迁移后需做少量真实编码确认。生成式模型通过明确的插件接口接入，具体表征协议需先提供。
- 分类 LP、BCE 微调与训练不包含在本包中。

目录：`funembed_eval/` 实现；`configs/` 协议配置；`tests/` 行为测试；`PROVENANCE.json` 原实现来源；`FRAMEWORK_LOCK.json` 框架源码锁定。

## 环境

CPU 检查只需要 Python 3.10+ 和 NumPy；实测 Python 3.13、NumPy 2.3.5。创建独立环境，避免改变已有训练环境：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m unittest discover -s tests -v
python -m funembed_eval --help
```

GPU 评测使用 PyTorch 2.9.1+cu128、Transformers 4.57.3、PEFT 0.18.1、Accelerate 1.12.0、qwen-vl-utils 0.0.14、Pillow 12.0.0。框架为本 fork 基准提交 `21ada46205e0abfc6ab33ef8b6dd260cc4eccf8d`（版本字符串 4.6.0.dev0）；528 个框架 Python 源文件与原实验本机逐文件一致，见 FRAMEWORK_LOCK。评测不需要项目训练插件或梯度缩放补丁。

在完整克隆本交付分支后，进入独立环境安装；下面 `MS_SWIFT_ROOT` 指该完整克隆根目录：

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c constraints-tested.txt -e "$MS_SWIFT_ROOT"
python -m pip install -c constraints-tested.txt transformers peft accelerate qwen-vl-utils Pillow numpy
```

`constraints-tested.txt` 是已使用的关键版本约束，其他框架依赖由其安装元数据解析。CUDA 驱动需与所装 PyTorch 匹配。`encoder.framework_path` 指向这个框架根目录；程序检查实际导入位置，并记录完整框架源码哈希。只复制评测目录时，也可另外检出上述框架提交。

## 数据与文本

manifest 是 JSONL，一行一个病例；标签采用 `classes.json` 的固定 0–21 ID，21 为明确无异常。简化结构：

```json
{"case_id":"synthetic_case_1","tie_key":"center/case_1","patient_id":"synthetic_patient_1","center":"synthetic","split":"val","images":{"cfp":"case_1.fundus.jpg","oct":[{"slice":7,"path":"case_1_007.jpg"},{"slice":8,"path":"case_1_008.jpg"},{"slice":9,"path":"case_1_009.jpg"},{"slice":10,"path":"case_1_010.jpg"},{"slice":11,"path":"case_1_011.jpg"},{"slice":12,"path":"case_1_012.jpg"}]},"labels":{"cfp":{"ids":[0],"state":"disease"},"oct":{"ids":[],"state":"empty"},"combined":{"ids":[0],"state":"disease"}}}
```

图片路径相对 `image_root`；配置文件中的相对路径相对该配置文件。`case_id`、稳定排序用的 `tie_key` 均须唯一。`patient_id` 由数据负责人提供，不从文件名猜测。匹配键为中心+患者，保持原协议。患者原标识不进入消息。输出逐查询文件包含病例关联字段，应留在评测机器或获授权的私有存储。

C=一张 CFP，O=007–012 六张 OCT，B=CFP 在前、六张 OCT 在后。OCT 数组必须按切片递增；缺中心切片时 O/B 不可用，C 可保留。标签状态支持 disease/normal/empty/unknown/conflict，只有前两种参与指标计算，空标签不变成正常；异常输入不补图。

`cohort: complete_B` 复现历史所有模态共用的完整病例范围；`input_available` 按各输入完整性选取当前范围。完整病例数与患者数需由接入数据单独统计。原有患者划分直接沿用。山东数据只允许 external_test。

若已有 FunEmbed phase1 manifest，可只做格式转换，不读取影像：

```bash
python -m funembed_eval.convert_phase1 --manifest /path/to/phase1/manifest.jsonl \
  --image-root /path/to/images --output /path/to/portable_data \
  --cfp-definitions /path/to/description_cfp_v2.json \
  --oct-definitions /path/to/description_oct_v2.json
```

转换保留既定病例、患者、标签、图像顺序；输出不能提交到公共 Git。跨机器移动时修改 image_root，保持相对路径及 tie_key。出现重复 case_id 会报错，由数据负责人先处理，不能静默丢弃。

文本文件是有序 22 项数组：`id`、`name`、`definition:{C,O,B}`。名称保留原字串和标点，B 定义沿用 `CFP features: ... OCT features: ...` 拼接。正式候选始终22类，模态标签频次为零也不删除；OCT 13类只属于另行命名的诊断。合成例子的定义不具有临床含义。

## 编码与协议

| 配置 | 模型后端 | 聚合/指令/文本 | 标签与范围 |
|---|---|---|---|
| historical_qwen.json | legacy Qwen 官方随权重脚本 | 逐图，I0，疾病名称 | combined，完整 B 病例；旧六病例方向 |
| historical_clip/siglip.json | legacy 原生 processor | 逐图，名称；无 Qwen 消息指令 | combined，完整 B 病例 |
| current_base.json | swift 原模型 | 联合，I1，模态疾病定义 | modal/combined 分开，按输入可用性 |
| current_lora.json | swift 基座+adapter | 同上 | 同上 |
| generative_plugin.json | 自定义插件 | 明确生成/直接编码与 pooling | 其余数据/指标配置保持一致 |

I0 system：`Represent the user's input.`。I1 完整原文在 `prompts.py`，影像 system 说明模态、病灶关注点及检索目的，user 只放 `<image>`，次数与图像数相同；文本 system 引导疾病视觉定义编码，user 为候选文本。影像消息不包含 GT、患者、中心、划分。

swift `qwen3_vl_emb` 使用输入末尾有效 token 的最终隐藏状态，attention mask 定位并 L2 归一化，不先生成文字。预算为每图1024 image tokens（上限约1024×1024像素），max_length8192，超长报错，BF16、SDPA。不能把预算当作每张实际分辨率。legacy 后端沿用原 `eval/models.py`，包含显式 `patch_embed_fp32` 选项；它与 swift 的实现分别记录，不能把两者误认为同一编码缓存。

历史逐图入口保留每病例七张连续展平后的分批顺序，CLIP/SigLIP 使用 float32、batch64/32，Qwen2B/8B 使用 BF16、batch8/4；8B 配置为 historical_qwen8b.json。逐图：每张先 L2；O=L2(mean(六个 OCT))；B=L2(0.5*C+0.5*O)。联合：六/七张组成一次输入，一次输出一个向量。保存时保留后端返回的单位向量，不额外重归一化；排序沿用旧实现转 float64 后 L2、余弦点积。

## 从编码到结果

编辑路径后执行（以下会加载 GPU，仅在部署者准备好后运行）：

```bash
python -m funembed_eval check --config configs/current_lora.json
CUDA_VISIBLE_DEVICES=0 python -m funembed_eval encode --config configs/current_lora.json
python -m funembed_eval evaluate --config configs/current_lora.json
```

编码器每进程单 GPU；可把不同模型/协议的独立配置分配到不同 GPU。当前包不提供跨卡切分同一病例清单、DDP 或模型并行。2B 历史批4联合编码峰值约9.50 GiB allocated/13.52 GiB reserved；原训练+验证清单全部 C/O/B 的编码约2.3 GPU小时量级。此数是旧源入口实测；本包单卡耗时需部署后测量，8B 和生成式模型另测。保守起点 batch1，确认序列长度和显存后再提高。

`check` 只检查配置、manifest及候选文本，不加载模型、不确认权重可编码。首次 `encode` 创建全新目录；完整缓存复用时核对配置、输入清单、影像 stat、文本、模型配置/权重 stat、adapter 内容 SHA、processor、代码、框架、依赖版本和 NPZ SHA。图片 stat 是廉价失效检查；若管理员保留大小与mtime替换了图像，需要清除缓存重新编码。模型大权重也采用 stat，不做每次昂贵全量哈希。批大小变化会失效，便于保持数值来源一致。

每个病例的重复图像排除检查其全部可用 CFP+中心六张，独立于当前编码模态。可提供与这些路径一一对应的 `image_md5`，其准确性由数据导出方保证；否则程序读取图片算 MD5，缓存带大小/mtime。排除优先级：同病例→同中心同患者→任意图片 MD5 重复，计数互斥。

输出：编码 NPZ/指纹/权重加载检查；`summary.json`、`results_long.csv`；每任务 `per_query__*.jsonl`、`scores__*.npz`；`complete.json`。Hit=前K至少一个相关；Recall=前K相关数/排除后整个候选库相关数；病例 NDCG 使用 Jaccard **线性** gain，文本用0/1；IDCG从整个有效候选库排序。无相关查询不进入均值，单独记录 `n_queries_no_relevant`，同时报告 total/valid/known/available。相似度并列按类别ID或稳定 tie_key 排序。

选模保持验证集，gallery 为内部训练集。当前 E2 规则为 modal C-C/O-O/B-B NDCG@5 等权平均，同分选较早检查点；本包生成指标，不自动改变模型选择。测试只评已选定模型。combined 与 modal、历史名称与当前定义、不同聚合和指令分别报告，不混排。

## 无 GPU 最小例子与检查

```bash
python -m funembed_eval demo --output /tmp/funembed_cpu_example
python -m funembed_eval check --config /tmp/funembed_cpu_example/config.json
python -m funembed_eval encode --config /tmp/funembed_cpu_example/config.json
python -m funembed_eval evaluate --config /tmp/funembed_cpu_example/config.json
python -m unittest discover -s tests -v
```

synthetic 后端只用于接口测试，不读取影像、不计算模型表征。测试覆盖输入次序、标签冲突、空标签、mask优先级、并列排序、全库Recall分母与线性IDCG、无相关查询均值、缓存失效和双协议端到端。源指标文件逐字复制；7个排名/排除函数 AST 与原实现一致。另对原模型/E2外部 OCT 文本的四组原始逐查询结果重新计算，共3694条，所有指标与原结果一致（1e-12容差）；真实数据未包含在仓库。

## 生成式模型接入

先提供：基座架构；完整/合并/LoRA格式；adapter_config/target_modules/modules_to_save；新增token及embedding变化；processor；直接编码或先生成；层、token/池化位置；system/user；图像与生成预算。只修改 `model` 路径不等于接通了生成式表征。

插件文件导出 `create_encoder(config)`，返回对象须有 `report:dict` 和 `encode(branches) -> (vectors,tokens)`。branches 是 messages/images；vectors 为二维 float32、每行L2单位向量，顺序不可改变；tokens 返回每行实际输入token数（无法测得则-1并在report说明）。`encoder.representation_protocol` 写明上述协议，纳入指纹。插件自行实现必要的生成和 pooling，不把 encoder 的输出替换为诊断标签；可使用师兄本机权重而不传输checkpoint。

接入后先在少量样本上检查加载张量、输出维度/单位范数、padding/截断、重复输入稳定性及影像顺序，再运行完整指标。当前具体生成式产物尚未提供，真实GPU接入验证待进行。

## 常见问题

- 缓存指纹不符：检查指令、标签清单、路径映射、文本、批大小及模型；新建缓存目录。
- `unexpected imported swift source`：检查独立环境和 framework_path；不要覆盖正在训练的框架。
- O/B不可用：核对007–012完整性，不重复切片补齐。
- 空标签较多/无相关候选多：查看覆盖计数和协议，不把缺失标签当正常，也不把无相关查询记0混入均值。
- 生成式 adapter 张量/词表不匹配：先确认新增token、模块和processor，再实现插件。
- OOM/长度超限：先降低推理批量；改变图像预算或切片数需要新协议ID。
