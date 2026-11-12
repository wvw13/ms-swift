# FunEmbed：训练、选模与评测

本包用于在模型所在环境内完成眼科单向量表征训练和评测，最后带回汇总结果。影像不随代码交付。当前模型入口为 Qwen3-VL-Embedding-2B，直接编码输入、提取末尾有效 token 并归一化，不先生成诊断。

## 1. 任务与配置

- **T1**：C/O/B影像分别检索对应疾病视觉定义。每行一正例、K个互异显式负例；多标签其他阳性不进入负例。
- **T2**：C→C、O→O、B→B，不同患者且对应标签集合完全一致为正例。
- **T3**：C→O、O→C，不同患者、两端模态标签完全一致为正例。不是同病例跨模态配对。
- T2/T3部分重叠标签不参与比较；负例标签无交集。同行query、正例与各负例的患者互异。空标签不视为正常。

C为一张CFP；O为007–012六张OCT联合输入；B为CFP在前、六张OCT在后联合输入。C/O/B分别依据CFP/OCT/综合标签。正常query概率0.2，层内先均匀抽患者再抽病例。

| 配置文件 | 训练内容 | 更新预算 | 验证情况 |
|---|---|---:|---|
| original_2B.json | 无训练，同条件检索与LP | 0 | 原实现GPU有证据，交付入口CPU通过 |
| T1_C.json / T1_O.json | 单模态图文 | 未指定；提供短测 | 采样CPU通过 |
| T1_B_historical.json | 仅B图文，四卡micro2×GA4 | 1183 | 历史原实现完成；交付复现未跑 |
| T1_COB.json | 三方向图文1:1:1 | 1200 | 原实现K4双卡GA16六步短测通过 |
| T2.json | 三方向病例1:1:1 | 1200 | 原实现正式训练完成 |
| T2_historical.json | 保留val→train选模 | 1200 | 原实现历史协议 |
| T3.json | C→O、O→C等比例 | 未指定；提供短测 | 独立交付双卡短测通过 |
| E3.json | T2:T3=3:2，五方向等比例 | 1200 | 交付版K4双卡GA16短测通过 |
| E4.json | T1:T2:T3=3:3:2，八方向等比例 | 1200 | 交付版K4双卡GA16短测通过 |

完整验证范围见[验证记录](docs/VALIDATION.md)。配置可用与外部环境实际运行成功分别记录。

## 2. 环境安装

最低 Python 版本为3.13，安装元数据与CLI入口均检查；不支持Python3.11/3.12。实际验收使用Python3.13，安全解压保留data过滤器、路径/链接检查和逐文件SHA核验。使用独立环境，不改已有训练环境。实验室验证环境为Linux、Python3.13、PyTorch2.9.1 CUDA12.8、Transformers4.57.3；关键版本在`constraints-tested.txt`。外部驱动与CUDA适配先由运行者确认。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.9.1' 'torchvision==0.24.1' --index-url https://download.pytorch.org/whl/cu128
# 使用锁定的官方框架源码，不以任意最新版替代
 git clone https://github.com/modelscope/ms-swift.git framework
 git -C framework checkout 21ada46205e0abfc6ab33ef8b6dd260cc4eccf8d
python -m pip install -c constraints-tested.txt -e ./framework
python -m pip install -c constraints-tested.txt transformers peft accelerate numpy Pillow qwen-vl-utils scikit-learn tensorboard safetensors
python -m pip install --no-deps .
```

`check --environment`按528个框架源文件哈希核对实现，避免不同Trainer行为混用。代码引用了框架的embedding模板、单正例InfoNCE和恢复机制。需要完整基座目录及其`config.json`、tokenizer/processor配置、权重、`scripts/qwen3_vl_embedding.py`。下载/访问权限由运行环境配置，不把凭据写进JSON。

## 3. 接入已交付数据

```bash
export MODEL_ROOT=/your/models/Qwen3-VL-Embedding-2B
export DATA_ROOT=/your/data/image_root
export PRIVATE_INPUT_ROOT=/your/private/funembed_inputs
export SWIFT_ROOT="$PWD/framework"
export OUTPUT_ROOT=/your/outputs/funembed
```

PRIVATE_INPUT_ROOT需要`manifest.jsonl`、`classes.json`、`texts.json`。这三份正式文件通过实验室私下交接，不在代码包内。影像已交付不代表这三份文件已到位，请先核对。字段和映射见[数据与协议](docs/PROTOCOL.md)。相对影像路径按DATA_ROOT解释。预检查保留ID、标签、划分、图像顺序和内容MD5；迁移后文件时间不同会核对内容MD5，再在本地生成规范化元数据。不会复制影像。

也可通过`--paths paths.json`覆盖五个路径；相对路径以配置文件所在目录解析。示例`configs/paths.example.json`单独用于`--paths`，不是实验配置。

## 4. 先检查，再跑短测

CPU合成例不加载真实模型：
```bash
python -m funembed_train synthetic-demo --output /tmp/funembed_demo_unique
```

真实数据CPU检查与八方向日程：
```bash
python -m funembed_train check --config configs/E4.json --environment
python -m funembed_train prepare --config configs/E4.json --smoke
```

每次迁移验收使用新的OUTPUT_ROOT（或新的experiment_id），不要复用旧版已完成smoke目录。确认两张卡空闲，再执行短测。`--gpus`填物理卡号，torchrun内部编号为0、1。
```bash
nvidia-smi
python -m funembed_train run --config configs/T3.json --smoke --gpus 0,1
python -m funembed_train run --config configs/E3.json --smoke --gpus 0,1
python -m funembed_train run --config configs/E4.json --smoke --gpus 0,1
```

每次串行，不同时执行上面三条。短测为独立`*_smoke`目录、两个完整调度周期：T3第2步保存退出再恢复到4；E3为5→10；E4为8→16；T1-C/O/B为3→6。保留正式K4、双卡micro1×GA16。随后对两个checkpoint在固定前16例val上联合编码、检索、audit、选模，固定选中的checkpoint后，对内外测试各前16例运行检索；再做32train/16val、内外各16例、seed0、2轮的小型分类LP训练与测试，最后导出。测试子集不参与选模。小样本使用`short_fixed_subset`协议，不产生正式成绩。

外部首次验收要完成上述训练→保存→恢复→加载→验证选模→检索→小LP→导出。详细状态写在`stages/`、`resume_acceptance.json`和日志中；若某方向无相关候选，覆盖率会显示出来。

## 5. 正式运行

短测通过后，由研究负责人确定执行哪些配置。以下命令均为独立实验，不会互相继承权重：
```bash
python -m funembed_train run --config configs/T1_B_historical.json --gpus 0,1,2,3
python -m funembed_train run --config configs/T1_COB.json --gpus 0,1
python -m funembed_train run --config configs/T2.json --gpus 0,1
python -m funembed_train run --config configs/T3.json --smoke --gpus 0,1
python -m funembed_train run --config configs/E3.json --gpus 0,1
python -m funembed_train run --config configs/E4.json --gpus 0,1
```

T1-C、T1-O、T3未预设正式长训练预算，未填写`training.steps`时正式入口拒绝运行；短测无需补预算。选择单方向只改JSON：例如`tasks={"T3":1}`、`directions={"T3":["C-O"]}`；T2单方向同理。训练方向与`evaluation.case_directions`独立，单方向训练仍可评测五方向。

建议screen内运行：
```bash
screen -S funembed_external
# screen内激活环境、设置以上五个路径，然后执行一条run命令
# Ctrl+A、D脱离；screen -r funembed_external重新接入
```

一个`run`顺序完成训练、全部候选val编码选模、选中模型内外测试、C/O/B三头LP、汇总导出。阶段失败即停，不循环重试。不会删除checkpoint，不自动启动其他实验。

有效query batch=world size×micro batch×GA。默认2×1×16=32；K4是每query四个显式负例，不是31个批内负例。任务权重控制更新频次，任务内方向轮换，不再乘loss权重。同次更新所有rank/累积微步保持同方向。训练编码候选也参与梯度，不用冻结向量缓存替代。

LoRA语言侧q/k/v/o及gate/up/down，默认rank16/alpha32/dropout0.05；原主干、视觉与连接层冻结。单正例InfoNCE温度0.1、AdamW、lr5e-5、WD0.01、cosine、warmup0.05。训练与检索每图1024图像token、8192上限、BF16/SDPA，检索batch1。参数可在training对象修改；变化后使用新experiment_id。

## 6. 恢复与分阶段操作

保留同卡数、micro/GA、配置及数据日程。不要把短测checkpoint作为正式训练起点，不默认支持改变卡数恢复。
```bash
python -m funembed_train resume --config configs/E3.json --gpus 0,1 --checkpoint 120
# 恢复完成后继续剩余流程，已完成阶段会校验命令后复用
python -m funembed_train run --config configs/E3.json --gpus 0,1
```

只在checkpoint完整且原进程已经退出后恢复。模型/optimizer/scheduler、Trainer随机状态和固定采样位置一起恢复；日志保留重复执行段，汇总loss使用最后checkpoint的trainer_state。

分阶段入口：
```bash
python -m funembed_train train --config configs/E3.json --gpus 0,1
python -m funembed_train evaluate --config configs/E3.json --checkpoint 120 --split val --gpus 0
python -m funembed_train select --config configs/E3.json
python -m funembed_train lp --config configs/E3.json --checkpoint 960 --modality B --gpus 0
```

上面的960仅说明参数形式，实际必须使用该实验selected.json中的step。选择器要求全部预定候选结果完整、数据/角色/指令/batch等一致；checkpoint权重允许变化。中断评测或LP保留失败状态和产物，请先定位日志；不要把失败标记手工改成完成。解决原因并确认原PID退出后，可执行`python -m funembed_train retry-stage --config configs/E3.json --stage 失败阶段名`：归档旧状态和日志，不删除产物；再显式执行run或相应入口。缓存不完整/不兼容仍会被拒绝，不能通过重置阶段绕过指纹。

## 7. 评测与结果

新实验：val→val，modal C-C/O-O/B-B的NDCG@5等权平均选模，同分选较早；跨模态和图文val轨迹另存，不参与该主指标。固定模型后才测internal→internal和external→external。T1-B历史配置按B-LP验证mAP选模，T2_historical按val→train选模，分开报告。

检索：Hit/Recall/NDCG@1/3/5，modal/combined独立；图文仍是22类文本。病例排除自身、同中心患者及共享影像MD5，然后计算相关集合、Recall分母与IDCG。Jaccard线性gain、并列排序及无相关候选处理复用原评测代码。输出有效数量、无相关候选和覆盖率。

LP：使用独立分类指令及原分类像素预算，不能复用I1检索缓存；冻结编码器，只训练Linear(2048,22)，综合标签、共同完整病例队列，BCE/AdamW，头batch256、最多1500轮/patience60，三头seed0/1/2；验证mAP选头、验证micro-F1选阈值后固定到测试。AUC指逐疾病ROC曲线下面积的宏平均。分类头三种子不代表三次编码器训练。

输出结构：
```
OUTPUT_ROOT/experiment_id/
  resolved.json, environment.json, source_fingerprints.json
  inputs/, pairs/                 # 本环境私有，不默认导出
  train/checkpoint-*/              # 始终留在模型环境
  logs/, stages/, status.json
  evaluation/checkpoint-N/{val,internal_test,external_test}/
  selected.json, lp/checkpoint-N/{C,O,B}/
  result_export.tar.gz
```

## 8. 导出允许带回的结果

```bash
python -m funembed_train export --config configs/E3.json --output /your/export/E3_summary_v1.tar.gz
```

导出写入目标目录的唯一`.partial`文件，完成gzip读至EOF、安全解压、完整文件清单和SHA检查后原子替换最终包；这之后工作流才写completed。失败原因记录在status.json的export阶段，保留临时文件。

```bash
# 检查已生成包，不加载模型或GPU
python -m funembed_train verify-export --output /your/export/E3_summary_v1.tar.gz
# 自动流程的export失败后，用原配置再次run，只恢复导出
python -m funembed_train run --config configs/E3.json --gpus 0,1
```

导出前已完成的计算由export_ready.json记录。旧版本completed但残缺的最终包会保留为`.invalid.<时间戳>`，再从原产物重建；不重跑训练、编码或LP。`.partial`残留留作排查，不会被当成结果。已有有效包内容相同则验证后复用，内容不同则保留并报错，请用export命令指定新文件名；不覆盖有效历史结果。export_complete.json保存最终包SHA与复用/损坏包保留信息。

默认包括脱敏配置、源代码/数据指纹、环境、loss/LR、验证轨迹、选模、汇总检索和LP指标/阈值、阶段耗时。每个包带文件SHA清单并解压核验。不会自动包含影像、病例ID、权重、向量或完整stdout。

只有对方环境允许时才加`--include-queries`、`--include-predictions`、`--include-embeddings`或`--include-weights`；这些选项彼此独立，默认全关闭。模型不能导出时保留全部checkpoint直至验证和测试结束。

## 9. 常见问题

- 缺三个私有输入文件：向数据交付人确认，不用示例数据替代正式数据。
- framework lock不同：核对官方commit与本地改动，不能关闭验证后宣称同版本。
- 缓存指纹不同：新建输出，检查模型、adapter、输入、指令、batch、processor与代码。
- OOM/超长：停止对应任务核查日志，不降低切片数或图像预算后沿用旧实验名。
- 恢复配置不同：不要跨卡数/GA恢复，保留原目录另做独立验证。
- 模型目录缺官方embedding脚本：LP路径需要完整基座推理实现；不能用任意生成式checkpoint猜测表征位置。

本地交付版已完成CPU检查和T3/E3/E4双卡真实小闭环，包括恢复、选模、检索、微型LP与结果导出；详见[GPU验收记录](docs/GPU_VALIDATION.md)。外部同学先在自己的环境完成短测，再运行确定的正式配置。本包没有自动发布或上传功能。
