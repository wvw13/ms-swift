# FunEmbed 统一检索评测

在模型所在机器上用 ms-swift 编码影像和文本，再用 CPU 计算检索指标。支持同模态、跨模态病例检索，以及 C/O/B 到 22 类疾病名称或模态疾病定义的检索。标签协议、候选库、编码方式和覆盖率随结果保存。

当前原生加载路径是 **Qwen3-VL-Embedding 的完整/合并权重，或原始基座 + PEFT LoRA**，使用 `qwen3_vl_emb` 的直接 embedding 推理。历史 CLIP、SigLIP、Qwen 默认指令评测另有 `legacy` 配置。分类 LP、训练和生成文字后再取向量不属于这个原生入口；其他表征方式的最小接入位置见 [模型接入](docs/MODEL_INPUT.md)。

## 1. 需要准备什么

| 来源 | 内容 |
|---|---|
| 本 Git 分支 | 评测源码、CLI、协议文档、配置模板、CPU 合成示例和测试 |
| 实验室私有配套包 `funembed_eval_linyan_20261109_v3_2.tar.gz` | 固定的 6,263 例轻量清单、22 类映射、正式名称与 C/O/B v2 定义、图像重复检测依据、实验室路径模板、计数与校验文件 |
| 实验室共享路径 | 原始影像；路径已经写在私有配置模板中，可用 `--image-root` 修改 |
| 模型所有者 | 本机模型、可选 LoRA、该模型实际的 embedding 推理方式；模型目录须带 tokenizer/processor |

配套包通过实验室私有渠道获取，**不含原始影像或模型**。清单和输出含病例关联信息，保存在个人私有目录中，不上传公共仓库。正式疾病定义也由私有包提供；CPU 示例只有标明 `SYNTHETIC` 的人工类别。

目录：`funembed_eval/` 源码，`configs/` 可编辑协议模板，`docs/` 详细说明，`tests/` 行为检查。这个子目录可独立安装，不要求从仓库根目录启动。

## 2. 安装环境

先获取完整分支，记录仓库位置：

```bash
git clone --branch funembed/unified-eval-20261107 https://github.com/wvw13/ms-swift.git ms-swift-funembed
cd ms-swift-funembed
FUNEMBED_REPO="$PWD"
python3 -m venv .venv-eval
. .venv-eval/bin/activate
python -m pip install "$FUNEMBED_REPO/examples/funembed_eval"
funembed-eval --help
```

CPU 示例只需要 Python ≥3.11 和 NumPy。真实 GPU 编码再在这个独立环境中安装框架依赖：

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c "$FUNEMBED_REPO/examples/funembed_eval/constraints-tested.txt" -e "$FUNEMBED_REPO"
python -m pip install -c "$FUNEMBED_REPO/examples/funembed_eval/constraints-tested.txt" \
  transformers peft accelerate qwen-vl-utils Pillow numpy
```

实测源环境：Python 3.13、torch 2.9.1+cu128、torchvision 0.24.1、transformers 4.57.3、peft 0.18.1、accelerate 1.12.0、qwen-vl-utils 0.0.14、Pillow 12.0.0、NumPy 2.3.5。框架版本字符串 `4.6.0.dev0`，来源提交 `21ada46205e0abfc6ab33ef8b6dd260cc4eccf8d`；本分支保留该框架，源码锁定见 `FRAMEWORK_LOCK.json`。无需 FunEmbed 训练插件。CUDA 驱动与 torch CUDA 构建需匹配。

已有兼容环境可直接安装本子包；不必修改训练环境。`--framework-path` 指框架仓库根目录，程序核对实际导入位置。`check --environment` 只检查依赖元信息，不加载模型或初始化 CUDA。

## 病例候选库协议

主要使用流程显式选择 `within_split`：val→val、internal_test→internal_test、external_test→external_test。每例轮流作query，逐query排除同病例、同中心患者和共享图像MD5病例；用于检查该划分内的跨患者疾病检索。模型是否训练见过这些病例，需要核对实际训练记录。

`reference_train`用于复现历史“新query检索已知训练参考库”场景。CLI默认仍保留该旧语义，下面的使用命令均显式指定协议。图文gallery始终为22类文本，两种病例gallery的绝对分数不混排。现有960按旧val→train选出；未来新协议选模应使用val→val。

query/gallery分别选择并独立限制smoke的2/4例，编码按split、模态、病例去重。缓存schema3记录角色和实际输入集合，不直接接收旧缓存改名。

## 3. 快速开始

### CPU 示例（人工数据，无影像、无模型）

```bash
funembed-eval demo --output ./synthetic_demo
funembed-eval check --config ./synthetic_demo/config.json --environment
funembed-eval encode --config ./synthetic_demo/config.json
funembed-eval evaluate --config ./synthetic_demo/config.json
funembed-eval audit-results --output ./synthetic_demo/results
```

目录必须是新目录。合成向量只用于核验接口和输出，不能解释为模型成绩。

### 实验室真实模型评测

将私有配套包放到自己的目录，解压并校验：

```bash
umask 077
mkdir -p "$HOME/funembed_eval_data"
tar -xzf funembed_eval_linyan_20261109_v3_2.tar.gz -C "$HOME/funembed_eval_data"
FUNEMBED_BUNDLE="$HOME/funembed_eval_data/funembed_eval_linyan_20261109_v3_2"
FUNEMBED_COMMIT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["code_commit"])' "$FUNEMBED_BUNDLE/bundle.json")
git -C "$FUNEMBED_REPO" checkout "$FUNEMBED_COMMIT"
python -m pip install "$FUNEMBED_REPO/examples/funembed_eval"
funembed-eval verify-bundle --bundle "$FUNEMBED_BUNDLE"
```

以下命令针对 **直接编码输入、提取末尾有效 token 的 ms-swift embedding 模型**。如果实际推理先生成描述，请先按 [模型接入](docs/MODEL_INPUT.md) 提供实际推理调用。

```bash
read -r -p '完整/合并模型或原始基座目录: ' FUNEMBED_MODEL
read -r -p 'LoRA 目录（完整模型直接回车）: ' FUNEMBED_ADAPTER
FUNEMBED_ARGS=(--bundle "$FUNEMBED_BUNDLE" --model "$FUNEMBED_MODEL" \
  --framework-path "$FUNEMBED_REPO" --model-type qwen3_vl_emb \
  --representation swift_embedding)
if [ -n "$FUNEMBED_ADAPTER" ]; then FUNEMBED_ARGS+=(--adapter "$FUNEMBED_ADAPTER"); fi

# 首次只选 B→B 与 B→文本。改为自己的空闲物理 GPU。
funembed-eval init-lab "${FUNEMBED_ARGS[@]}" --run-dir ./eval_val_B \
  --config ./eval_val_B.json --case-directions B-B --text-modalities B \
  --case-protocol within_split --query-split val --smoke
funembed-eval check --config ./eval_val_B.json --files --environment
set -o pipefail
CUDA_VISIBLE_DEVICES=0 funembed-eval encode --config ./eval_val_B.json 2>&1 | tee ./eval_val_B.encoding.log
funembed-eval evaluate --config ./eval_val_B.json
funembed-eval audit-results --output ./eval_val_B/results
```

`init-lab` 只写配置；`check` 只做 CPU 检查。`encode` 才会加载模型。默认 batch size 1；每图 1024 image-token 预算，序列上限 8192，超长报错而不截断。物理卡由 `CUDA_VISIBLE_DEVICES` 控制，进程内使用 `cuda:0`。

**上面的首次模型短测已启用 `--smoke`**。它固定最多 2 个 query、4 个 gallery，仍保留完整 22 类文本，协议名带 `_smoke`。再运行相同的 check→encode→evaluate→audit 命令。短测输出与正式输出分开。

上面的运行只检查验证集小样本接口。短测成功后，以下命令分别运行完整验证集、内部测试、外部测试，配置、缓存和输出各自独立。模型选择只使用验证集；两个测试集用于固定模型的最终报告。

```bash
for FUNEMBED_SPLIT in val internal_test external_test; do
  FUNEMBED_RUN="./within_${FUNEMBED_SPLIT}_all"
  funembed-eval init-lab "${FUNEMBED_ARGS[@]}" \
    --run-dir "$FUNEMBED_RUN" --config "${FUNEMBED_RUN}.json" \
    --case-protocol within_split --query-split "$FUNEMBED_SPLIT" \
    --case-directions C-C O-O B-B O-C C-O --text-modalities C O B
  funembed-eval check --config "${FUNEMBED_RUN}.json" --files --environment || break
  CUDA_VISIBLE_DEVICES=0 funembed-eval encode --config "${FUNEMBED_RUN}.json" || break
  funembed-eval evaluate --config "${FUNEMBED_RUN}.json" || break
  funembed-eval audit-results --output "$FUNEMBED_RUN/results" || break
done
```

只需要一个划分时，把循环列表改为该划分；只需病例检索时不传`--text-modalities`，可减少文本编码。

历史参考库复现用单独目录，显式选择`reference_train`：

```bash
funembed-eval init-lab "${FUNEMBED_ARGS[@]}" --run-dir ./reference_internal_test \
  --config ./reference_internal_test.json --case-protocol reference_train \
  --query-split internal_test --case-directions C-C O-O B-B O-C C-O
funembed-eval check --config ./reference_internal_test.json --files --environment
CUDA_VISIBLE_DEVICES=0 funembed-eval encode --config ./reference_internal_test.json
funembed-eval evaluate --config ./reference_internal_test.json
funembed-eval audit-results --output ./reference_internal_test/results
```

只做图文：省略 `--case-directions`；只做病例：省略 `--text-modalities`。必须至少选一项。选择图文时不会编码病例 gallery；选择 O→C 时只编码 O query 与 C gallery。历史 `complete_B + legacy + per_image` 保留七图连续展平的原批处理顺序。

## 4. 参数与配置

| 参数/字段 | 含义 |
|---|---|
| `model / adapter` | 完整基座与可选 LoRA；不能把仅 adapter 的目录当成完整模型 |
| `framework_path` | 兼容 ms-swift 源码根目录 |
| `image_root` | 所有清单相对影像路径的根目录 |
| `query_split` | val / internal_test / external_test；由 case_protocol 决定病例 gallery |
| `case_protocol / gallery_split` | reference_train 使用 internal_train；within_split 使用 query 的同划分 |
| `case_directions / text_modalities` | 本次实际运行的方向与文本模态 |
| `protocols` | modal 与 combined 分开计算 |
| `instruction / text_kind` | I0/I1、name/definition；改变后新建缓存 |
| `aggregation / cohort` | joint/per_image、input_available/complete_B |
| `cache / output` | 编码目录和指标目录，必须不同；输出存在时拒绝覆盖 |

JSON 中相对路径以配置文件所在目录为基准，CLI 路径覆盖以当前目录为基准。check/encode/evaluate 都支持 `--model --adapter --framework-path --image-root --cache --output --batch-size --device --case-directions --text-modalities`；如有覆盖，后续步骤必须使用同样参数，推荐修改 JSON 后统一运行。

私有包模板固定正式数据和文本；需要研究另一文本版本时，新建协议和输入版本，而不是修改已经交付的 bundle。

## 5. 输入与编码

清单字段、C/O/B、患者排除和公式见 [评测协议](docs/PROTOCOL.md)。每个病例影像分支的 user 消息只含 1/6/7 个 `<image>`，图片列表依序对应；system 使用 I0 或 I1。疾病分支单独编码候选文本。病例标签、患者、中心和划分均不拼入模型消息。

同一运行的图像与文本由同一模型/adapter 编码。原生 Qwen 路径使用最终层最后有效输入 token，并作 L2 归一化；不会生成答案。编码检查行数、维度、有限值、单位范数和序列长度；加载 LoRA 后核对实际 adapter 张量。

历史复现使用 `configs/historical_*.json`；当前原模型与 LoRA 使用 `current_base.json/current_lora.json`。模板里的 `/path/to/...` 必须替换，实验室使用者优先用 `init-lab` 自动填入私有包路径。[PDF 与当前协议对照](docs/HISTORICAL_PROTOCOL.md)。

## 6. 输出与复核

```text
run/
  encoding/  # 向量npz、fingerprint、encoder_report、resolved_config、complete
  results/
    resolved_config.json
    summary.json
    results_long.csv
    coverage.csv
    complete.json
    modal/    # per_query__*.jsonl、scores__*.npz
    combined/
```

Hit/Recall/NDCG@1/3/5 是有效 query 的等权均值。`coverage.csv` 同时给出输入可用数、标签有效数、无相关候选数、有效数和两种覆盖率。`n_queries_total` 沿用旧实现，指**标签有效 query 数**；不要将它理解为全部输入数。没有有效 query 时不输出伪造的零分。

`audit-results` 从逐 query 已记录指标重算汇总值；不重新编码，也不从 FP32 score 文件重建全排序。score 文件用于诊断，精确排序口径为 FP64 归一化后的余弦。

缓存锁定模型配置/权重大小与时间戳、adapter SHA、数据/文本/类别映射、图片 stat、所选清单与任务、指令、预算、框架及包源码、依赖版本。只有匹配且 `complete.json` 完整时才复用。基础权重以 stat 而非全文件 SHA 登记；替换权重后使用新缓存，不人为保留旧 stat。部分失败缓存不自动续写，保留现场后换新目录。

单进程编码使用一张卡；支持在不同 GPU 独立运行不同配置，缓存和输出目录必须分开，不支持共享目录并发写入。本入口未实现单次编码 DDP。

## 7. 常见问题

- **缺 model / representation / processor**：填写实际模型目录及 embedding 方式；泛用 Qwen 架构名不能证明其表征协议。
- **Permission denied**：在师兄自己的账户运行 `namei -l` 检查数据路径目录链和文件读取；由数据管理员处理授权，勿扩大整个数据目录权限。
- **image changed since checksum bundle**：图片 size/mtime 与重复检测依据不符；核对是否指向正确数据版本，更新校验依据后再运行。
- **cache fingerprint mismatch**：使用新缓存，或恢复完全一致的配置/代码/环境；不跳过检查。
- **OOM**：先用 batch size 1。不要静默减少 OCT 切片或图像预算来混报同一协议。
- **sequence exceeds max_length**：检查实际模板、额外 token 与输入预算。需改预算时单独命名协议。
- **没有有效 query**：查看覆盖率和标签状态；空标签不当正常。跨模态无共享标签候选也会产生无相关 query。

## 8. 验证与资源

CPU 流程包括独立目录安装/CLI、合成闭环、手算多标签指标、同病例/患者/重复图像排除、并列排序、无相关候选、双标签协议、任务选择、缓存失效、正式输入一致性及已有结果复算。

我们自己的原始Qwen3-VL-Embedding-2B与E2 checkpoint-960已完成C/O/B、跨模态和22类文本的发布入口真实GPU短测，包括同split角色去重、缓存、排除和指标复核。完整新协议结果使用核验后的历史向量CPU重检索；发布入口重新编码全量的逐值/指标复现尚未确认。师兄自身checkpoint需按上述smoke流程接入验证。

编码比较请固定batch size、样本顺序和批次组成，query/gallery采用同一配置。已验证小样本在同batch重复运行和两个入口之间逐值一致，但batch 1与4会改变部分向量、排序及指标。新实验先固定batch 1；历史batch 4缓存的精确复现应保留历史配置，不能用batch 1查询混入batch 4 gallery冒充全流程复现。

源实验使用 RTX 5880 48GB；2B 联合编码曾按 batch 1–4 运行。新机器和未知模型以 batch 1 短测确定显存与耗时；CPU 合成结果不能估算真实 GPU 成本。

测试代码有23个test_函数定义；由于继承与导入，unittest执行41项，不将重复执行计为新增覆盖。

开发检查：`python -m unittest discover -s "$FUNEMBED_REPO/examples/funembed_eval/tests" -v`。安装检查无需作者 cwd 或额外 PYTHONPATH。
