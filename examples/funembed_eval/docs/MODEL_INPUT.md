# ms-swift 模型接入

主流程：`TransformersEngine → InferRequest(messages, images) → EmbeddingResponse → [N,d]`。检索层不依赖模型维度；图像和文本必须在模型定义的同一表征空间。

## 原生已实现路径

- Qwen3-VL-Embedding 完整模型/合并权重：model目录包含config、权重、tokenizer和processor。
- 原基座 + PEFT LoRA：model指基座，adapter指checkpoint，包含adapter_config.json与adapter_model.safetensors。加载后使用PEFT导出已加载权重，与保存张量逐一核对。
- 明确指定 `model_type=qwen3_vl_emb`、`representation=swift_embedding`；模板 `qwen3_vl_emb`，task_type=embedding，BF16/SDPA，最后有效输入token/最终层/L2。

模型config中的通用 `Qwen3VLForConditionalGeneration` 架构名不能区分训练/推理语义。CPU检查读取checkpoint的args.json中模型类型、task_type、template、target_modules、modules_to_save等有限字段，并显示与本次编码协议的区别。训练任务是生成式，并不自动决定必须生成后取表征；真正采用哪种表征由其实际推理实现确定。

原模型和当前判别式LoRA的源实现有GPU证据。完整/合并自定义权重虽然遵循相同加载接口，仍需用实际模型短测确认。新增token、resize embedding、额外modules_to_save或改processor时，提供完整保存产物，不能仅复制部分adapter。

## 对模型所有者最少需要的信息

1. 本机完整/基座模型路径；如为LoRA，另给adapter路径。
2. 实际可运行的embedding推理命令或函数；或者确认上述直接末尾输入token协议。
3. 如果与原生路径不同：实际模板/processor、层与位置或池化方式、是否生成及其参数、新增token和额外保存模块。

本次未取得实验室接收者的checkpoint路径与推理调用。因此运行配置留给接收者填写，不从历史方案PDF写死EOS、生成步骤、输入mask或特殊token。

## 最小扩展位置

对于其他ms-swift embedding注册，先在 `preflight.model_contract` 和 `encode.SwiftEncoder` 接入实际的模型注册/模板，复用其官方推理响应；记录表征协议后测试。

如果实际模型必须生成后取表征，现有的可选 `encoder.backend=plugin` 只提供一个本地Python接入点：文件导出 `create_encoder(config)`，返回带 `report:dict` 和 `encode(branches)` 的对象；后者返回单位向量矩阵及每条实际token数量。`encoder.plugin` 必须是用户明确提供的可信代码，另填 `representation_protocol`。实现者负责把统一输入转换为其真实的image/text推理调用，并在report中写清生成参数、层、pooling和processor。这个接口不附带假定的生成模型实现，也不是实验室标准流程的前置步骤。

原生backend当前要求processor与model同目录；如有独立定制processor，提供可复现接入实现。加载失败会中止，不会自动退回原模型或忽略adapter。

## 最小 GPU 验证

按README生成独立 `--smoke --case-directions B-B --text-modalities B` 配置，batch1，2个query/4个gallery/22类文本。先check，再encode→evaluate→audit。验证向量形状与有限值、单位范数、序列未截断、adapter实际加载和模型报告。确认该模型语义后再扩展完整方向/划分。CPU检查不能替代这一步。
