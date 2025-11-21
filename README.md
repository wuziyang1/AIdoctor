# AIdoctor

AIdoctor 是一个智能医疗客服项目，可通过微信公众号提供医疗相关的咨询服务，依托医疗疾病数据为用户解答常见疾病相关问题。

## 项目结构

```
AIdoctor/
├── .gitignore          # 忽略不需要纳入版本控制的文件
├── wr.py               # 微信公众号服务相关代码
└── doctor_offline/
    ├── data/
    │   └── unstructured/
    │       └── norecognite/  # 存储各类疾病的文本信息
    └── review_model/
        └── model/
            └── bert-base-chinese/  # BERT基础中文模型（未纳入版本控制）
```

## 功能说明

1. **微信公众号集成**：通过 `wr.py` 实现与微信公众号的对接，接收用户消息并进行响应。
2. **首次交互欢迎**：用户首次关注公众号时，会收到欢迎信息“您好, 我是智能客服小艾, 有什么需要帮忙的吗?”。
3. **医疗咨询服务**：非首次交互时，将用户输入的文本发送至主逻辑服务接口，获取并返回相应的医疗咨询结果。
4. **疾病数据支持**：`doctor_offline/data/unstructured/norecognite/` 目录下存储了多种疾病的相关信息，包括疾病症状、临床表现等，为医疗咨询提供数据支持。

## 环境依赖

- Python 3.x
- `werobot` 库：用于处理微信公众号消息
- `requests` 库：用于发送 HTTP 请求

可通过以下命令安装依赖：
```bash
pip install werobot requests
```

## 配置与运行

1. **配置公众号信息**：在 `wr.py` 中，将 `token` 设置为你微信公众号的令牌。
2. **设置主逻辑服务地址**：修改 `url` 为实际的主逻辑服务接口地址。
3. **运行服务**：执行 `wr.py` 启动微信公众号服务，服务将运行在 80 端口。
```bash
python wr.py
```

## 注意事项

- `.gitignore` 中指定了忽略 `.cursor`、`.idea`、`.vscode` 等IDE配置文件以及 `bert-base-chinese/` 模型目录，避免这些文件纳入版本控制。
- 主逻辑服务接口需单独部署，确保 `wr.py` 中配置的 `url` 可正常访问。
- 疾病数据存储在 `doctor_offline/data/unstructured/norecognite/` 目录下，可根据实际需求补充或更新相关疾病信息。

## 疾病数据说明

`doctor_offline/data/unstructured/norecognite/` 目录下包含多种疾病的文本信息，每个文件对应一种疾病，内容涵盖疾病的症状、临床表现、所属科室等信息，为智能客服的医疗咨询提供数据支撑。例如：
- 附件炎：介绍了急性和慢性附件炎的症状表现。
- 血友病甲：描述了临床特点、出血与损伤的关系以及分型标准等。
