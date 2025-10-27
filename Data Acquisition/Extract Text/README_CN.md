# Extract Text with LLM

[English](README.md) | [中文](README_CN.md)

本模块使用大语言模型（LLM）从学术论文中提取与图表相关的上下文信息并生成摘要。

## 功能特性

1. **智能上下文提取**：使用 LLM 从整篇论文中智能提取与图表相关的段落，而非简单的前后段落匹配
2. **自动摘要生成**：基于提取的上下文，使用 LLM 生成简明的中文摘要
3. **双模式支持**：
   - **Ollama 本地模式**（默认）：使用本地部署的 Ollama 模型，无需 API 费用
   - **OpenAI API 模式**：使用 OpenAI 的 GPT 模型，质量更高但需付费

## 安装依赖

```bash
pip install pyyaml requests
```

**必须依赖**：
- `pyyaml`: 解析 YAML 配置文件
- `requests`: HTTP 请求（Ollama 和 OpenAI API）

## 配置说明

配置文件：`config.yaml`

### 基本配置

```yaml
# LLM 提供商选择: "ollama" 或 "openai"
llm_provider: ollama
```

### Ollama 配置（本地模式）

```yaml
ollama:
  base_url: http://localhost:11434  # Ollama 服务地址
  model: qwen3:8b                  # 使用的模型名称
  temperature: 0.7                   # 生成温度 (0-1)
  timeout: 120                       # 请求超时时间（秒）
  no_thinking: true                  # Qwen3 模型：禁用思维链输出
```

**配置说明**：
- `no_thinking`: 仅对 Qwen3 系列模型生效。设置为 `true` 时，会采取以下措施禁用思维链输出：
  1. 添加 system 消息明确要求直接回答
  2. 在 prompt 前添加明确指令
  3. 后处理移除 `<think>...</think>` 标签和内容
  
  默认为 `true`。对于非 Qwen3 模型，此参数无效。

**使用前需要：**
1. 安装 Ollama：https://ollama.ai/
2. 启动 Ollama 服务
3. 拉取模型：`ollama pull qwen3:8b`（或其他模型如 `llama3.2:3b`、`mistral:7b` 等）

### OpenAI 配置

```yaml
openai:
  api_key: ENV:OPENAI_API_KEY       # API Key，支持环境变量
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini                 # 模型名称
  temperature: 0.7
  timeout: 60
```

**API Key 配置方式：**

**环境变量**：`api_key: ENV:OPENAI_API_KEY`
   - 设置环境变量：`export OPENAI_API_KEY=sk-xxx...`（Linux/Mac）
   - Windows：`set OPENAI_API_KEY=sk-xxx...`

**请勿直接填写！！！**

### 提示词配置

可以在 `config.yaml` 中自定义提示词：

```yaml
prompts:
  extract_context: |  # 上下文提取提示词
    You are analyzing...
  generate_summary: |  # 摘要生成提示词
    Based on...
```

### 处理配置

```yaml
processing:
  max_context_chars: 50000  # 输入给 LLM 的最大字符数
  max_summary_length: 120   # 摘要最大长度
  enable_cache: true        # 是否启用缓存（预留）
```

## 使用方法

### 基本用法（使用 Ollama 本地模型）

```bash
# 1. 确保 Ollama 服务运行中
ollama serve

# 2. 运行提取脚本
python "Data Acquisition/Extract Text/extract_text_llm.py"
```

### 使用 OpenAI API

```bash
# 1. 修改 config.yaml，设置 llm_provider 为 "openai"
# 2. 配置 API Key（环境变量或直接填写）
export OPENAI_API_KEY=sk-xxx...

# 3. 运行脚本
python "Data Acquisition/Extract Text/extract_text_llm.py"
```

### 自定义参数

```bash
python "Data Acquisition/Extract Text/extract_text_llm.py" \
  --svg-root "Data Acquisition/Extract Diagram/extracted_svg" \
  --papers-root "Data Acquisition/Crawl Paper/all_conference_papers" \
  --out-root "Data Acquisition/Extract Text/extracted_text" \
  --config "Data Acquisition/Extract Text/config.yaml"
```

## 输出格式

每个图表生成一个 JSON 文件：`<basename>.json`

### JSON 结构

```json
{
  "summary": "This figure demonstrates a multimodal encoder architecture based on Transformer, which includes a dual-branch feature fusion module for both visual and text.",
  "caption": "Architecture of our proposed model...",
  "label": "fig:architecture",
  "extracted_context": "In Section 3.2, we introduce our novel architecture...\nThe model consists of three main components...\nAs shown in Figure 1, the encoder processes...",
  "extraction_method": "LLM-based"
}
```

### 字段说明

- **summary**: 一句话摘要（最多 120 字符）
- **caption**: 图表标题（从 SVG 侧车文件读取）
- **label**: LaTeX 标签（如 `fig:architecture`）
- **extracted_context**: LLM 提取的相关上下文段落
- **extraction_method**: 提取方法标识（`LLM-based` 或 `rule-based`）

## 推荐模型

### Ollama 本地模型
- **轻量级**：`llama3.2:3b` (2GB) - 快速，适合测试
- **平衡型**：`qwen3:8b` (4.7GB) - 平衡性能和质量，推荐
- **高质量**：`qwen3:14b` (9GB) - 质量最高（需要较好硬件）

### OpenAI 模型
- **经济型**：`gpt-4o-mini` - 性价比高，推荐
- **高质量**：`gpt-4o` - 最佳质量

## 性能对比

| 模式 | 优点 | 缺点 | 成本 |
|------|------|------|------|
| Ollama | 免费、私密、可离线 | 需要本地硬件、质量较 GPT-4 低 | 0 |
| OpenAI | 质量高、速度快 | 需要付费、依赖网络 | ~$0.0001/次 |

## 故障排除

### Ollama 连接失败
```
[error] Ollama request failed: ...
```
**解决方案**：
1. 确保 Ollama 服务运行：`ollama serve`
2. 检查端口是否正确：默认 11434
3. 拉取模型：`ollama pull qwen3:8b`

### OpenAI API 错误
```
[error] OpenAI request failed: 401 Unauthorized
```
**解决方案**：
1. 检查 API Key 是否正确
2. 确认环境变量已设置
3. 检查账户余额

### 生成内容为空
- 检查论文文本是否成功提取
- 尝试降低 `temperature` 参数
- 检查提示词是否合适

### Qwen3 模型输出 thinking 内容
```
[error] 输出包含 <think>...</think> 标签或思维链内容
```
**解决方案**：
1. 在 `config.yaml` 中设置 `no_thinking: true`（默认已启用）
2. 脚本会自动：
   - 添加 system 消息要求直接回答
   - 在 prompt 中明确指示不输出思考过程
   - 后处理移除 `<think>` 标签及内容
3. 如果仍然出现问题，请：
   - 检查 Ollama 版本是否为最新
   - 尝试调整 `temperature` 参数（降低到 0.3-0.5）
   - 检查是否使用了正确的 Qwen3 模型

## 高级用法

### 批量处理特定会议

修改脚本或手动选择目录：
```python
# 只处理 CVPR 论文
for paper_dir in sorted((svg_root / "cvpr").iterdir()):
    ...
```

### 自定义提示词

编辑 `config.yaml` 中的 `prompts` 部分，可以：
- 要求特定格式的输出
- 指定关注点（如方法、实验、结果）
- 调整摘要风格（技术性、通俗性）

### 使用其他 OpenAI 兼容 API

修改 `base_url` 即可使用其他兼容服务（如 Azure OpenAI、本地部署的 vLLM 等）：
```
openai:
  base_url: https://your-service.com/v1
  api_key: your-key
  # ...
  "base_url": "https://your-service.com/v1",
  "api_key": "your-key",
  ...

```

## 注意事项

1. **LLM 调用时间**：每个图表需要 2 次 LLM 调用（提取上下文 + 生成摘要），处理大量论文可能耗时较长
2. **API 费用**：使用 OpenAI 时注意控制调用量
3. **文本长度**：非常长的论文会被截断到 50000 字符，可在配置中调整
4. **网络稳定性**：使用 OpenAI 时需要稳定的网络连接

## 与原版对比

| 特性 | 原版 (extract_text.py) | LLM 版 (extract_text_llm.py) |
|------|----------------------|---------------------------|
| 上下文提取 | 固定前后段落 | LLM 智能提取相关段落 |
| 摘要质量 | 关键词模板 | LLM 生成语义摘要 |
| 准确性 | 中等 | 高 |
| 速度 | 快 | 较慢 |
| 成本 | 0 | Ollama: 0 / OpenAI: 低 |
| 依赖 | 无 | pyyaml, requests, Ollama/OpenAI |

## 示例输出对比

### 原版 (extract_text.py)

**输出文件**: `teaser2.json`

```json
{
  "summary": "This figure shows: perception, online, memory-based, applications, presented, different, adapters",
  "caption": "We propose a general framework for online 3D scene perception...",
  "label": "teaser",
  "prev_paragraph": "\\twocolumn[{\n\\maketitle\n\\vspace{-8mm}",
  "next_paragraph": "\\begin{abstract}\n  In this paper, we propose...",
  "ref_paragraphs": [],
  "extraction_method": "rule-based"
}
```

### LLM 版 (extract_text_llm.py)

**输出文件**: `teaser2.json`

```json
{
  "summary": "Memory-based adapters enable offline 3D models to handle online perception tasks efficiently and effectively.",
  "caption": "We propose a general framework for online 3D scene perception...",
  "label": "teaser",
  "extracted_context": "### Summary of Key Contributions and Findings\n\n**Problem Addressed**:  \nThe paper tackles the challenge of transitioning **offline 3D scene perception models** (which process single static views) to **online settings** (handling sequential, dynamic data from sensors like RGB-D cameras)...",
  "extraction_method": "LLM-based"
}
```
