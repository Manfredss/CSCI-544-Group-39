# Extract Text with LLM

[English](README.md) | [中文](README_CN.md)

This module uses Large Language Models (LLM) to extract context information related to figures from academic papers and generate summaries.

## Features

1. **Intelligent Context Extraction**: Uses LLM to intelligently extract figure-related paragraphs from the entire paper, rather than simple adjacent paragraph matching
2. **Automatic Summary Generation**: Based on extracted context, uses LLM to generate concise summaries
3. **Dual Mode Support**:
   - **Ollama Local Mode** (default): Uses locally deployed Ollama models, no API costs
   - **OpenAI API Mode**: Uses OpenAI's GPT models, higher quality but requires payment

## Installation

```bash
pip install pyyaml requests
```

**Required Dependencies**:
- `pyyaml`: Parse YAML configuration files
- `requests`: HTTP requests (Ollama and OpenAI API)

## Configuration

Configuration file: `config.yaml`

### Basic Configuration

```yaml
# LLM provider selection: "ollama" or "openai"
llm_provider: ollama
```

### Ollama Configuration (Local Mode)

```yaml
ollama:
  base_url: http://localhost:11434  # Ollama service address
  model: qwen3:8b                    # Model name to use
  temperature: 0.7                   # Generation temperature (0-1)
  timeout: 120                       # Request timeout (seconds)
  no_thinking: true                  # Qwen3 model: Disable thinking chain output
```

**Configuration Details**:
- `no_thinking`: Only effective for Qwen3 series models. When set to `true`, the following measures are taken to disable thinking chain output:
  1. Add system message explicitly requesting direct answers
  2. Add explicit instructions before the prompt
  3. Post-process to remove `<think>...</think>` tags and content
  
  Default is `true`. This parameter has no effect for non-Qwen3 models.

**Prerequisites**:
1. Install Ollama: https://ollama.ai/
2. Start Ollama service
3. Pull model: `ollama pull qwen3:8b` (or other models like `llama3.2:3b`, `mistral:7b`, etc.)

### OpenAI Configuration

```yaml
openai:
  api_key: ENV:OPENAI_API_KEY       # API Key, supports environment variables
  base_url: https://api.openai.com/v1
  model: gpt-4o-mini                 # Model name
  temperature: 0.7
  timeout: 60
```

**API Key Configuration**:

**Environment Variable**: `api_key: ENV:OPENAI_API_KEY`
   - Set environment variable: `export OPENAI_API_KEY=sk-xxx...` (Linux/Mac)
   - Windows: `set OPENAI_API_KEY=sk-xxx...`

**DO NOT hardcode the API key directly!!!**

### Prompt Configuration

You can customize prompts in `config.yaml`:

```yaml
prompts:
  extract_context: |  # Context extraction prompt
    You are analyzing...
  generate_summary: |  # Summary generation prompt
    Based on...
```

### Processing Configuration

```yaml
processing:
  max_context_chars: 50000  # Maximum characters to input to LLM
  max_summary_length: 120   # Maximum summary length
  enable_cache: true        # Whether to enable caching (reserved)
```

## Usage

### Basic Usage (Using Ollama Local Model)

```bash
# 1. Ensure Ollama service is running
ollama serve

# 2. Run extraction script
python "Data Acquisition/Extract Text/extract_text_llm.py"
```

### Using OpenAI API

```bash
# 1. Modify config.yaml, set llm_provider to "openai"
# 2. Configure API Key (environment variable or direct input)
export OPENAI_API_KEY=sk-xxx...

# 3. Run script
python "Data Acquisition/Extract Text/extract_text_llm.py"
```

### Custom Parameters

```bash
python "Data Acquisition/Extract Text/extract_text_llm.py" \
  --svg-root "Data Acquisition/Extract Diagram/extracted_svg" \
  --papers-root "Data Acquisition/Crawl Paper/all_conference_papers" \
  --out-root "Data Acquisition/Extract Text/extracted_text" \
  --config "Data Acquisition/Extract Text/config.yaml"
```

## Output Format

Each figure generates a JSON file: `<basename>.json`

### JSON Structure

```json
{
  "summary": "This figure demonstrates a multimodal encoder architecture based on Transformer, which includes a dual-branch feature fusion module for both visual and text.",
  "caption": "Architecture of our proposed model...",
  "label": "fig:architecture",
  "extracted_context": "In Section 3.2, we introduce our novel architecture...\nThe model consists of three main components...\nAs shown in Figure 1, the encoder processes...",
  "extraction_method": "LLM-based"
}
```

### Field Descriptions

- **summary**: One-sentence summary (max 120 characters)
- **caption**: Figure caption (read from SVG sidecar file)
- **label**: LaTeX label (e.g., `fig:architecture`)
- **extracted_context**: Relevant context paragraphs extracted by LLM
- **extraction_method**: Extraction method identifier (`LLM-based` or `rule-based`)

## Recommended Models

### Ollama Local Models
- **Lightweight**: `llama3.2:3b` (2GB) - Fast, suitable for testing
- **Balanced**: `qwen3:8b` (4.7GB) - Balance performance and quality, recommended
- **High Quality**: `qwen3:14b` (9GB) - Highest quality (requires better hardware)

### OpenAI Models
- **Economical**: `gpt-4o-mini` - Cost-effective, recommended
- **High Quality**: `gpt-4o` - Best quality

## Performance Comparison

| Mode | Advantages | Disadvantages | Cost |
|------|------------|---------------|------|
| Ollama | Free, private, offline capable | Requires local hardware, quality lower than GPT-4 | 0 |
| OpenAI | High quality, fast | Requires payment, network dependent | ~$0.0001/call |

## Troubleshooting

### Ollama Connection Failed
```
[error] Ollama request failed: ...
```
**Solution**:
1. Ensure Ollama service is running: `ollama serve`
2. Check if port is correct: default 11434
3. Pull model: `ollama pull qwen3:8b`

### OpenAI API Error
```
[error] OpenAI request failed: 401 Unauthorized
```
**Solution**:
1. Check if API Key is correct
2. Verify environment variable is set
3. Check account balance

### Empty Generated Content
- Check if paper text was successfully extracted
- Try lowering the `temperature` parameter
- Check if prompts are appropriate

### Qwen3 Model Outputs Thinking Content
```
[error] Output contains <think>...</think> tags or thinking chain content
```
**Solution**:
1. Set `no_thinking: true` in `config.yaml` (enabled by default)
2. The script will automatically:
   - Add system message requesting direct answers
   - Explicitly instruct in the prompt not to output thinking process
   - Post-process to remove `<think>` tags and content
3. If the problem persists:
   - Check if Ollama version is the latest
   - Try adjusting the `temperature` parameter (reduce to 0.3-0.5)
   - Verify you're using the correct Qwen3 model

## Advanced Usage

### Batch Process Specific Conference

Modify the script or manually select directory:
```python
# Process only CVPR papers
for paper_dir in sorted((svg_root / "cvpr").iterdir()):
    ...
```

### Customize Prompts

Edit the `prompts` section in `config.yaml` to:
- Request specific output formats
- Specify focus areas (e.g., methods, experiments, results)
- Adjust summary style (technical, popular)

### Use Other OpenAI-Compatible APIs

Modify `base_url` to use other compatible services (e.g., Azure OpenAI, locally deployed vLLM, etc.):
```yaml
openai:
  base_url: https://your-service.com/v1
  api_key: your-key
  # ...
```

## Notes

1. **LLM Call Time**: Each figure requires 2 LLM calls (extract context + generate summary), processing large volumes may take time
2. **API Costs**: Be mindful of call volume when using OpenAI
3. **Text Length**: Very long papers will be truncated to 50000 characters, adjustable in configuration
4. **Network Stability**: Stable network connection required when using OpenAI

## Comparison with Original Version

| Feature | Original (extract_text.py) | LLM Version (extract_text_llm.py) |
|---------|---------------------------|-----------------------------------|
| Context Extraction | Fixed adjacent paragraphs | LLM intelligent extraction of relevant paragraphs |
| Summary Quality | Keyword template | LLM-generated semantic summary |
| Accuracy | Medium | High |
| Speed | Fast | Slower |
| Cost | 0 | Ollama: 0 / OpenAI: Low |
| Dependencies | None | pyyaml, requests, Ollama/OpenAI |

## Example Output Comparison

### Original Version (extract_text.py)

**Output File**: `teaser2.json`

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

### LLM Version (extract_text_llm.py)

**Output File**: `teaser2.json`

```json
{
  "summary": "Memory-based adapters enable offline 3D models to handle online perception tasks efficiently and effectively.",
  "caption": "We propose a general framework for online 3D scene perception...",
  "label": "teaser",
  "extracted_context": "### Summary of Key Contributions and Findings\n\n**Problem Addressed**:  \nThe paper tackles the challenge of transitioning **offline 3D scene perception models** (which process single static views) to **online settings** (handling sequential, dynamic data from sensors like RGB-D cameras)...",
  "extraction_method": "LLM-based"
}
```
