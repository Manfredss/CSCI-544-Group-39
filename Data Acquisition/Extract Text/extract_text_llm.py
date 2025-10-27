#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract related text paragraphs for each extracted diagram caption using LLM and generate summaries.

功能概述：
- 读取 Data Acquisition/Extract Diagram/extracted_svg 下的各会议/论文子目录内的 *.txt（caption 侧车文件）
- 在 Data Acquisition/Crawl Paper/all_conference_papers 下找到对应论文的 TeX 工程，自动展开并形成扁平文本
- 使用 LLM (Ollama 本地模型或 OpenAI API) 从整篇论文中提取与图表相关的上下文段落
- 使用 LLM 基于提取的上下文生成简明的中文摘要
- 输出到 Data Acquisition/Extract Text/{conference}/{paper}/ 下

支持两种 LLM 模式：
1. Ollama 本地模型（默认）- 需要本地运行 Ollama 服务
2. OpenAI API - 需要配置 API Key
"""

from __future__ import annotations
import argparse
import os
import re
import sys
import json
import difflib
from pathlib import Path
from typing import Any, List, Tuple, Optional
import time

# ---------- Regexes ----------
COMMENT_RE = re.compile(r'(?m)(?<!\\)%.*$')  # strip comments
BEGIN_FIG_RE = re.compile(r'\\begin\{figure\*?\}', re.IGNORECASE)
END_FIG_RE   = re.compile(r'\\end\{figure\*?\}', re.IGNORECASE)
CAPTION_RE   = re.compile(r'\\caption(?:\[[^\]]*\])?\s*\{', re.IGNORECASE)
LABEL_RE     = re.compile(r'\\label\s*\{([^}]+)\}', re.IGNORECASE)

# \input{file}, \include{file}, \subfile{file}
INPUT_LIKE_RE  = re.compile(r'\\(?:input|include|subfile)\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}')
# \import{dir}{file}, \subimport{dir}{file}
IMPORT_LIKE_RE = re.compile(r'\\(?:import|subimport)\s*\{([^}]+)\}\s*\{([^}]+)\}')

def strip_comments(s: str) -> str:
    return COMMENT_RE.sub('', s)

def read_text_safe(path: Path) -> str:
    try:
        if not path.exists() or not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return path.read_bytes().decode("utf-8", "ignore")
        except Exception:
            return ""

def balanced_extract(text: str, start_idx: int, open_char='{', close_char='}') -> Tuple[str, int]:
    if start_idx >= len(text) or text[start_idx] != open_char:
        raise ValueError("balanced_extract must start at opening brace")
    depth = 1
    i = start_idx + 1
    out: List[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == open_char:
            depth += 1
            out.append(ch)
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                i += 1
                break
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    if depth != 0:
        raise ValueError("Unbalanced braces")
    return "".join(out), i

def iter_figure_blocks(tex_text: str) -> List[Tuple[int, int, str]]:
    res: List[Tuple[int, int, str]] = []
    pos = 0
    while True:
        m = BEGIN_FIG_RE.search(tex_text, pos)
        if not m:
            break
        endm = END_FIG_RE.search(tex_text, m.end())
        if not endm:
            break
        start, end = m.start(), endm.end()
        res.append((start, end, tex_text[start:end]))
        pos = end
    return res

def extract_caption(fig_block: str) -> Optional[str]:
    m = CAPTION_RE.search(fig_block)
    if not m:
        return None
    brace_idx = m.end() - 1
    try:
        content, _ = balanced_extract(fig_block, brace_idx)
        cleaned = re.sub(r'\\[a-zA-Z@]+(\s*\{[^{}]*\})?', '', content)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    except ValueError:
        return None

def extract_label(fig_block: str) -> Optional[str]:
    m = LABEL_RE.search(fig_block)
    return m.group(1).strip() if m else None

def _resolve_tex_like(pathish: str, base_dir: Path) -> Optional[Path]:
    p = Path(pathish)
    if p.is_absolute():
        cand = p if p.suffix else p.with_suffix(".tex")
        return cand if cand.exists() else None
    q = base_dir / p
    if q.exists():
        return q.resolve()
    if q.suffix == '' and (q.with_suffix('.tex')).exists():
        return (q.with_suffix('.tex')).resolve()
    return None

def flatten_tex(entry: Path, max_files: int = 2000) -> Tuple[str, List[Path]]:
    seen: set[Path] = set()
    order: List[Path] = []

    def _inline_for_file(p: Path) -> str:
        if p in seen or len(seen) >= max_files or not p.exists():
            return ""
        seen.add(p)
        order.append(p)
        base = p.parent
        text = read_text_safe(p)
        if not text:
            return ""
        text = strip_comments(text)

        def repl_input(m: re.Match) -> str:
            rel = m.group(1).strip()
            target = _resolve_tex_like(rel, base)
            return _inline_for_file(target) if target else ""

        def repl_import(m: re.Match) -> str:
            d: Any = m.group(1).strip()
            f = m.group(2).strip()
            target = _resolve_tex_like((Path(d) / Path(f)).as_posix(), base)
            if not target:
                alt = base / d / f
                if alt.suffix == '':
                    alt = alt.with_suffix('.tex')
                if alt.exists():
                    target = alt.resolve()
            return _inline_for_file(target) if target else ""

        text = INPUT_LIKE_RE.sub(repl_input, text)
        text = IMPORT_LIKE_RE.sub(repl_import, text)
        return text

    flat = _inline_for_file(entry.resolve())
    return flat, order

# ---------- LLM Integration ----------

class LLMConfig:
    def __init__(self, config_path: Path):
        # 仅支持 YAML 格式
        try:
            import yaml
        except ImportError:
            print("Error: PyYAML is required. Install: pip install pyyaml", file=sys.stderr)
            raise
        
        config_text = read_text_safe(config_path)
        self.config = yaml.safe_load(config_text)
        self.provider = self.config.get("llm_provider", "ollama")
        
    def get_api_key(self, key_spec: str) -> Optional[str]:
        """从配置或环境变量中获取 API Key"""
        if key_spec.startswith("ENV:"):
            env_var = key_spec[4:]
            return os.environ.get(env_var)
        return key_spec if key_spec else None

class OllamaClient:
    def __init__(self, config: dict):
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "qwen3:8b")
        self.temperature = config.get("temperature", 0.7)
        self.timeout = config.get("timeout", 120)
        self.no_thinking = config.get("no_thinking", True)  # 默认禁用 thinking
        
    def generate(self, prompt: str) -> str:
        """调用 Ollama API"""
        try:
            import requests
            url = f"{self.base_url}/api/generate"
            
            # 如果是 Qwen3 模型且启用 no_thinking，在 prompt 前添加指令
            if self.no_thinking and "qwen3" in self.model.lower():
                prompt = "/set nothink\n" + prompt
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature
                }
            }
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            raw_response = result.get("response", "").strip()
            
            # 后处理：移除可能残留的 thinking 标签
            if self.no_thinking and "qwen3" in self.model.lower():
                raw_response = self._remove_thinking_tags(raw_response)
            
            return raw_response
        except Exception as e:
            print(f"[error] Ollama request failed: {e}", file=sys.stderr)
            return ""
    
    def _remove_thinking_tags(self, text: str) -> str:
        """移除 Qwen3 可能输出的 thinking 标签和内容"""
        import re
        # 移除 <think>...</think> 标签及其内容
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # 移除可能的其他思维链标记
        text = re.sub(r'\[thinking\].*?\[/thinking\]', '', text, flags=re.DOTALL | re.IGNORECASE)
        # 移除 "Thinking:" 或 "思考：" 开头的段落
        text = re.sub(r'^(Thinking:|思考：).*?(?=\n\n|$)', '', text, flags=re.MULTILINE | re.DOTALL)
        return text.strip()

class OpenAIClient:
    def __init__(self, config: dict, config_obj: LLMConfig):
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 0.7)
        self.timeout = config.get("timeout", 60)
        self.api_key = config_obj.get_api_key(config.get("api_key", ""))
        
    def generate(self, prompt: str) -> str:
        """调用 OpenAI API"""
        try:
            import requests
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature
            }
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[error] OpenAI request failed: {e}", file=sys.stderr)
            return ""

class LLMProcessor:
    def __init__(self, config_path: Path):
        self.config = LLMConfig(config_path)
        self.prompts = self.config.config.get("prompts", {})
        self.processing = self.config.config.get("processing", {})
        
        # 初始化对应的 LLM 客户端
        if self.config.provider == "ollama":
            self.client = OllamaClient(self.config.config.get("ollama", {}))
            print("[info] Using Ollama local model", file=sys.stderr)
        elif self.config.provider == "openai":
            self.client = OpenAIClient(self.config.config.get("openai", {}), self.config)
            print("[info] Using OpenAI API", file=sys.stderr)
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.provider}")
    
    def extract_context(self, full_text: str, caption: str, label: Optional[str]) -> str:
        """使用 LLM 从全文中提取相关上下文"""
        max_chars = self.processing.get("max_context_chars", 50000)
        
        # 截断过长的文本
        if len(full_text) > max_chars:
            # 优先保留前半部分（通常包含方法描述）
            full_text = full_text[:max_chars] + "\n\n[... text truncated ...]"
        
        prompt = self.prompts.get("extract_context", "").format(
            full_text=full_text,
            caption=caption,
            label=label or "N/A"
        )
        
        print(f"  [llm] Extracting context...", file=sys.stderr)
        context = self.client.generate(prompt)
        return context
    
    def generate_summary(self, caption: str, context: str) -> str:
        """使用 LLM 生成简明摘要"""
        max_len = self.processing.get("max_summary_length", 120)
        
        prompt = self.prompts.get("generate_summary", "").format(
            caption=caption,
            context=context[:5000]  # 限制上下文长度
        )
        
        print(f"  [llm] Generating summary...", file=sys.stderr)
        summary = self.client.generate(prompt)
        
        # 确保长度限制
        if len(summary) > max_len:
            summary = summary[:max_len-3] + "..."
        
        return summary

# ---------- Main helpers ----------

CANDIDATE_MAIN_NAMES = [
    "main.tex", "main_arxiv.tex", "neurips_2024.tex",
    "anonymous-submission-latex-2024.tex", "paper.tex", "main_arXiv.tex"
]

def find_main_tex(paper_dir: Path) -> Optional[Path]:
    # Try common names first
    for name in CANDIDATE_MAIN_NAMES:
        p = paper_dir / name
        if p.exists():
            return p.resolve()
    # Fallback: any *.tex containing \begin{document}
    tex_files = sorted(paper_dir.rglob("*.tex"))
    for f in tex_files:
        content = read_text_safe(f)
        if "\\begin{document}" in content:
            return f.resolve()
    return tex_files[0].resolve() if tex_files else None

def normalize_caption(s: str) -> str:
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\\[a-zA-Z@]+(\s*\{[^{}]*\})?', '', s)  # strip LaTeX cmds
    return s.lower()

def match_figure_by_caption(figs: List[Tuple[int,int,str]], target_caption: str) -> Optional[Tuple[int,int,str]]:
    if not target_caption:
        return None
    target_norm = normalize_caption(target_caption)
    best: Tuple[float, Optional[Tuple[int,int,str]]] = (0.0, None)
    for (s,e,block) in figs:
        cap = extract_caption(block) or ""
        cap_norm = normalize_caption(cap)
        ratio = difflib.SequenceMatcher(None, target_norm, cap_norm).ratio()
        if ratio > best[0]:
            best = (ratio, (s,e,block))
    return best[1] if best[0] >= 0.55 else None  # threshold

def write_context_and_summary(out_dir: Path, base_name: str, caption: str,
                              label: Optional[str], context: str, summary: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{base_name}.json"

    # 统一的 JSON 格式
    result = {
        "summary": summary.strip(),
        "caption": caption,
        "label": label or "",
        "extracted_context": context,
        "extraction_method": "LLM-based"
    }
    
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

def process_paper(svg_paper_dir: Path, papers_root: Path, out_paper_dir: Path, 
                 llm_processor: LLMProcessor) -> int:
    # 寻找对应论文工程目录
    conference = svg_paper_dir.parent.name
    paper_code = svg_paper_dir.name
    tex_paper_dir = papers_root / conference / paper_code
    
    if not tex_paper_dir.exists():
        print(f"[skip] TeX paper dir not found: {tex_paper_dir}", file=sys.stderr)
        return 0

    main_tex = find_main_tex(tex_paper_dir)
    if not main_tex:
        print(f"[skip] No main tex found in {tex_paper_dir}", file=sys.stderr)
        return 0

    print(f"[processing] {conference}/{paper_code}", file=sys.stderr)
    flat_text, _order = flatten_tex(main_tex)
    if not flat_text:
        print(f"[skip] Flattened text empty: {main_tex}", file=sys.stderr)
        return 0

    fig_blocks = iter_figure_blocks(flat_text)

    # 遍历该论文的所有 caption 侧车文件（*.txt）
    count = 0
    for cap_file in sorted(svg_paper_dir.glob("*.txt")):
        base_name = cap_file.stem
        caption = read_text_safe(cap_file).strip()
        if not caption:
            continue
        
        print(f"  [figure] {base_name}", file=sys.stderr)
        
        # 尝试匹配图环境以获取 label
        matched = match_figure_by_caption(fig_blocks, caption)
        label = extract_label(matched[2]) if matched else None
        
        # 使用 LLM 提取上下文
        context = llm_processor.extract_context(flat_text, caption, label)
        
        # 使用 LLM 生成摘要
        summary = llm_processor.generate_summary(caption, context)
        
        # 写入文件
        write_context_and_summary(out_paper_dir, base_name, caption, label, context, summary)
        count += 1
        
        # 短暂延迟避免请求过快
        time.sleep(0.5)
    
    return count

def main():
    ap = argparse.ArgumentParser(description="Extract related text for diagram captions using LLM and generate summaries.")
    ap.add_argument("--svg-root", type=str, default=str(Path("Data Acquisition/Extract Diagram/extracted_svg")),
                    help="Root of extracted_svg (caption sidecar location)")
    ap.add_argument("--papers-root", type=str, default=str(Path("Data Acquisition/Crawl Paper/all_conference_papers")),
                    help="Root of all_conference_papers (TeX projects)")
    ap.add_argument("--out-root", type=str, default=str(Path("Data Acquisition/Extract Text/extracted_text")),
                    help="Output root for context and summary")
    ap.add_argument("--config", type=str, default=str(Path("Data Acquisition/Extract Text/config.yaml")),
                    help="Path to config file (YAML or JSON)")
    args = ap.parse_args()

    svg_root = Path(args.svg_root).resolve()
    papers_root = Path(args.papers_root).resolve()
    out_root = Path(args.out_root).resolve()
    config_path = Path(args.config).resolve()

    if not svg_root.exists():
        print(f"Error: svg_root not found: {svg_root}", file=sys.stderr)
        sys.exit(1)
    if not papers_root.exists():
        print(f"Error: papers_root not found: {papers_root}", file=sys.stderr)
        sys.exit(1)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # 初始化 LLM 处理器
    try:
        llm_processor = LLMProcessor(config_path)
    except Exception as e:
        print(f"Error initializing LLM processor: {e}", file=sys.stderr)
        sys.exit(1)

    total_count = 0
    # 遍历会议与论文目录
    for conference_dir in sorted(svg_root.iterdir()):
        if not conference_dir.is_dir():
            continue
        for paper_dir in sorted(conference_dir.iterdir()):
            if not paper_dir.is_dir():
                continue
            out_paper_dir = out_root / conference_dir.name / paper_dir.name
            try:
                count = process_paper(paper_dir, papers_root, out_paper_dir, llm_processor)
                if count > 0:
                    print(f"[{conference_dir.name}/{paper_dir.name}] Processed {count} caption(s)")
                total_count += count
            except Exception as ex:
                print(f"[error] {conference_dir.name}/{paper_dir.name}: {ex}", file=sys.stderr)
                import traceback
                traceback.print_exc()

    print(f"\nDone. Total processed: {total_count} caption(s).")

if __name__ == "__main__":
    main()
