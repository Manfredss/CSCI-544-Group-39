#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract related text paragraphs for each extracted diagram caption and generate a one-sentence insight.

功能概述：
- 读取 Data Acquisition/Extract Diagram/extracted_svg 下的各会议/论文子目录内的 *.txt（caption 侧车文件）
- 在 Data Acquisition/Crawl Paper/all_conference_papers 下找到对应论文的 TeX 工程，自动展开 \input/\include/\subfile/\import/\subimport，形成扁平文本
- 在扁平文本中按 caption 相似度匹配对应的 figure 环境，抽取：
  - 该图环境前后的相邻自然段（各一段）
  - 若该图包含 \label{...}，额外抽取全局文本中包含 \ref{label} 的段落
- 生成一句话长度的精简观点（中文模板，基于 caption 关键词压缩）
- 输出到 Data Acquisition/Extract Text/{conference}/{paper}/ 下：
  - <basename>_context.txt ：包含 caption、label、邻近段落与引用段落
  - <basename>_summary.txt ：一句话观点

仅使用 Python 标准库，无需新增第三方依赖。
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

def find_paragraph_bounds(text: str, idx: int) -> Tuple[int,int]:
    # Find start of current paragraph (after previous blank line)
    start = text.rfind("\n\n", 0, idx)
    start = 0 if start == -1 else start + 2
    # Find end of paragraph (before next blank line)
    end = text.find("\n\n", idx)
    end = len(text) if end == -1 else end
    return start, end

def adjacent_paragraphs(text: str, start: int, end: int) -> Tuple[str,str]:
    # Previous paragraph
    prev_break = text.rfind("\n\n", 0, start)
    prev_start = 0 if prev_break == -1 else prev_break + 2
    prev_end = start
    prev_para = text[prev_start:prev_end].strip()

    # Next paragraph
    next_break = text.find("\n\n", end)
    next_end_break = text.find("\n\n", next_break + 2) if next_break != -1 else -1
    next_start = end if next_break == -1 else next_break + 2
    next_end = len(text) if next_end_break == -1 else next_end_break
    next_para = text[next_start:next_end].strip()

    return prev_para, next_para

def paragraphs_with_label_refs(text: str, label: Optional[str]) -> List[str]:
    if not label:
        return []
    refs = []
    for m in re.finditer(re.escape(f"\\ref{{{label}}}"), text):
        s, e = find_paragraph_bounds(text, m.start())
        para = text[s:e].strip()
        if para and para not in refs:
            refs.append(para)
    return refs

def summarize_caption(caption: str, max_words: int = 18, max_chars: int = 120) -> str:
    # 简易摘要：抽取关键词（按长度/频次粗略），中文模板输出
    base = re.sub(r'\s+', ' ', caption.strip())
    tokens = re.findall(r'[A-Za-z0-9\-]+|[\u4e00-\u9fa5]+', base)
    # 过滤常见停用词（英语）
    stop = set("""
        the a an of for to in on with and or by from under over between among
        figure fig we our this that these those method model framework approach
        result results experiment experiments dataset datasets
    """.split())
    key = []
    freq = {}
    for t in tokens:
        tl = t.lower()
        if tl in stop or len(tl) <= 2:
            continue
        freq[tl] = freq.get(tl, 0) + 1
    # 依据频次与长度排序
    ranked = sorted(freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    for w, _ in ranked[:max_words]:
        key.append(w)
    phrase = "、".join(key) if any('\u4e00' <= ch <= '\u9fff' for ch in "".join(tokens)) else ", ".join(key)
    summary = f"This diagram shows:{phrase}" if phrase else "This diagram shows: the core structure and key processes."
    return summary[:max_chars]

def write_context_and_summary(out_dir: Path, base_name: str, caption: str,
                              label: Optional[str], prev_para: str,
                              next_para: str, ref_paras: List[str], summary: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_path = out_dir / f"{base_name}_context.txt"
    sum_path = out_dir / f"{base_name}_summary.txt"

    ctx = {
        "caption": caption,
        "label": label or "",
        "prev_paragraph": prev_para,
        "next_paragraph": next_para,
        "ref_paragraphs": ref_paras
    }
    # 写 context（人类可读 + JSON）
    ctx_human = []
    ctx_human.append(f"[Caption]\n{caption}\n")
    ctx_human.append(f"[Label]\n{label or ''}\n")
    ctx_human.append("[Prev Paragraph]\n" + (prev_para or "") + "\n")
    ctx_human.append("[Next Paragraph]\n" + (next_para or "") + "\n")
    if ref_paras:
        ctx_human.append("[Ref Paragraphs]\n" + "\n\n".join(ref_paras) + "\n")
    ctx_human.append("[JSON]\n" + json.dumps(ctx, ensure_ascii=False, indent=2) + "\n")

    ctx_path.write_text("\n".join(ctx_human), encoding="utf-8")
    sum_path.write_text(summary.strip(), encoding="utf-8")

def process_paper(svg_paper_dir: Path, papers_root: Path, out_paper_dir: Path) -> int:
    # 寻找对应论文工程目录
    # svg_paper_dir 形如 .../extracted_svg/{conference}/{paper}
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
        matched = match_figure_by_caption(fig_blocks, caption)
        if not matched:
            # 未匹配到具体图环境时，仍生成摘要但无上下文
            summary = summarize_caption(caption)
            write_context_and_summary(out_paper_dir, base_name, caption, None, "", "", [], summary)
            print(f"[warn] No figure match for {conference}/{paper_code}/{base_name}", file=sys.stderr)
            count += 1
            continue

        s, e, block = matched
        label = extract_label(block)
        prev_para, next_para = adjacent_paragraphs(flat_text, s, e)
        ref_paras = paragraphs_with_label_refs(flat_text, label)

        summary = summarize_caption(caption)
        write_context_and_summary(out_paper_dir, base_name, caption, label, prev_para, next_para, ref_paras, summary)
        count += 1
    
    return count

def main():
    ap = argparse.ArgumentParser(description="Extract related text for diagram captions and generate one-sentence insights.")
    ap.add_argument("--svg-root", type=str, default=str(Path("Data Acquisition/Extract Diagram/extracted_svg")),
                    help="Root of extracted_svg (caption sidecar location)")
    ap.add_argument("--papers-root", type=str, default=str(Path("Data Acquisition/Crawl Paper/all_conference_papers")),
                    help="Root of all_conference_papers (TeX projects)")
    ap.add_argument("--out-root", type=str, default=str(Path("Data Acquisition/Extract Text")),
                    help="Output root for context and summary")
    args = ap.parse_args()

    svg_root = Path(args.svg_root).resolve()
    papers_root = Path(args.papers_root).resolve()
    out_root = Path(args.out_root).resolve()

    if not svg_root.exists():
        print(f"Error: svg_root not found: {svg_root}", file=sys.stderr)
        sys.exit(1)
    if not papers_root.exists():
        print(f"Error: papers_root not found: {papers_root}", file=sys.stderr)
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
                count = process_paper(paper_dir, papers_root, out_paper_dir)
                if count > 0:
                    print(f"[{conference_dir.name}/{paper_dir.name}] Processed {count} caption(s)")
                total_count += count
            except Exception as ex:
                print(f"[error] {conference_dir.name}/{paper_dir.name}: {ex}", file=sys.stderr)

    print(f"\nDone. Total processed: {total_count} caption(s).")

if __name__ == "__main__":
    main()
