#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flatten a LaTeX project (inline \\input/\\include/\\subfile/\\import/\\subimport),
scan figure environments, and convert included graphics to SVGs using PyMuPDF.

Naming policy (per user request):
- SVG files are named by the figure's \\label (sanitized).
- A sidecar TXT file with the SAME basename holds the figure caption.
- If multiple \\includegraphics are inside one figure, they become:
    <label>.svg, <label>-2.svg, <label>-3.svg, ...
- If a figure has no \\label, we fall back to a sanitized slug of its caption,
  otherwise to "figure". Names are made unique within the single output folder.

Batching:
- The first CLI arg may be a single .tex file OR a directory.
- If it's a directory, processes all *.tex in that directory (use --recursive for subfolders).
- All outputs go into ONE folder given by --out-dir (no per-tex subfolders).

Dependencies:
    pip install pymupdf
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import sys
from pathlib import Path
from typing import Iterator, List, Tuple, Optional

try:
    import fitz  # PyMuPDF
except Exception as e:
    print("ERROR: PyMuPDF (pymupdf) is required. Install via: pip install pymupdf", file=sys.stderr)
    raise

# ---------- Regexes ----------

COMMENT_RE = re.compile(r'(?m)(?<!\\)%.*$')  # strip comments (not escaped \%)

# \input{file}, \include{file}, \subfile{file}
INPUT_LIKE_RE  = re.compile(r'\\(?:input|include|subfile)\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}')
# \import{dir}{file}, \subimport{dir}{file}
IMPORT_LIKE_RE = re.compile(r'\\(?:import|subimport)\s*\{([^}]+)\}\s*\{([^}]+)\}')

# \graphicspath{{dir/}{dir2/}...}
GRAPHICSPATH_RE = re.compile(
    r'\\graphicspath\s*\{(?P<paths>(?:\{[^{}]*\}\s*)+)\}',
    flags=re.IGNORECASE
)

# \includegraphics[opts]{path}
INCLUDEGRAPHICS_RE = re.compile(
    r'\\includegraphics\s*(?:\[(?P<opts>[^\]]*)\])?\s*\{(?P<path>[^}]+)\}',
    flags=re.IGNORECASE
)

# Figure env anchors
BEGIN_FIG_RE = re.compile(r'\\begin\{figure\*?\}', re.IGNORECASE)
END_FIG_RE   = re.compile(r'\\end\{figure\*?\}', re.IGNORECASE)

# Caption start (we will extract with balanced braces)
CAPTION_RE = re.compile(r'\\caption(?:\[[^\]]*\])?\s*\{', re.IGNORECASE)

# \label{...} inside figure
LABEL_RE = re.compile(r'\\label\s*\{([^}]+)\}', re.IGNORECASE)


# ---------- Utilities ----------

def log(msg: str, verbose: bool):
    if verbose:
        print(msg, file=sys.stderr)


def strip_comments(s: str) -> str:
    return COMMENT_RE.sub('', s)


def read_text_safe(path: Path, verbose: bool=False) -> str:
    """Read text safely; on PermissionError or issues, return ''."""
    try:
        if not path.exists() or not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except PermissionError:
        if verbose:
            print(f"[perm] Permission denied: {path}", file=sys.stderr)
        try:
            with open(path, "rb") as f:
                return f.read().decode("utf-8", "ignore")
        except Exception:
            return ""
    except OSError as e:
        if verbose:
            print(f"[oserror] {path} → {e}", file=sys.stderr)
        return ""


def _resolve_tex_like(pathish: str, base_dir: Path) -> Optional[Path]:
    """Resolve a TeX path (possibly lacking suffix) relative to base_dir."""
    p = Path(pathish)
    if p.is_absolute():
        cand = p if p.suffix else p.with_suffix('.tex')
        return cand if cand.exists() else None
    q = base_dir / p
    if q.suffix == '' and (q.with_suffix('.tex')).exists():
        return (q.with_suffix('.tex')).resolve()
    if q.exists():
        return q.resolve()
    if (q.with_suffix('.tex')).exists():
        return (q.with_suffix('.tex')).resolve()
    return None


def flatten_tex(entry: Path, verbose: bool=False, max_files: int=1000) -> Tuple[str, List[Path]]:
    """
    Recursively inline \input/\include/\subfile/\import/\subimport.
    Returns (flattened_text, file_order) where file_order is the DFS visitation order.
    """
    seen: set[Path] = set()
    order: List[Path] = []

    def _inline_for_file(p: Path) -> str:
        if p in seen:
            log(f"[flatten] Skipping already-seen: {p}", verbose)
            return ""
        if len(seen) >= max_files:
            log("[flatten] Reached max_files limit; stopping traversal.", verbose)
            return ""
        if not p.exists():
            log(f"[flatten] Missing file: {p}", verbose)
            return ""
        seen.add(p)
        order.append(p)
        base = p.parent
        text = read_text_safe(p, verbose=verbose)
        if not text:
            log(f"[flatten] Unreadable or empty: {p}", verbose)
            return ""
        text = strip_comments(text)

        # Inline \input/\include/\subfile
        def repl_input(m: re.Match) -> str:
            rel = m.group(1).strip()
            target = _resolve_tex_like(rel, base)
            if target is None:
                log(f"[flatten] Could not resolve: {rel} from {p}", verbose)
                return ""
            log(f"[flatten] Inlining {target}", verbose)
            return _inline_for_file(target)

        text = INPUT_LIKE_RE.sub(repl_input, text)

        # Inline \import/\subimport
        def repl_import(m: re.Match) -> str:
            d = m.group(1).strip()
            f = m.group(2).strip()
            target = _resolve_tex_like((Path(d) / Path(f)).as_posix(), base)
            if target is None:
                alt = (base / d / f)
                if alt.suffix == '':
                    alt = alt.with_suffix('.tex')
                if alt.exists():
                    target = alt.resolve()
            if target is None:
                log(f"[flatten] Could not resolve import: {d} + {f} from {p}", verbose)
                return ""
            log(f"[flatten] Inlining {target}", verbose)
            return _inline_for_file(target)

        text = IMPORT_LIKE_RE.sub(repl_import, text)
        return text

    flat = _inline_for_file(entry.resolve())
    return flat, order


def parse_graphicspaths(tex_text: str, base_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for m in GRAPHICSPATH_RE.finditer(tex_text):
        inner = m.group('paths')
        for sub in re.findall(r'\{([^{}]*)\}', inner):
            s = sub.strip()
            if not s:
                continue
            candidate = (base_dir / s).resolve()
            paths.append(candidate)
    # De-dup in order
    dedup: List[Path] = []
    seen = set()
    for p in paths:
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    return dedup


def balanced_extract(text: str, start_idx: int, open_char='{', close_char='}') -> Tuple[str, int]:
    """
    Extract content inside balanced braces starting at start_idx which is the index
    of the opening brace. Returns (content_without_braces, end_idx_after_closing).
    Raises ValueError if not balanced.
    """
    if text[start_idx] != open_char:
        raise ValueError("balanced_extract must start at opening brace")
    depth = 1
    i = start_idx + 1
    content_chars: List[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == open_char:
            depth += 1
            content_chars.append(ch)
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                i += 1
                break
            content_chars.append(ch)
        else:
            content_chars.append(ch)
        i += 1
    if depth != 0:
        raise ValueError("Unbalanced braces in caption or argument")
    return "".join(content_chars), i


def iter_figure_blocks(tex_text: str) -> Iterator[Tuple[int, int, str]]:
    """
    Yield (start_idx, end_idx, block_text) for each figure environment in order.
    """
    text = tex_text
    pos = 0
    while True:
        m = BEGIN_FIG_RE.search(text, pos)
        if not m:
            return
        start = m.start()
        m_end = END_FIG_RE.search(text, m.end())
        if not m_end:
            return
        end = m_end.end()
        yield (start, end, text[start:end])
        pos = end


def extract_caption(fig_block: str) -> Optional[str]:
    m = CAPTION_RE.search(fig_block)
    if not m:
        return None
    brace_idx = m.end() - 1  # at '{'
    try:
        content, _ = balanced_extract(fig_block, brace_idx)
        # Remove simple TeX commands for keyword matching / saving
        cleaned = re.sub(r'\\[a-zA-Z@]+(\s*\{[^{}]*\})?', '', content)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    except ValueError:
        return None


def extract_label(fig_block: str) -> Optional[str]:
    """Return the first \\label{...} inside the figure block, if any."""
    m = LABEL_RE.search(fig_block)
    if not m:
        return None
    return m.group(1).strip()


def find_includegraphics_in_block(fig_block: str) -> List[Tuple[str, Optional[str]]]:
    """Return list of (path, opts) within the figure block."""
    items: List[Tuple[str, Optional[str]]] = []
    for m in INCLUDEGRAPHICS_RE.finditer(fig_block):
        path = m.group('path').strip()
        opts = m.group('opts')
        items.append((path, opts))
    return items


def parse_page_from_opts(opts: Optional[str]) -> Optional[int]:
    if not opts:
        return None
    m = re.search(r'page\s*=\s*([0-9]+)', opts)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def resolve_graphic(name: str, search_dirs: List[Path], base_dir: Path,
                    exts=('.pdf', '.png', '.jpg', '.jpeg', '.svg')) -> Optional[Path]:
    """Resolve includegraphics path according to graphicspath + file dir + base dir."""
    path = Path(name)
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        for d in search_dirs + [base_dir]:
            candidates.append((d / path))
    tries: List[Path] = []
    if path.suffix:
        tries = [c.resolve() for c in candidates]
    else:
        for c in candidates:
            for ext in exts:
                tries.append((c.with_suffix(ext)).resolve())
    for t in tries:
        if t.exists():
            return t
    return None


def slugify(s: str, maxlen: int=80) -> str:
    """Sanitize for filename: lower, keep [a-z0-9-_], collapse dashes."""
    s = s.strip().lower()
    s = re.sub(r'[^a-z0-9\-_]+', '-', s)
    s = re.sub(r'-{2,}', '-', s).strip('-')
    return s[:maxlen] if len(s) > maxlen else s


def sanitize_label(label: str) -> str:
    """Sanitize \\label content for a cross-platform filename."""
    return slugify(label)


def pdf_to_svg(pdf_path: Path, svg_path: Path, page: int=1, zoom: float=1.0):
    doc = fitz.open(pdf_path.as_posix())
    try:
        idx = max(0, min(page - 1, len(doc) - 1))
        pg = doc[idx]
        mat = fitz.Matrix(zoom, zoom)
        svg = pg.get_svg_image(matrix=mat)
        svg_path.write_text(svg, encoding='utf-8')
    finally:
        doc.close()


def raster_to_svg(image_path: Path, svg_path: Path):
    """Wrap a PNG/JPEG inside an SVG as an embedded image (base64)."""
    data = image_path.read_bytes()
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(data).decode('ascii')
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 800 600">\n'
        f'  <image x="0" y="0" width="800" height="600" '
        f'preserveAspectRatio="xMidYMid meet" xlink:href="data:{mime};base64,{b64}"/>\n'
        '</svg>\n'
    )
    svg_path.write_text(svg, encoding='utf-8')


# DEFAULT_DIAGRAM_KEYWORDS = [
#     "architecture", "network", "pipeline", "framework", "block diagram",
#     "model overview", "overview", "diagram", "flowchart", "flow chart",
#     "graph", "graphical model", "transformer", "cnn", "rnn", "encoder-decoder",
#     "encoder–decoder", "module", "backbone", "feature pyramid", "resnet",
#     "u-net", "unet", "attention", "multi-head", "head", "layer", "ablation",
#     "system", "design", "schematic", "overview of the model",
# ]

DEFAULT_DIAGRAM_KEYWORDS = [
    "architecture", "network", "pipeline", "framework", "block diagram",
    "overview", "diagram", "flowchart", "flow chart",
    "graphical mode", "transformer", "cnn", "rnn", "encoder-decoder",
    "encoder–decoder", "module", "feature pyramid", "resnet",
    "u-net", "unet", "attention", "multi-head", "head", "layer",
    "system", "design", "schematic",
]

def contains_keywords(s: Optional[str], keywords: List[str]) -> bool:
    if not s:
        return False
    low = s.lower()
    return any(k in low for k in keywords)

def is_diagram_caption(caption: Optional[str], keywords: List[str]) -> bool:
    if not caption:
        return False
    cap = caption.lower()
    return any(k in cap for k in keywords)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Flatten TeX and extract diagram-like figures to SVG (named by \\label).")
    ap.add_argument("tex", type=str,
                    help="Path to a main .tex file OR a directory containing .tex files")
    ap.add_argument("--out-dir", type=str, default="out_diagrams",
                    help="Output directory for all SVG/TXT files (single folder)")
    ap.add_argument("--flatten-out", type=str, default=None,
                    help="If single-file: path to write flattened TeX;"
                         " if batching: treated as a directory, each file writes <stem>_flattened.tex")
    ap.add_argument("--all-figures", action="store_true",
                    help="Extract all figures (ignore caption keywords)")
    ap.add_argument("--keywords", nargs="*", default=[],
                    help="Extra caption keywords to treat as diagram")
    ap.add_argument("--dry-run", action="store_true", help="Show actions but do not write files")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--recursive", action="store_true",
                    help="When input is a directory, recurse into subdirectories for *.tex")
    args = ap.parse_args()

    inp = Path(args.tex).resolve()
    if not inp.exists():
        print(f"Error: {inp} not found.", file=sys.stderr)
        sys.exit(1)

    # Choose targets
    if inp.is_dir():
        targets = sorted(inp.rglob("*.tex") if args.recursive else inp.glob("*.tex"))
        if not targets:
            print(f"No .tex files found in {inp}", file=sys.stderr)
            sys.exit(1)
    else:
        if inp.suffix.lower() != ".tex":
            print("Error: input must be a .tex file or a directory.", file=sys.stderr)
            sys.exit(1)
        targets = [inp]

    # Flatten-out handling
    flatten_out_dir: Optional[Path] = None
    flatten_out_file: Optional[Path] = None
    if args.flatten_out:
        fo = Path(args.flatten_out)
        if len(targets) == 1 and (fo.suffix.lower() == ".tex" or not fo.exists()):
            flatten_out_file = fo  # explicit file path for single target
        else:
            flatten_out_dir = fo   # directory for batch mode
            if not args.dry_run:
                flatten_out_dir.mkdir(parents=True, exist_ok=True)

    out_root = Path(args.out_dir).resolve()
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    # Keep global uniqueness across all processed files
    used_bases: set[str] = set()

    total_exported = 0

    for entry in targets:
        if args.verbose:
            print(f"[process] {entry}", file=sys.stderr)

        # 1) Flatten this entry
        flat_text, order = flatten_tex(entry, verbose=args.verbose)

        # 2) Write flattened TeX
        if args.flatten_out:
            if flatten_out_file and len(targets) == 1:
                out_flat = flatten_out_file
            else:
                base_dir = flatten_out_dir if flatten_out_dir else out_root
                out_flat = base_dir / f"{entry.stem}_flattened.tex"
            if not args.dry_run:
                out_flat.write_text(flat_text, encoding="utf-8")
            if args.verbose:
                print(f"[write] Flattened TeX -> {out_flat}", file=sys.stderr)

        # 3) Gather \graphicspath from visited files (earlier declarations first)
        search_dirs: List[Path] = []
        seen_dirs = set()
        for p in order:
            local_text = read_text_safe(p, verbose=args.verbose)
            if not local_text:
                continue
            local_dirs = parse_graphicspaths(local_text, p.parent)
            for d in local_dirs:
                if d not in seen_dirs:
                    search_dirs.append(d)
                    seen_dirs.add(d)
        base_dir = entry.parent
        if base_dir not in search_dirs:
            search_dirs.append(base_dir)

        # 4) Figure extraction loop (all outputs into out_root)
        keywords = DEFAULT_DIAGRAM_KEYWORDS + [k.lower() for k in args.keywords]
        exported_here = 0

        for (_s, _e, fig_block) in iter_figure_blocks(flat_text):
            caption = extract_caption(fig_block)
            include_items = find_includegraphics_in_block(fig_block)
            if not include_items:
                continue

            for (path_str, opts) in include_items:
                # Resolve the graphic path first
                if '\\' in path_str or '#' in path_str:
                    if args.verbose:
                        print(f"[skip] Macro/special path: {path_str}", file=sys.stderr)
                    continue

                page_opt = parse_page_from_opts(opts)
                resolved = resolve_graphic(path_str, search_dirs, base_dir=base_dir)
                if not resolved:
                    if args.verbose:
                        print(f"[warn] Could not resolve graphic: {path_str}", file=sys.stderr)
                    continue

                # Eligibility: caption OR filename contains any keyword (unless --all-figures)
                if not args.all_figures:
                    fname_for_match = resolved.name  # includes extension
                    if not (contains_keywords(caption, keywords) or
                            contains_keywords(fname_for_match, keywords)):
                        continue

                # Naming: base name from the SOURCE file name (stem)
                base_name = slugify(resolved.stem) or "figure"

                # Ensure global uniqueness across the entire run
                unique_base = base_name
                if (unique_base in used_bases or
                    (out_root / f"{unique_base}.svg").exists() or
                    (out_root / f"{unique_base}.txt").exists()):
                    i = 2
                    while True:
                        cand = f"{base_name}-{i}"
                        if (cand not in used_bases and
                            not (out_root / f"{cand}.svg").exists() and
                            not (out_root / f"{cand}.txt").exists()):
                            unique_base = cand
                            break
                        i += 1
                used_bases.add(unique_base)

                # Write caption sidecar for THIS graphic
                if not args.dry_run:
                    cap_txt_path = out_root / f"{unique_base}.txt"
                    cap_txt_path.write_text((caption or "").strip(), encoding="utf-8")

                # Convert to SVG
                out_svg = out_root / f"{unique_base}.svg"
                if not args.dry_run:
                    try:
                        ext = resolved.suffix.lower()
                        if ext == ".pdf":
                            pdf_to_svg(resolved, out_svg, page=page_opt or 1, zoom=1.0)
                        elif ext in (".png", ".jpg", ".jpeg"):
                            raster_to_svg(resolved, out_svg)
                        elif ext == ".svg":
                            out_svg.write_text(resolved.read_text(encoding="utf-8", errors="ignore"),
                                               encoding="utf-8")
                        else:
                            if args.verbose:
                                print(f"[skip] Unsupported format: {resolved}", file=sys.stderr)
                            # remove the reserved base if we skipped output
                            used_bases.discard(unique_base)
                            continue
                    except Exception as ex:
                        print(f"[error] Failed to convert {resolved}: {ex}", file=sys.stderr)
                        # free the reserved base on failure
                        used_bases.discard(unique_base)
                        continue

                exported_here += 1

        print(f"{entry.name}: exported {exported_here} item(s) to {out_root}")
        total_exported += exported_here

    if len(targets) > 1:
        print(f"All done. Total exported: {total_exported} figure(s) across {len(targets)} .tex file(s).")


if __name__ == "__main__":
    main()
