# LaTeX Diagram Extractor

Convert figures from a LaTeX project into **SVGs**—great for architecture/network diagrams in CS papers.

**What it does**  
1) **Flattens** your TeX project by inlining `\input`, `\include`, `\subfile`, `\import`, and `\subimport`.  
2) **Finds figure graphics** from `\includegraphics{...}` inside `figure` environments.  
3) **Filters** figures if **either the caption OR the source filename** contains diagram-related **keywords** (configurable). Use `--all-figures` to export everything.  
4) **Converts to SVG**: PDFs via **PyMuPDF** (vector when possible), PNG/JPG embedded in an SVG wrapper, and native SVG passthrough.  
5) **Names outputs** after the **source graphic’s filename** (stem), **not** the caption/label:  
   - `model_architecture.pdf` → `model_architecture.svg` + `model_architecture.txt`  
   - Caption text is stored in the **sidecar** `.txt`.  
6) **Single output folder**: all SVG/TXT files go into the same `--out-dir` (no per-TeX subfolders).

---

## Requirements

- Python 3.9+ (recommended)
- PyMuPDF
  ```bash
  pip install pymupdf
  ```

---

## Installation

Place `flatten_and_extract_diagrams.py` anywhere and run with Python, or add it to your PATH.

---

## Quick Start

### Process a single `.tex` file
```bash
python flatten_and_extract_diagrams.py main.tex --out-dir out_svgs
```

### Process a directory (non-recursive)
```bash
python flatten_and_extract_diagrams.py ./paper --out-dir out_svgs
```

### Process a directory (recursive)
```bash
python flatten_and_extract_diagrams.py ./paper --recursive --out-dir out_svgs
```

### Export **all** figures (ignore keywords)
```bash
python flatten_and_extract_diagrams.py main.tex --all-figures --out-dir out_svgs
```

### Add/override keywords
```bash
python flatten_and_extract_diagrams.py main.tex --keywords architecture transformer pipeline "block diagram"
```

### Save flattened TeX (for debugging)
```bash
# Single file → specific path
python flatten_and_extract_diagrams.py main.tex --flatten-out flattened.tex

# Directory mode → treat as a folder; each file writes <stem>_flattened.tex inside
python flatten_and_extract_diagrams.py ./paper --flatten-out ./flattened
```

---

## CLI Options (summary)

- `tex` — A **.tex file** or a **directory** containing `.tex` files.
- `--out-dir` — Output directory for **all** SVG/TXT files (single folder).
- `--flatten-out` — Save flattened TeX (file in single mode; directory in batch mode).
- `--all-figures` — Export every figure (skip keyword filtering).
- `--keywords ...` — Extra keywords for matching **caption** or **source filename**.
- `--recursive` — When input is a directory, also search subdirectories for `.tex`.
- `--dry-run` — Show what would be exported; don’t write files.
- `--verbose` — More logging to stderr.

---

## How It Works

1. **Flattening**  
   Recursively replaces include commands with actual file contents, strips comments, and avoids cycles. Produces one flattened TeX blob for scanning.

2. **Figure & Graphic Resolution**  
   - Finds `\begin{figure}`…`\end{figure}` blocks.  
   - Inside each block, collects `\includegraphics[opts]{path}` items.  
   - Resolves paths using any discovered `\graphicspath{...}` plus the including file’s directory. Tries common extensions (`.pdf`, `.png`, `.jpg`, `.jpeg`, `.svg`) if missing.  
   - Honors `page=N` from `\includegraphics[page=N]{...}`.

3. **Filtering**  
   A graphic is selected if **caption contains a keyword** **OR filename contains a keyword**. Defaults include:  
   `architecture, network, pipeline, framework, block diagram, model overview, diagram, flowchart, transformer, u-net, attention, backbone, schematic, ...`  
   Use `--all-figures` to bypass filtering.

4. **Output Naming**  
   For each selected graphic:  
   - **SVG name**: `<source-stem>.svg` (e.g., `model_architecture.svg`).  
   - **Caption**: `<source-stem>.txt` with the figure’s caption text.  
   - Name collisions get `-2`, `-3`, … appended.  
   - All outputs go into `--out-dir`.

5. **Conversion**  
   - **PDF → SVG** via PyMuPDF’s `page.get_svg_image()` (retains vectors when possible).  
   - **PNG/JPG → SVG** by embedding the raster as base64 within an SVG wrapper.  
   - **SVG → SVG** copied through unchanged.

---

## Example Output Layout

```
out_svgs/
├── model_architecture.svg
├── model_architecture.txt      # caption text
├── backbone-pipeline.svg
├── backbone-pipeline.txt
└── attention-ablations.svg
    attention-ablations.txt
```

---

## Troubleshooting

- **No outputs produced**
  - Try `--all-figures` to test the pipeline.
  - Add more `--keywords` or adjust phrasing (matching is substring, case-insensitive).
  - Use `--flatten-out` and inspect the flattened TeX for `figure` blocks.
  - Use `--verbose` to see path resolutions and skips.

- **“Permission denied” while reading**
  - Ensure the project is in a user-writable location (avoid protected/system folders).
  - The script uses a safe read fallback and will log unreadable paths with `--verbose`.

- **Graphics not found**
  - Confirm `\graphicspath{...}` points to your image folders.
  - Macro-generated paths (e.g., `\imgdir/model.pdf`) aren’t expanded; replace with plain paths if needed.

- **EPS or other formats**
  - Convert to PDF first (e.g., `epstopdf`) or extend the converter to support more types.

- **Unexpected appearance**
  - PyMuPDF is generally faithful; verify complex TikZ/embedded fonts manually.

---

## Limitations

- Macro expansion for dynamic paths is not implemented.  
- “Diagram-ness” is heuristic; refine with `--keywords` or bypass via `--all-figures`.  
- PNG/JPG remain raster inside SVG (not vectorized).  
- Only one page is exported unless `page=N` is supplied in `\includegraphics[...]`.

---

## Tips

- Keep diagrams as single-page PDFs when possible.
- Centralize image directories using `\graphicspath{...}` in the preamble.
- Version-control the generated SVGs and captions alongside your paper if they’re part of a pipeline.

---

**Happy extracting!**
