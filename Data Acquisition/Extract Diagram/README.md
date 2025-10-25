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
