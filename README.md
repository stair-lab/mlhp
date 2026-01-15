# Machine Learning from Human Preferences
### Sang Truong, Andreas Haupt, and Sanmi Koyejo

This book is available online at: [ai.stanford.edu/~sttruong/mlhp](https://ai.stanford.edu/~sttruong/mlhp)

## Prerequisites

Before building the book, ensure you have the following installed:

- **Quarto** >= 1.5.56 (CI uses this version)
- **Python** >= 3.8
- **R** >= 4.3.1

## Installation

### 1. Install Quarto

- **macOS (Homebrew):** `brew install quarto`
- **macOS/Windows:** Download from https://quarto.org/docs/get-started/

### 2. Install Python

- **macOS:** `brew install python`
- **Windows:** Download from https://www.python.org/downloads/

### 3. Install R

- **macOS:** `brew install r`
- **Windows:** Download from https://cran.r-project.org/

### 4. Clone the repository

```bash
git clone https://github.com/sangttruong/mlhp
cd mlhp
```

### 5. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 6. Install R dependencies (one-time setup)

```bash
R -e "renv::restore()"
```

Or interactively:
```bash
R
> renv::restore()
> q()
```

## Building the Book

### Preview (development)

To preview the book locally with live reload:

```bash
quarto preview
```

Then open http://localhost:4200/ in your browser.

### Build HTML

```bash
quarto render --to html --profile html
```

### Build PDF

```bash
quarto render --to pdf --profile pdf
```

### Build both HTML and PDF

```bash
quarto render --to pdf --profile pdf
quarto render --to html --profile html --no-clean
```

The built website will be stored in the `_book/` folder.

## Publishing

After completing edits and building:

```bash
git add .
git commit -m "your commit message"
git push origin main
```

The GitHub Actions workflow will automatically build and publish to GitHub Pages.

## Troubleshooting

- **Quarto version issues:** Ensure you have Quarto >= 1.5.56. Check with `quarto --version`.
- **R package issues:** Run `renv::restore()` in R to reinstall dependencies.
- **Python package issues:** Run `pip install -r requirements.txt` to reinstall dependencies.
- **LaTeX/PDF issues:** The CI uses TinyTeX. Install with `quarto install tinytex`.

## Reproducibility

This book is designed to be reproducible. Key version recommendations:

- **Python:** 3.11 (used in CI)
- **R:** 4.3.1 (locked in renv.lock)
- **Quarto:** 1.5.56

### Clearing caches

If you upgrade dependencies or encounter stale results, clear the caches:

```bash
rm -rf _freeze/ src/.jupyter_cache/ _book/
quarto render --to html --profile html
```

## License

This book is licensed [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
