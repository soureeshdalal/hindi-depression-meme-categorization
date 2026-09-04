# Category distribution figures (ACL paper)

Source: `gemini_categorization/analysis_output/gemini_analysis_results.json` (N=1,021), consolidated to 13 categories per `01_documentation/Category_definitions.md`.

## Top 5 categories

| Rank | Category | Count | % |
|------|----------|------:|--:|
| 1 | Universal Human | 827 | 81.0% |
| 2 | Language | 41 | 4.0% |
| 3 | Pop Culture | 31 | 3.0% |
| 4 | Cultural Reference | 25 | 2.4% |
| 5 | Brand | 24 | 2.4% |

Top 5 combined: **948 / 1,021 (92.8%)**

## Files

| File | Description |
|------|-------------|
| `category_distribution.csv` | Full counts + test/val/train split |
| `category_distribution_pie.pdf` / `.png` | All 13 categories (legend with counts) |
| `category_distribution_bar.pdf` / `.png` | Horizontal bar chart (single-column friendly) |
| `category_top5_pie.pdf` / `.png` | Top 5 + Other |
| `category_distribution_table.tex` | ACL `booktabs` table |
| `category_distribution_figure.tex` | `\includegraphics` snippets |

## LaTeX usage

```latex
\usepackage{booktabs}
\input{paper_figures/category_distribution_table.tex}
\includegraphics[width=\linewidth]{paper_figures/category_distribution_pie.pdf}
```
