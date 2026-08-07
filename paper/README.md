# SERVIS paper

This directory is the publication workspace, kept separate from the runtime
code and local research-paper library.

## Layout

- `main.tex` — manuscript source.
- `references.bib` — BibTeX database.
- `figures/` — selected publication figures and their provenance notes.

Build from the repository root with:

```bash
make -C paper
```

The manuscript deliberately marks unfilled quantitative claims with `TODO`.
Replace those markers only after the final benchmark tables have been generated
from `RUNS/`; do not infer missing numbers from illustrative plots.
