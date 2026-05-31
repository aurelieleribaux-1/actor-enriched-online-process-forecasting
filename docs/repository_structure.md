# Repository design

The repository separates:

- **configuration** (`configs/`) from executable code;
- **archival notebooks** (`notebooks/archive/`) from reproducible scripts;
- **source modules** (`src/`) from generated outputs;
- **interim drafting exports** (`results/interim/`) from final generated results.

This structure lets the paper cite exact final specifications while keeping training, evaluation, significance testing, and visualisation reproducible.
