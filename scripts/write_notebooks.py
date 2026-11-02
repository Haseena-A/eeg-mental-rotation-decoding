"""Write the eight analysis notebooks. They call the library and do not embed results."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

NOTEBOOKS = {
    "01_dataset_exploration.ipynb": (
        "Dataset exploration",
        "Inventory the local archive. This notebook does not assume a column that the loader did not find.",
        "from eeg_mental_rotation.config import load_config, resolve\n"
        "from eeg_mental_rotation.data_loader import find_dataset_root, inspect_dataset\n"
        "config = load_config()\n"
        "root = find_dataset_root(resolve(config, 'paths.raw_dir'))\n"
        "print(root)\n"
        "inspect_dataset(root) if root else 'No local dataset. Run scripts/download_data.py.'\n",
    ),
    "02_eeg_signal_analysis.ipynb": (
        "EEG signal analysis",
        "Estimate the sampling rate from timestamps before any filter is applied. "
        "Prefer unix_time when that column exists. An irregular grid is not filtered.",
        "from eeg_mental_rotation.data_validation import estimate_sampling_rate, parse_timestamps\n"
        "print('Use the inventory notebook to load one EEG frame, then:')\n"
        "print('estimate_sampling_rate(parse_timestamps(eeg[\"timestamp\"]))')\n",
    ),
    "03_feature_extraction.ipynb": (
        "Feature extraction",
        "Vendor band power and recomputed Welch power use different prefixes.",
        "from eeg_mental_rotation.feature_extraction import features_for_window\n"
        "print(features_for_window.__doc__)\n",
    ),
    "04_eeg_behavior_integration.ipynb": (
        "EEG and behaviour",
        "Trial alignment is accepted only when the clocks overlap.",
        "from eeg_mental_rotation.synchronization import assign_level\n"
        "print(assign_level.__doc__)\n",
    ),
    "05_statistical_analysis.ipynb": (
        "Statistics",
        "Tests are computed on participant means. See docs/methodology.md.",
        "from eeg_mental_rotation.statistical_analysis import compare_conditions\n"
        "print(compare_conditions.__doc__)\n",
    ),
    "06_model_training.ipynb": (
        "Model training",
        "GroupKFold by participant. Correctness is not a feature.",
        "from eeg_mental_rotation.model_training import eeg_feature_columns\n"
        "print(eeg_feature_columns.__doc__)\n",
    ),
    "07_model_evaluation.ipynb": (
        "Evaluation",
        "Read results/tables/model_results.json after the pipeline has run. Do not type metrics in.",
        "import json\n"
        "from pathlib import Path\n"
        "path = Path('../results/tables/model_results.json')\n"
        "print(json.loads(path.read_text()) if path.exists() else 'Not yet computed.')\n",
    ),
    "08_model_interpretation.ipynb": (
        "Interpretation",
        "Permutation importance is a property of the model, not a cortical mechanism.",
        "from eeg_mental_rotation.interpretation import permutation_importance\n"
        "print(permutation_importance.__doc__)\n",
    ),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (title, body, code) in NOTEBOOKS.items():
        notebook = new_notebook()
        notebook.cells = [new_markdown_cell(f"# {title}\n\n{body}"), new_code_cell(code)]
        notebook.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        path = OUT / name
        path.write_text(nbformat.writes(notebook), encoding="utf-8")
        print(path.name)


if __name__ == "__main__":
    main()
