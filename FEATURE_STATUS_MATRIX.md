# Feature Status Matrix

This document classifies the status of each identified feature and component using the project's standard ontology.

| Feature/Component | Status | Justification |
| :--- | :--- | :--- |
| **Overall Project** | `Incomplete/Partial` | The project is a functional set of scripts but is not a distributable or easily reproducible tool. It lacks dependency management and relies on hardcoded paths. |
| | | |
| **Core Logic** | | |
| Data Loading & Prep | `Needs Improvement (Technical Debt)` | The logic in `load_data.py` and `prepro_data.py` is functional, but the hardcoded paths in `main.py` are a major issue for portability. |
| Modality Orchestration | `Complete` | `main.py` successfully orchestrates the processing for all data modalities as designed in the architecture diagram. |
| | | |
| **EEG Modality** | | |
| EEG Preprocessing | `Incomplete/Partial` | The `PreProEEG` class implements all necessary functions, but crucial steps like `apply_baseline_correction` and `apply_rejection` are commented out in `main.py`, meaning the default pipeline is not using the full capabilities of the code. |
| `apply_baseline_correction()` | `Orphaned/Dead Code` | The method is implemented in `prepro_eeg.py` but its call is commented out in `main.py`. |
| `apply_rejection()` | `Orphaned/Dead Code` | The method is implemented in `prepro_eeg.py` but its call is commented out in `main.py`. |
| EEG Feature Extraction | `Complete` | `featex_eeg.py` correctly implements and exposes all the features described in the documentation (power bands, entropy, etc.). |
| | | |
| **Other Modalities (PPG, ACC, etc.)** | | |
| Preprocessing & Feature Extraction | `Complete` | Based on the file structure and the pattern established by the EEG module, these components appear to be fully implemented as per the modular design. |
| | | |
| **Supporting Components** | | |
| Visualization | `Complete` | Each `featex_*.py` module contains extensive plotting functions that are used in `main.py` to generate and save `.png` files of the results. |
| Dependency Management | `Broken (Needs Fixes)` | The `README.md` explicitly mentions a `requirements.txt` file for installation, but this file is missing from the repository. This is a critical failure for reproducibility. |
| Configuration | `Broken (Needs Fixes)` | All configuration, especially the input data paths, is hardcoded into `main.py`. This makes the application unusable in any other environment without modifying the source code. |
