# Actionable Stabilization Roadmap

This document provides a prioritized roadmap for stabilizing the ProSense project. The tasks are ordered by priority, addressing the most critical issues first.

## Phase 1: Foundational Fixes (Critical Priority)

*These tasks must be completed to make the project usable and maintainable.*

1.  **Generate `requirements.txt` File:**
    - **Action:** Create a `requirements.txt` file listing all third-party dependencies (`pandas`, `mne`, `numpy`, `scipy`, `matplotlib`). Pin the versions to ensure a stable, reproducible environment.
    - **Justification:** This is the highest priority task. Without it, no one can reliably install or run the project.

2.  **Externalize Configuration:**
    - **Action:** Remove the hardcoded data path from `main.py`. Modify the script to accept the input path as a command-line argument (e.g., using Python's `argparse` module).
    - **Justification:** This makes the tool portable and usable with different datasets, which is essential for its primary function.

3.  **Update `README.md`:**
    - **Action:** Correct the repository name and remove the incorrect reference to `StreamSense`. Ensure the setup and execution instructions are accurate and reflect the changes from the two points above.
    - **Justification:** Provides accurate documentation for new users.

## Phase 2: Improving Robustness (High Priority)

*These tasks focus on ensuring the code is correct and preventing future regressions.*

1.  **Develop a Basic Test Suite:**
    - **Action:** Create a `tests/` directory. Add initial unit tests for at least one feature extraction function (e.g., `test_extract_power_band_ratios` in `tests/test_featex_eeg.py`). The test should use a small, sample data file and assert that the output has the correct shape and type.
    - **Justification:** Introduces a testing culture and provides a safety net for future changes. Verifies the scientific correctness of the core algorithms.

2.  **Activate Incomplete Features:**
    - **Action:** Un-comment the calls to `apply_baseline_correction()` and `apply_rejection()` in `main.py` to make them part of the default pipeline.
    - **Justification:** Completes the intended feature set of the application. If these features are optional, this should be controlled by a command-line flag.

## Phase 3: Code Quality and Refactoring (Medium Priority)

*These tasks focus on improving the long-term maintainability of the codebase.*

1.  **Refactor for Code Reuse:**
    - **Action:** Create a `BaseProcessor` class that contains the duplicated logic from the `process_*` functions in `main.py`. Refactor the main loop to use this base class.
    - **Justification:** Reduces code duplication, making the code easier to read and maintain.

2.  **Separate Concerns in Feature Extractors:**
    - **Action:** Move all plotting functions from the `FeatEx*` classes into a separate `visualization.py` module. The `FeatEx*` classes should only be responsible for extracting features and returning data.
    - **Justification:** Improves modularity and adheres to the Single Responsibility Principle, making the code easier to test and understand.

3.  **Remove Dead Code:**
    - **Action:** Delete the `manual_component_selection()` method from `prepro_eeg.py` if it is confirmed to be unused.
    - **Justification:** Reduces code clutter and developer confusion.
