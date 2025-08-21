# Technical Debt Report

This report summarizes the key areas of technical debt identified during the audit of the ProSense codebase.

## 1. Critical Issues (High Priority)

### 1.1. Missing Dependency Management
- **Observation**: The `README.md` instructs users to install dependencies from a `requirements.txt` file, but this file does not exist in the repository.
- **Impact**: This makes the project impossible to set up reliably, as dependencies and their versions are unknown. It is the single largest barrier to reproducibility and collaboration.
- **Risk**: High. New developers cannot run the project. The original environment may be lost, making the code unusable.

### 1.2. Hardcoded Configuration
- **Observation**: The input data path (`D:/Study Data/...`) is hardcoded in `main.py`.
- **Impact**: The application is not portable and cannot be run on any machine or with any other dataset without modifying the source code.
- **Risk**: High. This severely limits the tool's utility and makes it brittle.

## 2. Major Issues (Medium Priority)

### 2.1. No Automated Testing
- **Observation**: There are zero test files (`test_*.py`, etc.) in the repository. The project has 0% test coverage.
- **Impact**: The correctness of the complex data processing and feature extraction algorithms cannot be verified automatically. Any refactoring or addition of new features is extremely risky and may introduce silent bugs.
- **Risk**: High. The scientific validity of the results produced by this tool is unverified.

### 2.2. Significant Code Duplication
- **Observation**:
    - The `process_*` functions in `main.py` for each modality are nearly identical in their structure (instantiate preprocessor, run steps, instantiate feature extractor, save results).
    - The plotting code within each `featex_*.py` class is highly repetitive.
- **Impact**: This makes the code harder to maintain. A bug fix or change in logic needs to be applied in many different places.
- **Risk**: Medium. Leads to code bloat and a high likelihood of introducing inconsistencies.

## 3. Minor Issues (Low Priority)

### 3.1. Single Responsibility Principle Violation
- **Observation**: The `FeatEx*` classes are responsible for both feature extraction and generating complex visualizations (plots).
- **Impact**: This makes the classes large and complex. The core logic is tightly coupled with the presentation logic, making it harder to test or reuse the feature extraction logic independently.
- **Risk**: Medium. Hinders maintainability and testability.

### 3.2. Unused / Unreachable Code
- **Observation**: Several methods, such as `apply_baseline_correction` and `apply_rejection` in `prepro_eeg.py`, are implemented but the calls are commented out in `main.py`.
- **Impact**: This "dead code" can confuse new developers about the intended workflow and adds unnecessary complexity to the codebase.
- **Risk**: Low. The code is not actively causing bugs, but it represents an incomplete feature implementation and adds clutter.

### 3.3. Inconsistent Documentation
- **Observation**: The `README.md` refers to a repository named `StreamSense` and mentions a `requirements.txt` file that does not exist.
- **Impact**: This can cause confusion for new users and suggests a lack of care in maintaining the project.
- **Risk**: Low. A minor issue, but it affects the project's professionalism and ease of use.
