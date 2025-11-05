# Project Structure Manifest - ProSense
**Audit Date:** November 5, 2025
**Auditor:** Claude Code (Senior Technical Auditor AI)
**Previous Audit:** August 21, 2025 (google-labs-jules[bot])

---

## Executive Summary

**Project Name:** ProSense (formerly referenced as StreamSense)
**Project Type:** Python Data Science / Biometric Signal Processing Research Tool
**Domain:** Psychophysiology (stress analysis, mental workload assessment)
**Primary Language:** Python 3.11+
**Total Files:** 43 (21 Python source files)
**Total Lines of Code:** 6,631 LOC
**Project Age:** 18 months (Created February 2024)
**Last Active Development:** April 2024 (17 months ago)
**Project Status:** STALE, INCOMPLETE, NON-PRODUCTION

---

## Directory Structure

```
ProSense/
├── .git/                      # Version control (146 objects)
├── .idea/                     # IDE configuration (PyCharm/IntelliJ)
│   ├── .gitignore
│   ├── .name
│   ├── ProSense.iml
│   ├── misc.xml
│   ├── modules.xml
│   ├── vcs.xml
│   └── inspectionProfiles/
│       └── profiles_settings.xml
│
├── images/                    # Documentation images (899KB)
│   ├── Slide30.JPG           # System Architecture diagram
│   ├── Slide31.JPG           # Features visualization
│   ├── Slide35.JPG           # Bar plot example
│   ├── Slide36.JPG           # Line plot example
│   ├── Slide37.JPG           # Box plot example (phased)
│   ├── Slide39.JPG
│   ├── Slide40.JPG           # Box plot example (events)
│   └── Slide41.JPG           # Line plot grouped
│
├── modalities/                # Signal processing modules (14 files)
│   ├── prepro_acc.py         # Accelerometer preprocessing
│   ├── prepro_bvp.py         # Blood Volume Pulse preprocessing
│   ├── prepro_eeg.py         # EEG preprocessing (largest: 9.4KB)
│   ├── prepro_gsr.py         # GSR preprocessing
│   ├── prepro_gyro.py        # Gyroscope preprocessing
│   ├── prepro_ppg.py         # PPG preprocessing
│   ├── prepro_temp.py        # Temperature preprocessing
│   ├── featex_acc.py         # Accelerometer feature extraction
│   ├── featex_bvp.py         # BVP feature extraction
│   ├── featex_eeg.py         # EEG feature extraction (largest: 61.6KB)
│   ├── featex_gsr.py         # GSR feature extraction
│   ├── featex_gyro.py        # Gyroscope feature extraction
│   ├── featex_ppg.py         # PPG feature extraction
│   └── featex_temp.py        # Temperature feature extraction
│
├── main.py                    # Application entry point (19KB)
├── load_data.py              # Data loading module (11.7KB)
├── prepro_data.py            # Initial data preparation (22.7KB)
├── correlate_datasets.py     # Dataset correlation analysis (9.4KB)
├── correlate_features.py     # Feature correlation analysis (16.3KB)
├── load_features.py          # Feature loading utilities (17.2KB)
├── log_parser.py             # Log file parser (34.7KB - largest)
│
├── README.md                  # Primary documentation (3.3KB)
├── FEATURE_MAP.md            # Feature-to-code mapping (previous audit)
├── FEATURE_STATUS_MATRIX.md  # Feature status (previous audit)
├── ACTUAL_ARCHITECTURE.md    # Architecture diagram (previous audit)
├── STABILIZATION_ROADMAP.md  # Roadmap (previous audit)
├── TECHNICAL_DEBT_REPORT.md  # Technical debt (previous audit)
│
├── project_manifest.json     # Old manifest (previous audit - incomplete)
├── AUDIT_PROJECT_MANIFEST.json  # Current audit manifest
└── AUDIT_2025_* files        # Current audit deliverables

Total Project Size: 1.16 MB (290KB source code, 900KB images, 24KB docs)
```

---

## File Categorization

### Source Code (21 files, 289,558 bytes)
**Data Processing & Orchestration (6 files):**
- `main.py` - Entry point and orchestration
- `prepro_data.py` - Initial preprocessing
- `load_data.py` - Data loading
- `correlate_datasets.py` - Dataset correlation
- `correlate_features.py` - Feature correlation
- `load_features.py` - Feature loading utilities

**Utility Scripts (1 file):**
- `log_parser.py` - Log parsing (34.7KB - complex)

**Modality Preprocessing (7 files):**
- `modalities/prepro_acc.py` (6.6KB)
- `modalities/prepro_bvp.py` (9.7KB)
- `modalities/prepro_eeg.py` (9.4KB)
- `modalities/prepro_gsr.py` (6.1KB)
- `modalities/prepro_gyro.py` (6.2KB)
- `modalities/prepro_ppg.py` (6.3KB)
- `modalities/prepro_temp.py` (6.2KB)

**Modality Feature Extraction (7 files):**
- `modalities/featex_acc.py` (7.2KB)
- `modalities/featex_bvp.py` (6.9KB)
- `modalities/featex_eeg.py` (61.6KB - most complex)
- `modalities/featex_gsr.py` (4.5KB)
- `modalities/featex_gyro.py` (13.5KB)
- `modalities/featex_ppg.py` (10.6KB)
- `modalities/featex_temp.py` (3.9KB)

### Documentation (6 files, 16,313 bytes)
- `README.md` - Primary documentation
- `FEATURE_MAP.md` - Feature mapping (previous audit)
- `FEATURE_STATUS_MATRIX.md` - Status classification (previous audit)
- `ACTUAL_ARCHITECTURE.md` - Architecture (previous audit)
- `STABILIZATION_ROADMAP.md` - Roadmap (previous audit)
- `TECHNICAL_DEBT_REPORT.md` - Debt analysis (previous audit)

### Assets (8 files, 899,746 bytes)
- 8 JPG images (slide deck screenshots for documentation)

### IDE Configuration (7 files, 1,230 bytes)
- `.idea/` directory (PyCharm/IntelliJ IDEA configuration)

### Configuration (1 file, 7,784 bytes)
- `project_manifest.json` (old audit metadata)

---

## Class and Module Inventory

### Core Classes (3)
1. **LoadData** (`load_data.py`) - Data file loading and parsing
2. **PreProData** (`prepro_data.py`) - Initial data preparation
3. **LogParser** (`log_parser.py`) - Log file parsing and analysis

### Preprocessing Classes (7)
1. **PreProEEG** (`modalities/prepro_eeg.py`)
2. **PreProPPG** (`modalities/prepro_ppg.py`)
3. **PreProACC** (`modalities/prepro_acc.py`)
4. **PreProGYRO** (`modalities/prepro_gyro.py`)
5. **PreProBVP** (`modalities/prepro_bvp.py`)
6. **PreProGSR** (`modalities/prepro_gsr.py`)
7. **PreProTEMP** (`modalities/prepro_temp.py`)

### Feature Extraction Classes (7)
1. **FeatExEEG** (`modalities/featex_eeg.py`) - Most complex (61.6KB)
2. **FeatExPPG** (`modalities/featex_ppg.py`)
3. **FeatExACC** (`modalities/featex_acc.py`)
4. **FeatExGYRO** (`modalities/featex_gyro.py`)
5. **FeatExBVP** (`modalities/featex_bvp.py`)
6. **FeatExGSR** (`modalities/featex_gsr.py`)
7. **FeatExTEMP** (`modalities/featex_temp.py`)

**Total Classes:** 17

---

## Dependency Analysis

### Third-Party Dependencies (6 packages)
**CRITICAL:** No `requirements.txt` file exists

**Core Scientific Computing:**
- `numpy` - Used in 19/21 files (90% coverage)
- `pandas` - Used in 21/21 files (100% coverage) - CRITICAL DEPENDENCY
- `scipy` - Used in 14/21 files (67% coverage)

**Visualization:**
- `matplotlib` - Used in 20/21 files (95% coverage) - CRITICAL DEPENDENCY
- `seaborn` - Used in 3/21 files (14% coverage)

**Domain-Specific:**
- `mne` - EEG analysis library, used in 5/21 files (24% coverage)

### Standard Library Usage (12 modules)
os, pathlib, json, pickle, csv, datetime, glob, logging, math, re, traceback, ast

### Security Risk: Pickle Usage
**Files using pickle (4):**
- `load_data.py`
- `log_parser.py`
- `main.py`
- `prepro_data.py`

**Risk:** Arbitrary code execution if unpickling untrusted data

---

## Missing Critical Infrastructure

### Build & Packaging (NONE)
- ❌ requirements.txt
- ❌ setup.py / pyproject.toml
- ❌ Makefile
- ❌ MANIFEST.in

### Testing (NONE)
- ❌ test files (0% coverage)
- ❌ pytest.ini
- ❌ tox.ini
- ❌ .coveragerc

### Code Quality (NONE)
- ❌ .pylintrc / .flake8
- ❌ .pre-commit-config.yaml
- ❌ .mypy.ini (type checking)

### CI/CD (NONE)
- ❌ .github/workflows/
- ❌ .gitlab-ci.yml
- ❌ Jenkinsfile

### Containerization (NONE)
- ❌ Dockerfile
- ❌ docker-compose.yml
- ❌ .dockerignore

### Configuration (NONE)
- ❌ .env / .env.example
- ❌ config.yaml / config.json
- ❌ Any configuration management system

---

## Git Repository Analysis

### Commit History
- **Total Commits:** 11
- **Contributors:** 2 (Siddhant Gadamsetti + 1 bot)
- **First Commit:** February 4, 2024
- **Last Core Development:** April 1, 2024
- **Previous Audit:** August 21, 2025 (bot-generated docs)
- **Current Audit:** November 5, 2025

### Development Timeline
- **Active Development:** 3 days total (March 1, April 1, August 21)
- **Dormancy Period:** 477 days (April 1, 2024 - August 21, 2025)
- **Activity Ratio:** 0.55% active, 99.45% dormant

### Branch Status
- **master** - Primary branch (remote)
- **audit-deliverables** - Previous audit PR (merged, can be archived)
- **claude/codebase-audit-protocol-...** - 2 audit branches (1 stale, 1 active)

---

## Code Quality Metrics

**Total Lines of Code:** 6,631 LOC

**Largest Files (by LOC):**
1. `modalities/featex_eeg.py` - 61.6KB
2. `log_parser.py` - 34.7KB
3. `prepro_data.py` - 22.7KB
4. `main.py` - 19KB
5. `load_features.py` - 17.2KB

**Code Distribution:**
- Modality-specific code: 48% (14 files)
- Core orchestration: 29% (6 files)
- Utilities: 19% (2 files)
- Documentation: 4% (6 files)

---

## Critical Findings Summary

### Infrastructure Maturity: 0/10
- No build system
- No dependency management
- No testing infrastructure
- No CI/CD
- No configuration system

### Security Posture: 2/10
- Unversioned dependencies
- Pickle usage without safety review
- No security scanning
- No vulnerability management

### Development Velocity: STALLED
- 16-month abandonment
- Bulk upload pattern (no iterative development)
- 3 days of active commits in 18 months

### Project State: PRE-ALPHA
- Not distributable
- Not reproducible
- Not portable (hardcoded paths)
- Not testable
- Research prototype only

---

**End of Manifest**
