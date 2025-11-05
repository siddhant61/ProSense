# Stabilization Roadmap - ProSense
**Audit Date:** November 5, 2025
**Auditor:** Claude Code (Senior Technical Auditor AI)
**Roadmap Duration:** 12 weeks
**Target State:** Production-Ready Release v1.0

---

## Executive Summary

This roadmap provides a **structured, prioritized plan** to stabilize ProSense from its current state (PRE-ALPHA, non-functional for external users) to a **production-ready v1.0 release**. The roadmap is organized into 4 phases over 12 weeks, focusing first on critical blockers, then quality improvements, and finally production readiness.

**Current State:**
- Health Score: 2.8/10
- Test Coverage: 0%
- Deployment Readiness: 0%
- Can be used by: 1 person (original developer only)

**Target State (v1.0):**
- Health Score: 8.0/10
- Test Coverage: 80%+
- Deployment Readiness: 100%
- Can be used by: Anyone with Python

---

## Roadmap Overview

| Phase | Duration | Focus | Deliverables |
|-------|----------|-------|--------------|
| **Phase 0: Emergency Stabilization** | Week 1 | Unblock external usage | requirements.txt, config.yaml, basic docs |
| **Phase 1: Foundation** | Weeks 2-4 | Infrastructure & Safety | Build system, tests, CI/CD, security |
| **Phase 2: Quality & Maintainability** | Weeks 5-8 | Code quality, refactoring | Eliminate duplication, separate concerns, docs |
| **Phase 3: Production Readiness** | Weeks 9-12 | Polish & Release | Comprehensive tests, Docker, CLI, v1.0 release |

---

## Phase 0: Emergency Stabilization (Week 1)
**Goal:** Make the project usable by external developers
**Duration:** 5 days
**Priority:** P0 - CRITICAL

### Objectives
✓ Enable anyone to set up and run the project
✓ Remove hardcoded paths
✓ Document actual dependencies
✓ Fix critical README errors

### Tasks

#### Task 0.1: Create requirements.txt ⏱️ 1 hour
**Priority:** P0 - BLOCKER
**Assignee:** DevOps/Platform Engineer

```bash
# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core Scientific Computing
numpy>=1.24.0,<2.0.0
pandas>=2.0.0,<3.0.0
scipy>=1.10.0,<2.0.0

# Visualization
matplotlib>=3.7.0,<4.0.0
seaborn>=0.12.0,<1.0.0

# Domain-Specific
mne>=1.4.0,<2.0.0

# Development (optional)
pytest>=7.4.0
pytest-cov>=4.1.0
flake8>=6.0.0
black>=23.0.0
bandit>=1.7.5
safety>=2.3.0
EOF

# Test installation
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Acceptance Criteria:**
- [ ] requirements.txt file exists in repository root
- [ ] All dependencies install without errors
- [ ] `pip check` shows no conflicts
- [ ] README updated with correct installation command

---

#### Task 0.2: Create Configuration System ⏱️ 3 hours
**Priority:** P0 - BLOCKER
**Assignee:** Backend Engineer

**Subtasks:**
1. Create `config.yaml` template (30 min)
2. Create `config.py` loader module (1 hour)
3. Update `main.py` to use config (1 hour)
4. Test with different configurations (30 min)

```yaml
# config.yaml
data:
  input_path: ./data/raw
  output_path: ./data/processed

processing:
  eeg:
    sampling_rate: 200
    notch_filter_freq: 50
    bandpass_low: 1
    bandpass_high: 40
    epoch_duration: 5.0

  ppg:
    sampling_rate: 64
    filter_low: 0.5
    filter_high: 4.0

output:
  save_features: true
  save_plots: true
  plot_format: png
```

```python
# config.py
import yaml
from pathlib import Path

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required = ['data', 'processing', 'output']
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config section: {field}")

    return config
```

**Acceptance Criteria:**
- [ ] config.yaml exists with all parameters
- [ ] config.py validates and loads configuration
- [ ] main.py uses config (no hardcoded paths)
- [ ] .env.example created for sensitive values
- [ ] Documentation updated with config instructions

---

#### Task 0.3: Fix README Errors ⏱️ 30 minutes
**Priority:** P0 - CRITICAL
**Assignee:** Technical Writer / Developer

**Changes Required:**
1. Fix repository name (StreamSense → ProSense)
2. Remove reference to missing requirements.txt (or add after creating it)
3. Add actual setup instructions
4. Add system requirements (Python version, OS, etc.)
5. Add quick start example

```markdown
# ProSense

**Multimodal Physiological Signal Processing Platform**

## System Requirements
- Python 3.9 or higher
- 4GB RAM minimum (8GB recommended)
- Linux, macOS, or Windows

## Quick Start

### 1. Clone Repository
\```bash
git clone https://github.com/siddhant61/ProSense.git
cd ProSense
\```

### 2. Install Dependencies
\```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
\```

### 3. Configure Data Paths
\```bash
cp config.yaml.example config.yaml
# Edit config.yaml to point to your data directory
\```

### 4. Run
\```bash
python main.py --config config.yaml
\```

## Documentation
See [docs/](docs/) for detailed documentation.
```

**Acceptance Criteria:**
- [ ] All references to "StreamSense" removed
- [ ] Installation instructions are accurate and tested
- [ ] Quick start example works on fresh installation
- [ ] System requirements documented

---

#### Task 0.4: Create Basic Documentation ⏱️ 2 hours
**Priority:** P0 - CRITICAL
**Assignee:** Technical Writer

**Documents to Create:**
1. `docs/installation.md` - Detailed setup guide
2. `docs/configuration.md` - Configuration reference
3. `docs/quickstart.md` - Tutorial with example
4. `CONTRIBUTING.md` - Contribution guidelines

**Acceptance Criteria:**
- [ ] All docs/ directory created with 4 documents
- [ ] Documentation is clear and tested
- [ ] Includes troubleshooting section

---

### Phase 0 Success Criteria
- ✓ New developer can clone, install, and run in < 15 minutes
- ✓ No code editing required to get started
- ✓ All documentation is accurate
- ✓ Basic usage example works

**Phase 0 Total Effort:** 6.5 hours
**Phase 0 Completion Date:** End of Week 1

---

## Phase 1: Foundation (Weeks 2-4)
**Goal:** Build essential infrastructure for quality and safety
**Duration:** 3 weeks
**Priority:** P0-P1

### Objectives
✓ Create build/packaging system
✓ Establish testing framework
✓ Set up CI/CD pipeline
✓ Address security vulnerabilities
✓ Add error handling

### Week 2: Build System & Initial Tests

#### Task 1.1: Create Package Build System ⏱️ 4 hours
**Priority:** P0 - BLOCKER

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "prosense"
version = "0.1.0"
description = "Multimodal physiological signal processing platform"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
    {name = "Siddhant Gadamsetti", email = "siddhant.gadamsetti@gmail.com"}
]
keywords = ["eeg", "ppg", "biosignals", "physiological-signals"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "mne>=1.4.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "flake8>=6.0.0",
    "black>=23.0.0",
    "mypy>=1.5.0",
    "bandit>=1.7.5",
    "safety>=2.3.0",
]

[project.scripts]
prosense = "prosense.main:main"

[project.urls]
Homepage = "https://github.com/siddhant61/ProSense"
Issues = "https://github.com/siddhant61/ProSense/issues"
```

**Acceptance Criteria:**
- [ ] pyproject.toml created with correct metadata
- [ ] `pip install -e .` works
- [ ] Package can be imported: `import prosense`
- [ ] Entry point works: `prosense --help`

---

#### Task 1.2: Set Up Testing Framework ⏱️ 2 hours
**Priority:** P0 - CRITICAL

**Directory Structure:**
```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration & fixtures
├── test_data_loading.py     # Data loading tests
├── test_preprocessing.py    # Preprocessing tests
├── test_feature_extraction.py  # Feature extraction tests
└── fixtures/
    ├── sample_eeg.pkl
    ├── sample_ppg.pkl
    └── config_test.yaml
```

```python
# tests/conftest.py
import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_eeg_data():
    """Generate synthetic EEG data for testing"""
    n_channels = 64
    n_samples = 1000
    sfreq = 250
    data = np.random.randn(n_channels, n_samples)
    return data, sfreq

@pytest.fixture
def sample_config():
    """Load test configuration"""
    return {
        'eeg': {
            'sampling_rate': 250,
            'bandpass_low': 1,
            'bandpass_high': 40,
        }
    }
```

**Acceptance Criteria:**
- [ ] tests/ directory structure created
- [ ] pytest.ini configured
- [ ] Sample fixtures created
- [ ] `pytest` runs successfully (even with 0 tests)

---

#### Task 1.3: Write Critical Unit Tests ⏱️ 16 hours
**Priority:** P0 - CRITICAL

**Test Plan:**

| Module | Function | Test Cases | Effort |
|--------|----------|------------|--------|
| prepro_eeg.py | apply_bandpass_filter | Filter response, frequency cutoffs, edge cases | 2h |
| prepro_eeg.py | apply_artifact_removal | ICA components, artifact detection | 3h |
| prepro_eeg.py | apply_normalization | Normality test, z-score correctness | 2h |
| featex_eeg.py | extract_power_bands | Band power calculations, frequency ranges | 3h |
| featex_eeg.py | extract_spectral_entropy | Entropy calculation, edge cases | 2h |
| load_data.py | LoadData class | File loading, format validation, error handling | 2h |
| prepro_data.py | PreProData class | Stream separation, data validation | 2h |

**Example Test:**
```python
# tests/test_preprocessing.py
import pytest
import numpy as np
from prosense.modalities.prepro_eeg import PreProEEG

def test_bandpass_filter_frequency_response(sample_eeg_data):
    """Test that bandpass filter attenuates frequencies outside pass band"""
    data, sfreq = sample_eeg_data
    prepro = PreProEEG(data, sfreq)

    # Apply 8-12 Hz bandpass (alpha band)
    prepro.apply_bandpass_filter(low=8, high=12)
    filtered = prepro.processed_data

    # Generate frequency spectrum
    freqs, psd = compute_psd(filtered, sfreq)

    # Assert: Power at 10 Hz (alpha) should be preserved
    alpha_power = psd[(freqs >= 8) & (freqs <= 12)].mean()

    # Assert: Power at 1 Hz and 40 Hz should be attenuated
    low_power = psd[(freqs >= 0.5) & (freqs <= 2)].mean()
    high_power = psd[(freqs >= 35) & (freqs <= 45)].mean()

    assert alpha_power > low_power
    assert alpha_power > high_power

def test_bandpass_filter_invalid_params():
    """Test that invalid filter parameters raise errors"""
    data = np.random.randn(64, 1000)
    prepro = PreProEEG(data, sfreq=250)

    # Test: low_freq >= high_freq
    with pytest.raises(ValueError):
        prepro.apply_bandpass_filter(low=40, high=10)

    # Test: high_freq > Nyquist
    with pytest.raises(ValueError):
        prepro.apply_bandpass_filter(low=10, high=150)  # Nyquist is 125 Hz
```

**Acceptance Criteria:**
- [ ] At least 20 unit tests written
- [ ] All critical preprocessing functions tested
- [ ] All critical feature extraction functions tested
- [ ] Tests pass on CI
- [ ] Test coverage > 40%

---

### Week 3: CI/CD & Security

#### Task 1.4: Set Up CI/CD Pipeline ⏱️ 4 hours
**Priority:** P1 - HIGH

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run tests
      run: |
        pytest --cov=prosense --cov-report=xml --cov-report=term

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install linters
      run: pip install flake8 black mypy bandit
    - name: Run flake8
      run: flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    - name: Run black
      run: black --check .
    - name: Run mypy
      run: mypy prosense --ignore-missing-imports
    - name: Run bandit
      run: bandit -r prosense -f json -o bandit-report.json

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install security tools
      run: pip install safety pip-audit
    - name: Check dependencies for vulnerabilities
      run: |
        safety check --json
        pip-audit
```

**Acceptance Criteria:**
- [ ] CI pipeline runs on every push
- [ ] Tests run on multiple OS and Python versions
- [ ] Code coverage tracked and reported
- [ ] Linting enforced
- [ ] Security scanning automated

---

#### Task 1.5: Replace Pickle with Safe Serialization ⏱️ 8 hours
**Priority:** P1 - HIGH (Security)

**Changes Required:**
```python
# OLD (insecure)
import pickle
features.to_pickle('features.pkl')
data = pickle.load(open('data.pkl', 'rb'))

# NEW (secure)
import pandas as pd
features.to_parquet('features.parquet', compression='snappy')
data = pd.read_parquet('data.parquet')
```

**Files to Update:**
- load_data.py
- log_parser.py
- main.py
- prepro_data.py

**Acceptance Criteria:**
- [ ] No pickle usage remains in codebase
- [ ] All data persistence uses parquet or HDF5
- [ ] Tests verify serialization/deserialization
- [ ] Migration guide provided for existing .pkl files

---

#### Task 1.6: Add Comprehensive Error Handling ⏱️ 6 hours
**Priority:** P1 - HIGH

**Error Handling Strategy:**
1. **Input Validation:** Validate all external inputs
2. **Graceful Degradation:** Continue processing when possible
3. **Informative Errors:** Clear error messages
4. **Logging:** Log all errors and warnings

```python
# Example: main.py with error handling
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Load configuration
        try:
            config = load_config('config.yaml')
        except FileNotFoundError:
            logger.error("config.yaml not found. Create from config.yaml.example")
            return 1
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config.yaml: {e}")
            return 1

        # Validate data path
        data_path = Path(config['data']['input_path'])
        if not data_path.exists():
            logger.error(f"Data path does not exist: {data_path}")
            return 1

        # Load data with error handling
        try:
            data = load_data(data_path)
        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            return 1

        # Process each modality
        for modality in config['processing']:
            try:
                logger.info(f"Processing {modality}...")
                process_modality(modality, data, config)
                logger.info(f"✓ {modality} processed successfully")
            except Exception as e:
                logger.error(f"✗ Failed to process {modality}: {e}")
                # Continue with other modalities
                continue

        logger.info("All processing complete")
        return 0

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 130
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        return 1
```

**Acceptance Criteria:**
- [ ] All file I/O has error handling
- [ ] All external inputs validated
- [ ] Clear error messages for users
- [ ] Logging configured throughout application
- [ ] Tests for error conditions

---

### Week 4: Validation & Documentation

#### Task 1.7: Add Input Validation ⏱️ 4 hours
**Priority:** P1 - HIGH

```python
# validation.py
from pathlib import Path
from typing import Union

def validate_data_path(path: Union[str, Path]) -> Path:
    """Validate and sanitize data path"""
    path = Path(path).resolve()

    # Check existence
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Check it's a directory
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    # Security: Prevent path traversal
    # (Ensure path is within allowed directories)
    return path

def validate_filter_params(low: float, high: float, sfreq: float):
    """Validate filter parameters"""
    if low <= 0:
        raise ValueError(f"Low frequency must be positive: {low}")
    if high <= low:
        raise ValueError(f"High frequency must be > low: {high} <= {low}")
    if high >= sfreq / 2:
        raise ValueError(f"High frequency exceeds Nyquist: {high} >= {sfreq/2}")

def validate_config(config: dict):
    """Validate configuration dictionary"""
    required_sections = ['data', 'processing', 'output']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate data section
    if 'input_path' not in config['data']:
        raise ValueError("config.data.input_path is required")

    # Validate each modality config
    for modality, params in config['processing'].items():
        if 'sampling_rate' not in params:
            raise ValueError(f"sampling_rate required for {modality}")
```

**Acceptance Criteria:**
- [ ] All external inputs validated
- [ ] Validation functions have tests
- [ ] Security considerations addressed (path traversal, etc.)

---

### Phase 1 Success Criteria
- ✓ Package installable via `pip install -e .`
- ✓ 40% test coverage achieved
- ✓ CI/CD pipeline operational
- ✓ No security vulnerabilities in dependencies
- ✓ No pickle usage
- ✓ Comprehensive error handling
- ✓ All tests passing

**Phase 1 Total Effort:** 44 hours
**Phase 1 Completion Date:** End of Week 4

---

## Phase 2: Quality & Maintainability (Weeks 5-8)
**Goal:** Improve code quality, eliminate technical debt
**Duration:** 4 weeks
**Priority:** P1-P2

### Objectives
✓ Eliminate code duplication
✓ Separate concerns (refactor FeatEx classes)
✓ Increase test coverage to 70%
✓ Add comprehensive documentation
✓ Improve code quality metrics

### Week 5-6: Refactoring

#### Task 2.1: Eliminate Code Duplication ⏱️ 12 hours
**Priority:** P1 - HIGH

**Current Problem:**
```python
# 7 nearly-identical functions in main.py
def process_eeg(...): ...
def process_ppg(...): ...
def process_acc(...): ...
# etc.
```

**Refactored Solution:**
```python
# Unified processing with factory pattern
def process_modality(modality_name: str, data, config):
    """Process any modality dynamically"""
    prepro_cls = get_preprocessor_class(modality_name)
    featex_cls = get_feature_extractor_class(modality_name)

    prepro = prepro_cls(data)
    prepro.run_pipeline(config[modality_name])

    featex = featex_cls(prepro.processed_data)
    features = featex.extract_all()
    plots = featex.generate_plots()

    return features, plots

# Factory functions
def get_preprocessor_class(modality: str):
    """Get preprocessor class for modality"""
    from prosense import modalities
    class_name = f"PrePro{modality.upper()}"
    return getattr(modalities, class_name)

# Main loop
for modality in config['modalities']:
    features, plots = process_modality(modality, data[modality], config)
    save_results(features, plots, modality)
```

**Acceptance Criteria:**
- [ ] Single `process_modality()` function replaces 7 process_* functions
- [ ] Factory pattern implemented for class loading
- [ ] All modalities still process correctly
- [ ] Tests verify refactored code
- [ ] main.py reduced from 19KB to <10KB

---

#### Task 2.2: Separate Visualization from Feature Extraction ⏱️ 16 hours
**Priority:** P2 - MEDIUM

**Current Problem:**
```python
# FeatExEEG does both extraction AND visualization
class FeatExEEG:
    def extract_power_bands(self): ...  # Feature extraction ✓
    def plot_power_bands(self): ...     # Visualization ❌ (wrong class)
```

**Refactored Solution:**
```python
# features/extractors.py
class BaseFeatureExtractor:
    def extract_all(self):
        """Extract all features (pure computation)"""
        features = {}
        features['power_bands'] = self.extract_power_bands()
        features['entropy'] = self.extract_spectral_entropy()
        return pd.DataFrame(features)

class EEGFeatureExtractor(BaseFeatureExtractor):
    def extract_power_bands(self, data):
        # Pure computation, no visualization
        return power_bands

# visualization/plotters.py
class BaseFeaturePlotter:
    def plot_all(self, features):
        """Generate all plots"""
        figs = []
        figs.append(self.plot_power_bands(features))
        figs.append(self.plot_time_series(features))
        return figs

class EEGFeaturePlotter(BaseFeaturePlotter):
    def plot_power_bands(self, features):
        # Pure visualization
        fig, ax = plt.subplots()
        # ...
        return fig

# Usage
extractor = EEGFeatureExtractor()
features = extractor.extract_all(data)

plotter = EEGFeaturePlotter()
plots = plotter.plot_all(features)
```

**Acceptance Criteria:**
- [ ] Feature extraction classes only do computation
- [ ] Visualization moved to separate plotter classes
- [ ] Can extract features without generating plots
- [ ] Can generate plots from saved features
- [ ] All existing plots still generated correctly

---

### Week 7-8: Testing & Documentation

#### Task 2.3: Increase Test Coverage to 70% ⏱️ 24 hours
**Priority:** P2 - MEDIUM

**Test Coverage Goals:**
- preprocessing modules: 80%
- feature extraction modules: 70%
- utilities: 60%
- main orchestration: 50%

**Test Types to Add:**
1. **Integration Tests** (8h)
   - End-to-end pipeline tests
   - Multi-modality processing
   - Configuration variations

2. **Edge Case Tests** (8h)
   - Empty data
   - Malformed inputs
   - Extreme parameter values
   - Boundary conditions

3. **Property-Based Tests** (8h)
   - Using hypothesis library
   - Test mathematical properties
   - Invariants (e.g., filtering preserves signal energy)

**Acceptance Criteria:**
- [ ] Test coverage ≥ 70%
- [ ] All modules have tests
- [ ] Integration tests pass
- [ ] Property-based tests for numerical functions

---

#### Task 2.4: Generate API Documentation ⏱️ 8 hours
**Priority:** P2 - MEDIUM

**Documentation Tools:**
- Sphinx for API docs
- autodoc for docstring extraction
- Read the Docs for hosting

**Steps:**
1. Add comprehensive docstrings (4h)
2. Set up Sphinx (2h)
3. Generate and deploy docs (2h)

```python
# Example docstring format (Google style)
def apply_bandpass_filter(self, low: float, high: float) -> None:
    """Apply a bandpass filter to the signal.

    Args:
        low: Lower cutoff frequency in Hz. Must be positive.
        high: Upper cutoff frequency in Hz. Must be less than Nyquist frequency.

    Raises:
        ValueError: If frequency parameters are invalid.

    Example:
        >>> prepro = PreProEEG(data, sfreq=250)
        >>> prepro.apply_bandpass_filter(1, 40)  # 1-40 Hz bandpass

    Note:
        This uses a 4th-order Butterworth filter.
    """
    pass
```

**Acceptance Criteria:**
- [ ] All public functions have docstrings
- [ ] Sphinx documentation builds successfully
- [ ] API docs published online
- [ ] Usage examples in documentation

---

### Phase 2 Success Criteria
- ✓ Code duplication eliminated (main.py < 10KB)
- ✓ Visualization separated from feature extraction
- ✓ 70% test coverage achieved
- ✓ API documentation published
- ✓ Code quality improved significantly

**Phase 2 Total Effort:** 60 hours
**Phase 2 Completion Date:** End of Week 8

---

## Phase 3: Production Readiness (Weeks 9-12)
**Goal:** Polish and release production-ready v1.0
**Duration:** 4 weeks
**Priority:** P2-P3

### Objectives
✓ Achieve 80%+ test coverage
✓ Containerize with Docker
✓ Create CLI interface
✓ Performance optimization
✓ Release v1.0

### Week 9-10: Polish & Features

#### Task 3.1: Create CLI Interface ⏱️ 8 hours
**Priority:** P2 - MEDIUM

```python
# cli.py
import click
from pathlib import Path

@click.group()
@click.version_option()
def cli():
    """ProSense: Multimodal Physiological Signal Processing"""
    pass

@cli.command()
@click.option('--config', type=click.Path(), default='config.yaml',
              help='Path to configuration file')
@click.option('--input', type=click.Path(exists=True), required=True,
              help='Input data directory')
@click.option('--output', type=click.Path(), required=True,
              help='Output directory for results')
@click.option('--modality', multiple=True,
              help='Specific modalities to process (default: all)')
@click.option('--verbose', '-v', count=True, help='Increase verbosity')
def process(config, input, output, modality, verbose):
    """Process physiological signal data"""
    # Implementation
    pass

@cli.command()
@click.argument('feature_file', type=click.Path(exists=True))
def visualize(feature_file):
    """Generate visualizations from extracted features"""
    pass

@cli.command()
def validate():
    """Validate configuration and dependencies"""
    pass

if __name__ == '__main__':
    cli()
```

**Usage:**
```bash
# Process all modalities
prosense process --input ./data/raw --output ./results

# Process specific modalities
prosense process --input ./data/raw --output ./results --modality eeg --modality ppg

# Generate plots from saved features
prosense visualize features.parquet

# Validate setup
prosense validate
```

**Acceptance Criteria:**
- [ ] CLI interface implemented with click
- [ ] All main operations accessible via CLI
- [ ] Help text comprehensive
- [ ] Works on Windows, Mac, Linux

---

#### Task 3.2: Docker Containerization ⏱️ 4 hours
**Priority:** P2 - MEDIUM

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies (for MNE)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN pip install -e .

# Create data directories
RUN mkdir -p /data/input /data/output

# Set entrypoint
ENTRYPOINT ["prosense"]
CMD ["--help"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  prosense:
    build: .
    volumes:
      - ./data:/data/input:ro
      - ./results:/data/output
      - ./config.yaml:/app/config.yaml:ro
    command: process --input /data/input --output /data/output
```

**Usage:**
```bash
# Build image
docker build -t prosense:latest .

# Run with docker-compose
docker-compose up

# Or run directly
docker run -v $(pwd)/data:/data/input:ro \
           -v $(pwd)/results:/data/output \
           -v $(pwd)/config.yaml:/app/config.yaml:ro \
           prosense process --input /data/input --output /data/output
```

**Acceptance Criteria:**
- [ ] Dockerfile builds successfully
- [ ] docker-compose.yml functional
- [ ] Runs on Linux, Mac, Windows (Docker Desktop)
- [ ] Documentation includes Docker instructions

---

#### Task 3.3: Achieve 80%+ Test Coverage ⏱️ 16 hours
**Priority:** P2 - MEDIUM

**Focus Areas:**
- Integration tests (full pipelines)
- Error handling paths
- Edge cases
- Performance benchmarks

**Acceptance Criteria:**
- [ ] Overall coverage ≥ 80%
- [ ] All critical paths covered
- [ ] Performance regression tests added

---

### Week 11-12: Release Preparation

#### Task 3.4: Performance Optimization ⏱️ 8 hours
**Priority:** P3 - LOW

**Optimization Targets:**
1. Profile slow functions
2. Optimize numpy/pandas operations
3. Add parallel processing for multiple files
4. Cache expensive computations

**Acceptance Criteria:**
- [ ] Processing time reduced by 20%+
- [ ] Memory usage optimized
- [ ] Benchmark tests added

---

#### Task 3.5: Prepare v1.0 Release ⏱️ 8 hours
**Priority:** P2 - MEDIUM

**Release Checklist:**
- [ ] All tests passing
- [ ] 80%+ test coverage
- [ ] Documentation complete
- [ ] CHANGELOG.md created
- [ ] Version bumped to 1.0.0
- [ ] Release notes written
- [ ] Git tag created
- [ ] PyPI package published
- [ ] Docker image published
- [ ] GitHub release created

**CHANGELOG.md:**
```markdown
# Changelog

## [1.0.0] - 2025-XX-XX (Release Date)

### Added
- Full multimodal physiological signal processing pipeline
- Support for 7 modalities: EEG, PPG, ACC, GYRO, BVP, GSR, TEMP
- Comprehensive preprocessing and feature extraction
- Configuration system (YAML)
- CLI interface
- Docker containerization
- 80%+ test coverage
- Comprehensive API documentation

### Changed
- Eliminated code duplication (refactored process_* functions)
- Separated visualization from feature extraction
- Replaced pickle with parquet for safer serialization

### Security
- Fixed arbitrary code execution vulnerability (pickle removal)
- Added input validation
- Configured security scanning (bandit, safety)

### Fixed
- Hardcoded paths removed (now configurable)
- README errors corrected
- Missing requirements.txt added
```

---

### Phase 3 Success Criteria
- ✓ 80%+ test coverage
- ✓ CLI interface functional
- ✓ Docker container working
- ✓ Performance benchmarks met
- ✓ v1.0.0 released to PyPI
- ✓ Documentation complete and published

**Phase 3 Total Effort:** 44 hours
**Phase 3 Completion Date:** End of Week 12

---

## Success Metrics

### Before Stabilization (Current)
| Metric | Value |
|--------|-------|
| Health Score | 2.8/10 |
| Test Coverage | 0% |
| Code Duplication | High (~300 lines) |
| Documentation | Minimal, with errors |
| Usability | 1 person (original dev only) |
| Deployment | Impossible |
| Security Score | 2/10 |

### After Stabilization (Target)
| Metric | Value |
|--------|-------|
| Health Score | 8.0/10 |
| Test Coverage | 80%+ |
| Code Duplication | Minimal (<5%) |
| Documentation | Comprehensive, published |
| Usability | Anyone with Python |
| Deployment | Docker + PyPI |
| Security Score | 9/10 |

---

## Resource Requirements

### Team Composition
- **Backend Engineer** (Lead): 60% time for 12 weeks
- **DevOps Engineer**: 20% time for 12 weeks
- **QA Engineer**: 30% time for 12 weeks
- **Technical Writer**: 15% time for 12 weeks

### Total Effort Estimate
- Phase 0: 6.5 hours
- Phase 1: 44 hours
- Phase 2: 60 hours
- Phase 3: 44 hours
- **Total: 154.5 hours** (~4 weeks of full-time work)

### Budget Estimate
- At $150/hour: $23,175
- At $200/hour: $30,900

---

## Risk Management

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Dependencies break with upgrades | Medium | High | Pin all versions; test upgrades on CI |
| MNE installation issues | Medium | Medium | Document system dependencies; provide Docker |
| Test data unavailable | Low | High | Generate synthetic test data |
| Performance regressions | Medium | Medium | Add performance benchmarks to CI |
| Original developer unavailable | Medium | High | Comprehensive documentation; knowledge transfer |

---

## Post-v1.0 Roadmap (Future)

### v1.1 (3 months after v1.0)
- Web API (REST/GraphQL)
- Real-time processing support
- Plugin system for custom modalities
- Advanced visualization dashboard

### v2.0 (6 months after v1.0)
- Machine learning integration
- Automated feature selection
- Multi-user support
- Cloud deployment (AWS/GCP/Azure)

---

## Conclusion

This roadmap provides a clear, actionable path to stabilize ProSense from its current state to a production-ready v1.0 release in **12 weeks**. The phased approach ensures:

1. **Immediate unblocking** (Week 1): Anyone can use the project
2. **Foundation building** (Weeks 2-4): Safety and quality infrastructure
3. **Quality improvement** (Weeks 5-8): Maintainable, well-tested code
4. **Production readiness** (Weeks 9-12): Polished, distributable product

**Next Steps:**
1. Review and approve this roadmap
2. Allocate resources (team members)
3. Create GitHub project board with tasks
4. Begin Phase 0 immediately

**Success depends on:**
- Dedicated team commitment
- Following the roadmap sequentially (don't skip P0 tasks)
- Regular testing and validation
- Clear communication and documentation

---

**End of Stabilization Roadmap**
