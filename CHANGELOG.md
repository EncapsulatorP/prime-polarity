# Changelog

## 0.2.0 — Repository Improvements & Code Quality

### ✨ New Features
- Added comprehensive type hints throughout the codebase
- Expanded test suite with CLI functionality tests
- Enhanced documentation with better examples and quickstart guide

### 🐛 Bug Fixes
- Fixed CLI quickstart script (removed non-existent 'score' subcommand)
- Resolved all linting issues (12 ruff errors fixed)
- Fixed version consistency across project files

### 🔧 Project Structure
- Removed redundant `setup.py` in favor of modern `pyproject.toml`
- Added missing `pandas` dependency to project configuration
- Enhanced `.gitignore` with comprehensive exclusions
- Standardized code formatting with Black

### 📚 Documentation
- Added comprehensive docstrings to all public functions
- Updated README with quickstart section and CI badge
- Fixed broken CLI command examples

### 🧪 Testing & CI
- Added integration tests for CLI functionality
- Maintained 100% test pass rate
- Existing GitHub Actions CI continues to work

## 0.1.0 — initial release
- CLI scorer
- Transforms: Möbius twist, fractional-part, forward diff, log-Mellin slope, Dirichlet projections
- Tests and examples
- Docs + appendix on irrational/transcendental generators
