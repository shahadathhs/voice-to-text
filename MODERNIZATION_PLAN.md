# Voice-to-Text Modernization Plan

**Date**: 2026-04-26
**Reference**: EcoRoute Atlas tooling and best practices
**Goal**: Update voice-to-text project to match ecoroute's modern Python tooling

---

## 📊 Executive Summary

The voice-to-text project currently uses legacy Python packaging and tooling. This plan outlines the migration to modern tooling matching the ecoroute project's standards, which uses:
- **uv** for fast, modern package management
- **semantic-release** for automated versioning
- **Comprehensive code quality tooling** (ruff, black, mypy, bandit)
- **Pre-commit hooks** for automated quality checks
- **GitHub Actions CI/CD** for automated testing and releases
- **Modern Docker practices** with multi-stage builds

---

## 🔍 Current State Analysis

### voice-to-text (Current)
- **Package Manager**: pip + requirements.txt
- **Build System**: None (minimal pyproject.toml with only ruff config)
- **Version Management**: Manual, no automation
- **Code Quality**: ruff only (minimal config)
- **Pre-commit**: None
- **CI/CD**: None
- **Python Version**: 3.10
- **Docker**: Legacy single-stage build with pip
- **Documentation**: Basic README only

### ecoroute (Reference Standard)
- **Package Manager**: uv (ultra-fast Python package manager)
- **Build System**: hatchling with proper pyproject.toml
- **Version Management**: semantic-release + bumpversion
- **Code Quality**: ruff + black + mypy + bandit + pre-commit
- **Pre-commit**: Comprehensive hook configuration
- **CI/CD**: GitHub Actions (ci.yml, release.yml)
- **Python Version**: 3.14
- **Docker**: Modern multi-stage build with uv
- **Documentation**: Comprehensive CLAUDE.md

---

## 🎯 Modernization Plan

### Phase 1: Package Management Migration ⭐ HIGH PRIORITY

#### 1.1 Migrate to UV Package Manager

**Current State**: Uses requirements.txt with pip

**Target State**: Use uv with pyproject.toml

**Changes Required**:
1. Create comprehensive `pyproject.toml` with:
   - Proper build system configuration (hatchling)
   - Project metadata (name, version, authors, etc.)
   - All dependencies from requirements.txt
   - Development dependencies (ruff, black, mypy, bandit, pre-commit)
   - Tool configurations (ruff, black, mypy, semantic-release)

2. Remove `requirements.txt` (can keep for reference but mark as deprecated)

3. Update `pyproject.toml` structure:
   ```toml
   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [project]
   name = "voice-to-text"
   description = "AI-powered voice transcription service"
   readme = "README.md"
   requires-python = ">=3.14"
   dynamic = ["version"]
   dependencies = [
       # Move all dependencies from requirements.txt here
   ]

   [dependency-groups]
   dev = [
       "pre-commit>=4.5.1",
       "build>=1.4.3",
       "bandit>=1.9.4",
       "python-semantic-release>=10.5.3",
       "ruff==0.15.11",
       "black==26.3.1",
       "mypy==1.20.1",
   ]
   ```

**Benefits**:
- 10-100x faster dependency installation
- Better dependency resolution
- Consistent with modern Python packaging standards
- Lock file support (uv.lock)

**Estimated Effort**: 2-3 hours

---

### Phase 2: Build System & Version Management ⭐ HIGH PRIORITY

#### 2.1 Implement Semantic Release

**Current State**: No automated versioning

**Target State**: semantic-release with bumpversion

**Changes Required**:
1. Add semantic-release configuration to pyproject.toml
2. Create `.bumpversion.cfg` for version tracking
3. Add version variable to package `__init__.py`
4. Configure conventional commit parsing

**Files to Create**:
- `.bumpversion.cfg`:
  ```ini
  [bumpversion]
  current_version = 1.0.0
  commit = True
  tag = True
  tag_name = v{new_version}

  [bumpversion:file:pyproject.toml]
  search = version = "{current_version}"
  replace = version = "{new_version}"

  [bumpversion:file:voice_to_text/__init__.py]
  search = __version__ = "{current_version}"
  replace = __version__ = "{new_version}"
  ```

- Add version to `voice_to_text/__init__.py`:
  ```python
  __version__ = "1.0.0"
  ```

**Benefits**:
- Automated version bumps based on commits
- Git tags created automatically
- Changelog generation
- No more manual version management

**Estimated Effort**: 1-2 hours

---

### Phase 3: Code Quality Tooling ⭐ HIGH PRIORITY

#### 3.1 Expand Ruff Configuration

**Current State**: Minimal ruff config (only E and F rules)

**Target State**: Comprehensive ruff configuration matching ecoroute

**Changes Required**:
Update `[tool.ruff]` section in pyproject.toml:
- Enable more rule sets (I, N, W, UP, etc.)
- Configure proper excludes
- Set line-length to 88 (Black compatible)
- Add per-file ignores

#### 3.2 Add Black Code Formatter

**Current State**: Not using Black

**Target State**: Black for consistent code formatting

**Changes Required**:
Add to pyproject.toml:
```toml
[tool.black]
line-length = 88
target-version = ['py314']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''
```

#### 3.3 Add MyPy Type Checker

**Current State**: No type checking

**Target State**: MyPy for static type checking

**Changes Required**:
Add to pyproject.toml:
```toml
[tool.mypy]
python_version = "3.14"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = true
```

#### 3.4 Add Bandit Security Scanner

**Current State**: No security scanning

**Target State**: Bandit for security vulnerability detection

**Changes Required**:
Add to dev dependencies and create `.bandit` config:
```ini
[bandit]
exclude_dirs = ['/tests', '/venv']
skips = ['B101']
```

**Estimated Effort**: 2-3 hours

---

### Phase 4: Pre-commit Hooks ⭐ HIGH PRIORITY

#### 4.1 Implement Pre-commit Configuration

**Current State**: No pre-commit hooks

**Target State**: Comprehensive pre-commit hooks matching ecoroute

**Changes Required**:
Create `.pre-commit-config.yaml`:
```yaml
repos:
  # General pre-commit hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: debug-statements

  # Ruff - Fast Python linter and formatter
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.9.6
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  # Type checking with mypy
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.14.1
    hooks:
      - id: mypy
        additional_dependencies:
          - pydantic>=2.0
          - types-requests
```

**Estimated Effort**: 1 hour

---

### Phase 5: Makefile Enhancement ⭐ MEDIUM PRIORITY

#### 5.1 Expand Makefile Targets

**Current State**: Basic Makefile with ~15 targets

**Target State**: Comprehensive Makefile matching ecoroute's 40+ targets

**New Targets to Add**:
```makefile
# Setup
setup: venv install
venv: # Create virtual environment with uv
reset-venv: # Force reset venv
install: # Install dependencies with uv

# Code Quality
format-check: # Check if code needs formatting
type-check: # Run mypy
check-all: lint type-check # Run all checks
fix-all: lint-fix format # Fix all auto-fixable issues

# Build & Release
build: # Build distribution packages
release-dry-run: # Preview release
release-changelog: # Generate changelog
release-publish: # Publish release

# Utilities
clean: # Clean up all generated files
shell: # Open Python shell with app context
logs: # Show application logs
update: # Update dependencies
freeze: # Update lock file
list: # List installed dependencies
add PKG=name: # Add new package
add-dev PKG=name: # Add dev package
remove PKG=name: # Remove package

# CI/CD
security: # Run bandit security scan
ci: pre-commit-run security build # Run full CI pipeline
```

**Estimated Effort**: 2-3 hours

---

### Phase 6: Docker Modernization ⭐ MEDIUM PRIORITY

#### 6.1 Update Dockerfile

**Current State**: Legacy single-stage build with Python 3.10

**Target State**: Modern multi-stage build with Python 3.14 and uv

**Changes Required**:
```dockerfile
FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    gcc libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml README.md ./

# Install dependencies using uv
RUN uv sync --no-dev

# Copy application source
COPY . .

# Expose port
EXPOSE 8000

# Start command
CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 6.2 Migrate to compose.yaml

**Current State**: docker-compose.yml

**Target State**: compose.yaml (newer standard) with profiles

**Changes Required**:
1. Rename `docker-compose.yml` to `compose.yaml`
2. Add profiles (infra, prod)
3. Add health checks
4. Use environment variables from .env
5. Add proper volume management
6. Update service names and configuration

**Estimated Effort**: 2-3 hours

---

### Phase 7: CI/CD Pipeline ⭐ HIGH PRIORITY

#### 7.1 Add GitHub Actions Workflows

**Current State**: No CI/CD

**Target State**: Comprehensive CI/CD matching ecoroute

**Workflows to Create**:

**ci.yml** - Run on PRs:
```yaml
name: CI
on:
  pull_request:
    branches: [main, develop]
jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: '3.14'
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Run checks
        run: |
          make clean
          make install
          make pre-commit-run
          make security
          make build
```

**release.yml** - Automated releases on main:
```yaml
name: Release
on:
  push:
    branches: [main]
permissions:
  contents: write
jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v6
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Release
        run: |
          make clean
          make install
          make pre-commit-run
          make security
          make build
          make release-publish
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

**Estimated Effort**: 2-3 hours

---

### Phase 8: Documentation ⭐ MEDIUM PRIORITY

#### 8.1 Create CLAUDE.md

**Current State**: Basic README only

**Target State**: Comprehensive CLAUDE.md development guide

**Content to Include**:
- Project overview and architecture
- Development workflow
- Code conventions
- Testing guidelines
- Deployment considerations
- Common tasks
- Debugging tips
- Pre-commit checklist

**Estimated Effort**: 3-4 hours

---

### Phase 9: Project Structure ⭐ LOW PRIORITY

#### 9.1 Reorganize Project Structure

**Current State**: Flat structure with minimal organization

**Target State**: Organized structure matching ecoroute

**Proposed Structure**:
```
voice-to-text/
├── voice_to_text/
│   ├── __init__.py           # Package init with __version__
│   ├── api/                  # API routes (if applicable)
│   ├── core/                 # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration settings
│   │   ├── errors.py         # Error handlers
│   │   └── logger.py         # Logging setup
│   ├── models/               # Data models
│   ├── services/             # Business logic
│   │   ├── __init__.py
│   │   └── transcriber.py    # Transcription service
│   └── main.py               # Application entry point
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── test_api.py
│   └── test_services.py
├── docs/                     # Documentation
├── pyproject.toml            # Project configuration
├── Makefile                  # Development automation
├── compose.yaml              # Docker compose setup
├── Dockerfile                # Production container image
└── CLAUDE.md                 # Development guide
```

**Estimated Effort**: 4-5 hours

---

### Phase 10: Environment Management ⭐ MEDIUM PRIORITY

#### 10.1 Improve Environment Configuration

**Current State**: Basic .env file

**Target State**: Structured environment management with Pydantic Settings

**Changes Required**:
1. Create `.env.example` with all required variables
2. Add Pydantic Settings for type-safe configuration
3. Add environment validation
4. Document all environment variables

**Files to Create**:
- `.env.example`: Template for environment variables
- `voice_to_text/core/config.py`: Pydantic settings class

**Estimated Effort**: 2-3 hours

---

## 📋 Implementation Priority

### Phase 1 (HIGH) - Core Infrastructure
1. ✅ Package Management Migration (UV)
2. ✅ Build System & Version Management
3. ✅ Code Quality Tooling
4. ✅ Pre-commit Hooks
5. ✅ CI/CD Pipeline

**Total Estimated Time**: 10-14 hours

### Phase 2 (MEDIUM) - Developer Experience
6. ✅ Docker Modernization
7. ✅ Makefile Enhancement
8. ✅ Documentation (CLAUDE.md)
9. ✅ Environment Management

**Total Estimated Time**: 10-13 hours

### Phase 3 (LOW) - Code Organization
10. ✅ Project Structure Reorganization

**Total Estimated Time**: 4-5 hours

---

## 🎯 Success Criteria

### Functional Requirements
- [ ] All dependencies managed via uv and pyproject.toml
- [ ] Automated version management with semantic-release
- [ ] Comprehensive code quality checks passing
- [ ] Pre-commit hooks running on all commits
- [ ] CI/CD pipeline passing on all PRs
- [ ] Automated releases working on main branch
- [ ] Modern Docker build with multi-stage optimization

### Quality Requirements
- [ ] All code formatted with Black
- [ ] No linting errors (ruff)
- [ ] No type errors (mypy)
- [ ] No security vulnerabilities (bandit)
- [ ] Comprehensive documentation in CLAUDE.md

### Developer Experience Requirements
- [ ] Single command setup (`make setup`)
- [ ] Easy dependency management (`make add`, `make remove`)
- [ ] One-command quality checks (`make check-all`)
- [ ] Clear documentation for common tasks
- [ ] Consistent with ecoroute development workflow

---

## 🚀 Quick Start Implementation

### Minimum Viable Migration (1-2 days)
For immediate benefits, implement at minimum:

1. **UV Migration** (Phase 1.1) - 2-3 hours
   - Migrate to pyproject.toml
   - Replace requirements.txt
   - Update Makefile for uv commands

2. **Pre-commit Hooks** (Phase 4) - 1 hour
   - Add .pre-commit-config.yaml
   - Install hooks

3. **Basic CI** (Phase 7.1 - ci.yml only) - 2 hours
   - Add CI workflow for PRs

4. **Expand Ruff Config** (Phase 3.1) - 1 hour
   - Enable more rules
   - Fix existing issues

**Total**: 6-7 hours for foundational improvements

### Complete Migration (1-2 weeks)
Implement all phases following the priority order above.

---

## 📝 Migration Checklist

### Pre-Migration
- [ ] Create feature branch for migration
- [ ] Backup current requirements.txt
- [ ] Document current Docker setup
- [ ] Tag current version in git

### Phase 1: Core Infrastructure
- [ ] Migrate to UV (pyproject.toml)
- [ ] Add semantic-release configuration
- [ ] Create .bumpversion.cfg
- [ ] Add version to package __init__.py
- [ ] Expand code quality tools (black, mypy, bandit)
- [ ] Create .pre-commit-config.yaml
- [ ] Create GitHub Actions workflows
- [ ] Update .gitignore for uv.lock

### Phase 2: Developer Experience
- [ ] Modernize Dockerfile
- [ ] Migrate to compose.yaml
- [ ] Expand Makefile targets
- [ ] Create CLAUDE.md
- [ ] Add .env.example
- [ ] Implement Pydantic Settings

### Phase 3: Code Organization (Optional)
- [ ] Reorganize project structure
- [ ] Move code to new structure
- [ ] Update imports
- [ ] Update documentation

### Post-Migration
- [ ] Test all Makefile targets
- [ ] Verify Docker build works
- [ ] Test CI/CD pipeline
- [ ] Run full quality check suite
- [ ] Update README with new commands
- [ ] Create PR for review
- [ ] Merge to main

---

## ⚠️ Risks and Mitigations

### Risk 1: Dependency Conflicts
**Risk**: Some dependencies may not be compatible with Python 3.14 or uv
**Mitigation**: Test all dependencies during migration, pin versions if needed

### Risk 2: Breaking Changes
**Risk**: Changes may break existing functionality
**Mitigation**: Comprehensive testing, gradual rollout, feature flags

### Risk 3: Learning Curve
**Risk**: Team unfamiliar with new tools (uv, semantic-release)
**Mitigation**: Documentation, training sessions, gradual adoption

### Risk 4: Docker Issues
**Risk**: New Dockerfile may have compatibility issues
**Mitigation**: Test thoroughly, keep old Dockerfile as backup

---

## 📚 Resources

### Documentation
- [UV Documentation](https://github.com/astral-sh/uv)
- [Semantic Release](https://python-semantic-release.readthedocs.io/)
- [Pre-commit Hooks](https://pre-commit.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Black Documentation](https://black.readthedocs.io/)

### Reference Implementation
- EcoRoute Atlas: `/home/sajib/code/python/ecoroute/`
- Reference files:
  - [pyproject.toml](/home/sajib/code/python/ecoroute/pyproject.toml)
  - [Makefile](/home/sajib/code/python/ecoroute/Makefile)
  - [Dockerfile](/home/sajib/code/python/ecoroute/Dockerfile)
  - [compose.yaml](/home/sajib/code/python/ecoroute/compose.yaml)
  - [.pre-commit-config.yaml](/home/sajib/code/python/ecoroute/.pre-commit-config.yaml)

---

## 🔄 Next Steps

1. **Review and Approve**: Review this plan with the team
2. **Create Feature Branch**: Start with `feature/modernize-tooling`
3. **Begin Migration**: Start with Phase 1 (Core Infrastructure)
4. **Iterate**: Test and adjust as needed
5. **Document**: Update this plan with lessons learned

---

**Status**: Planning
**Owner**: Development Team
**Review Date**: 2026-04-26
**Target Completion**: 2026-05-10 (2 weeks)
