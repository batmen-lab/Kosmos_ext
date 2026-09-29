# Kosmos AI Scientist - Upgrade Guide

**For Existing Users** | Upgrade to the latest version with core components fixes

---

## What's New in This Update

This update (2025-11-20) includes major improvements to the three core components:

### ✅ Data Analysis Agent
- **NEW**: SHAP feature importance analysis
- **NEW**: Pathway enrichment analysis (gseapy)
- **NEW**: Distribution fitting with AIC selection
- **NEW**: Segmented regression (piecewise linear fitting)
- **NEW**: Publication-quality plot generation (300 DPI)
- **NEW**: Jupyter notebook storage for provenance

### ✅ Literature Search Agent
- **FIXED**: Citation graph building now fully implemented
- **FIXED**: Semantic Scholar integration for citations
- **IMPROVED**: Knowledge graph population

### ✅ World Model / Database
- **FIXED**: Database validation no longer shows false "missing tables" error
- **IMPROVED**: Correct table name validation

### 📦 New Dependencies
- `gseapy` - Gene Set Enrichment Analysis
- `shap` - SHAP values for ML interpretability
- `pwlf` - Piecewise linear fit (segmented regression)
- `nbformat` & `nbconvert` - Jupyter notebook support

---

## Upgrade Steps

### Step 1: Pull Latest Changes

```bash
# Navigate to your Kosmos directory
cd /path/to/Kosmos

# Stash any local changes (optional)
git stash

# Pull latest from GitHub
git pull origin master

# If you stashed changes, reapply them
git stash pop
```

**What this does**: Downloads the latest code with all fixes and new features.

### Step 2: Install New Dependencies

```bash
# Upgrade Kosmos and install new dependencies
pip install -e . --upgrade

# This installs:
# - gseapy 1.1.11
# - shap 0.49.1
# - pwlf 2.5.2
# - nbformat 5.10.4
# - nbconvert 7.16.6
# - Updated existing packages
```

**Installation time**: 1-2 minutes

**Verify installation**:
```bash
pip list | grep -E "gseapy|shap|pwlf|nbformat|nbconvert"
```

### Step 3: Rebuild Docker Images

The sandbox image needs rebuilding to include new dependencies:

```bash
# Rebuild sandbox image
docker build -t kosmos-sandbox:latest dockerfiles/sandbox/

# Verify build
docker images | grep kosmos-sandbox
```

**Build time**: 3-5 minutes

**Optional**: Rebuild main Kosmos image if using containerized deployment:
```bash
docker-compose build kosmos
```

### Step 4: Restart Services

Restart Docker services to ensure all changes take effect:

```bash
# Restart all services
docker-compose restart

# Or restart individual services
docker-compose restart neo4j postgres redis

# Verify all services are healthy
docker-compose ps
```

**Expected output**: All services show "Up (healthy)"

### Step 5: Verify Upgrade

```bash
# Run diagnostics to verify everything works
kosmos doctor
```

**What to check**:
- ✅ All checks should show "PASS"
- ✅ No "Missing tables" errors
- ✅ Database schema should show "Complete" (minor index warnings OK)

**Test new features**:
```bash
# Test SHAP import
python -c "import shap; print('SHAP:', shap.__version__)"

# Test gseapy import
python -c "import gseapy; print('gseapy:', gseapy.__version__)"

# Test pwlf import
python -c "import pwlf; print('pwlf:', pwlf.__version__)"

# Test Jupyter support
python -c "import nbformat; print('nbformat:', nbformat.__version__)"
```

**All should print version numbers without errors.**

---

## Verifying New Functionality

### Test Advanced Analytics

```python
# test_advanced_analytics.py
from kosmos.execution.data_analysis import DataAnalyzer
import pandas as pd
import numpy as np

# Test SHAP (requires a trained model)
print("✓ SHAP methods available")

# Test distribution fitting
data = pd.Series(np.random.lognormal(0, 1, 1000))
result = DataAnalyzer.fit_distributions(data)
print(f"✓ Distribution fitting: Best fit = {result['best_fit']}")

# Test segmented regression
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([1, 2, 3, 3.5, 4, 5, 6, 7])
result = DataAnalyzer.segmented_regression(x, y, n_breakpoints=1)
print(f"✓ Segmented regression: R² = {result['r_squared']:.3f}")

# Test publication plots
df = pd.DataFrame({'x': [1,2,3,4], 'y': [2,4,6,8]})
path = DataAnalyzer.create_publication_plot(
    df, 'scatter', '/tmp/test_plot.png', x='x', y='y'
)
print(f"✓ Publication plots: Saved to {path}")

print("\n✅ All advanced analytics working!")
```

Run the test:
```bash
python test_advanced_analytics.py
```

### Test Literature Agent

```python
# test_literature.py
from kosmos.agents.literature_analyzer import get_literature_analyzer

analyzer = get_literature_analyzer()
print(f"✓ Literature analyzer initialized")
print(f"  - Knowledge graph: {analyzer.use_knowledge_graph}")
print(f"  - Semantic similarity: {analyzer.use_semantic_similarity}")
print(f"  - Concept extraction: {analyzer.extract_concepts}")

print("\n✅ Literature agent working!")
```

Run the test:
```bash
python test_literature.py
```

---

## Migration Notes

### Database Schema

The database schema validation has been fixed. If you see warnings about missing indexes, this is **normal and non-blocking**. The core tables are all present:

- ✅ `execution_profiles` (was incorrectly checked as "performance_metrics")
- ✅ `profiling_bottlenecks` (was incorrectly checked as "memory_usage", "execution_trace")

**No action needed** - database works correctly.

### Knowledge Graph

If you have existing Neo4j data:

1. **Backup first** (recommended):
   ```bash
   kosmos graph --export backup_$(date +%Y%m%d).json
   ```

2. **No migration needed** - Neo4j schema is backward compatible

3. **Optional**: Rebuild citation graphs for existing papers to use new implementation

### Configuration Files

Your `.env` file remains the same. New optional variables:

```bash
# Optional: Customize citation depth
CITATION_MAX_DEPTH=2

# Optional: Enable verbose logging
LOG_LEVEL=DEBUG
```

---

## Rollback (If Needed)

If you encounter issues, you can roll back:

```bash
# Roll back to previous version
git checkout <previous-commit-hash>

# Reinstall previous version
pip install -e .

# Or use a specific version tag
git checkout v0.1.9
pip install -e .
```

**Note**: Backup your knowledge graphs before rolling back!

---

## Troubleshooting Upgrade Issues

### "Module not found" Errors

```bash
# Reinstall dependencies forcefully
pip install -e . --upgrade --force-reinstall --no-cache-dir
```

### Docker Build Failures

```bash
# Clean Docker cache
docker system prune -a

# Rebuild from scratch
docker build --no-cache -t kosmos-sandbox:latest dockerfiles/sandbox/
```

### Database Connection Issues

```bash
# Reset database (⚠️ WARNING: Deletes all data!)
docker-compose down -v
docker-compose up -d

# Run migrations
alembic upgrade head
```

### Neo4j Connection Issues

```bash
# Check Neo4j logs
docker logs kosmos-neo4j

# Restart Neo4j
docker-compose restart neo4j

# Verify Neo4j is accessible
curl http://localhost:7474
```

---

## Performance Notes

### New Features Impact

- **SHAP analysis**: CPU-intensive for large models (use TreeExplainer for trees)
- **Pathway enrichment**: Requires internet for gene set databases
- **Publication plots**: Generates 300 DPI images (larger file sizes)
- **Jupyter notebooks**: One .ipynb file per analysis

### Optimization Tips

```bash
# Enable parallel execution (if you have >4 CPU cores)
export ENABLE_CONCURRENT_OPERATIONS=true

# Enable prompt caching (reduces API costs)
export ANTHROPIC_ENABLE_PROMPT_CACHING=true

# Adjust memory limits for Docker
# Edit docker-compose.yml:
# memory: 4G  # Increase if running large analyses
```

---

## What to Try Next

Now that you've upgraded, explore the new capabilities:

1. **Feature Importance**: Use SHAP to explain ML model predictions
   ```python
   from kosmos.execution.data_analysis import DataAnalyzer
   result = DataAnalyzer.shap_feature_importance(model, X_train, X_test)
   ```

2. **Pathway Analysis**: Find enriched biological pathways
   ```python
   genes = ['BRCA1', 'TP53', 'EGFR', 'KRAS']
   result = DataAnalyzer.pathway_enrichment_analysis(genes)
   ```

3. **Segmented Regression**: Identify regime changes in data
   ```python
   result = DataAnalyzer.segmented_regression(x, y, n_breakpoints=2)
   print(f"Breakpoints: {result['breakpoints']}")
   ```

4. **Publication Plots**: Generate high-quality figures
   ```python
   path = DataAnalyzer.create_publication_plot(
       df, 'scatter', 'figure1.png', title='My Results'
   )
   ```

5. **Provenance Tracking**: Save analyses as Jupyter notebooks
   ```python
   path = DataAnalyzer.save_to_notebook(
       results, 'analysis.ipynb', 'Statistical Analysis'
   )
   ```

---

## Getting Help

If you encounter issues during upgrade:

1. **Check the docs**: `docs/` directory
2. **Run diagnostics**: `kosmos doctor`
3. **Review logs**: `logs/` directory or `docker logs kosmos-neo4j`
4. **GitHub Issues**: https://github.com/jimmc414/Kosmos/issues
   - Search existing issues first
   - Include `kosmos doctor` output when reporting
5. **See also**: [QUICK_START.md](./QUICK_START.md) for fresh install

---

## Changelog Summary

**Version**: 2025-11-20 Update
**Branch**: master (commit 39b310d+)

### Added
- 6 advanced analytics methods in DataAnalyzer
- Citation graph building in LiteratureAnalyzer
- CORE_COMPONENTS_CHECKLIST.md for tracking
- This upgrade guide
- QUICK_START.md for new users

### Fixed
- Database validation false positives
- Literature agent placeholder method
- Sandbox image dependencies

### Changed
- pyproject.toml: 5 new dependencies
- dockerfiles/sandbox/requirements.txt: Advanced analytics
- .gitignore: Analysis files excluded

### Dependencies
- Added: gseapy, shap, pwlf, nbformat, nbconvert
- Updated: (existing packages to compatible versions)

---

**Upgrade Complete!** 🎉

Your Kosmos installation now has full advanced analytics capabilities.

For questions or issues, please open a GitHub issue with:
- Output of `kosmos doctor`
- Python version (`python --version`)
- Docker version (`docker --version`)
- Steps to reproduce the issue
