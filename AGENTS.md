# AGENTS.md - AI Assistant Guide for graph_data

## Dev environment tips
- Use `python -m venv venv` to create a virtual environment, then `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
- Run `pip install -r requirements.txt` to install all dependencies in one go
- Use `pip install -e .` if you want to install the project in editable mode for development
- Check data files exist in `data/` folder before running analysis - expects `.hdf5` or `.h5` files
- Output files will be created in `outputs/eda/` - this directory is auto-created if missing

## Testing instructions
- Use `flake8 *.py --max-line-length=100` to check code style
- Use `mypy *.py --ignore-missing-imports` for type checking
- Use `pylint *.py --max-line-length=100` for comprehensive code analysis

## Code quality checklist
- All functions must have type hints: `def function(param: type) -> return_type:`
- All functions must have Google-style docstrings
- Keep line length under 100 characters
- Use meaningful variable names (avoid single letters except for indices)
- Handle file I/O errors explicitly with try/except blocks
- Validate array shapes before operations to avoid dimension mismatches

## PR instructions
- Title format: [graph_data] <Description of changes>
- Always run `flake8`, `mypy`, and `pylint` before committing
- Test with at least 2 different HDF5 files to ensure compatibility
- Update docstrings if function signatures change
- Add new dependencies to `requirements.txt` if needed
- Document any new output files or folders created
