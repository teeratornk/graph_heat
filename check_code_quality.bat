@echo off
echo Running code quality checks...

echo 1. Running flake8...
flake8 *.py

echo.
echo 2. Running mypy...
mypy *.py

echo.
echo 3. Running pylint...
pylint *.py

echo.
echo Code quality check complete!
pause
