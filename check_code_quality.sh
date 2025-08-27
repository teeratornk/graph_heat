#!/bin/bash

echo "Running code quality checks..."

echo "1. Running flake8..."
flake8 *.py

echo -e "\n2. Running mypy..."
mypy *.py

echo -e "\n3. Running pylint..."
pylint *.py

echo -e "\nCode quality check complete!"
