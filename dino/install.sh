#!/bin/bash

set -e

# Clean up conda environment
conda deactivate
conda env remove -n mvfoul -y
conda env create -f environment.yml
conda activate mvfoul

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-source.txt

conda deactivate && conda env remove -n mvfoul -y && conda env create -f environment.yml && conda activate mvfoul && pip install -r requirements.txt && pip install -r requirements-source.txt 