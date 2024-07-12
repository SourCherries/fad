#!/usr/bin/env bash

eval "$(conda shell.bash hook)"

# conda remove -n fad --all
# conda create --name fad conda-forge::dlib "python>=3.9" scikit-image
conda activate fad
# conda install -c conda-forge matplotlib
# conda install tqdm
pip install .
pytest -v src/fad/align/tests/
mother=$(pwd)
cd "$mother/demos/align/1_basic"
python run_demo.py