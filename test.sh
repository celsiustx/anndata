#!/bin/bash
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd "$DIR"
virtualenv --python=python3 env
. env/bin/activate
pip install -r requirements.txt
pip install -r requirements_tests.txt
pip install -e .
python -mpytest -vv .
