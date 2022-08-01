#!/bin/bash

BASEDIR=$(dirname "$0")
cd "$BASEDIR" || exit

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt > /dev/null

if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v gcc-11 &> /dev/null
    then
        echo "Install gcc-11 first. (brew install gcc@11)"
        exit
    fi
    pip install fast-bleu --install-option="--CC=$(which gcc-11)" --install-option="--CXX=$(which g++-11)"
else
    pip install fast-bleu
fi

pip install git+https://github.com/microsoft/fastseq.git > /dev/null

cd transformers || exit
pip install -e .
cd ..

pip install -U git+https://github.com/pltrdy/pyrouge
git clone https://github.com/pltrdy/files2rouge.git 
cd files2rouge || exit
python setup_rouge.py
python setup.py install > /dev/null

pip uninstall py-rouge
pip install rouge

exit 0