#!/bin/bash

BASEDIR=$(dirname "$0")
cd "$BASEDIR" || exit

echo "
████████╗███████╗██╗  ██╗████████╗██████╗  ██████╗ ██╗  ██╗
╚══██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔═══██╗╚██╗██╔╝
   ██║   █████╗   ╚███╔╝    ██║   ██████╔╝██║   ██║ ╚███╔╝
   ██║   ██╔══╝   ██╔██╗    ██║   ██╔══██╗██║   ██║ ██╔██╗
   ██║   ███████╗██╔╝ ██╗   ██║   ██████╔╝╚██████╔╝██╔╝ ██╗
   ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═════╝  ╚═════╝ ╚═╝  ╚═╝

"

echo "Installing torch ..."
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch > /dev/null

echo "Installing requirements ..."
pip install -r requirements.txt > /dev/null

if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v gcc-11 &> /dev/null
    then
        echo "Please install gcc-11 first. (brew install gcc@11)"
        exit
    fi
    pip install fast-bleu --install-option="--CC=$(which gcc-11)" --install-option="--CXX=$(which g++-11)" > /dev/null
else
    pip install fast-bleu > /dev/null
fi

pip install git+https://github.com/microsoft/fastseq.git > /dev/null

pip install -U git+https://github.com/pltrdy/pyrouge > /dev/null
git clone https://github.com/pltrdy/files2rouge.git  > /dev/null
cd files2rouge || exit
python setup_rouge.py > /dev/null
python setup.py install > /dev/null

pip uninstall py-rouge
pip install rouge

echo "Installing transformers (modified) ..."
cd transformers || exit
pip install -e .
cd ..

wandb login

exit 0
