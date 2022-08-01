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

read -p "A modified version of transformers will be installed to python environment. Create a new conda environment? (y/n) " yn

case $yn in
	[yY] ) echo "Creating conda environment (python=3.8) ..."
	       conda create -n TextBox python=3.8
	       break;;
        [nN] ) break;;
esac

echo "Installation may take a few minutes."
echo "Installing torch ..."
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch > /dev/null

echo "Installing requirements ..."
pip install -r requirements.txt > /dev/null

echo "Installing requirements (fast-bleu) ..."
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

echo "Installing requirements (fastseq) ..."
pip install git+https://github.com/microsoft/fastseq.git > /dev/null

echo "Installing requirements (rouge) ..."
pip install -U git+https://github.com/pltrdy/pyrouge > /dev/null
git clone https://github.com/pltrdy/files2rouge.git  > /dev/null
cd files2rouge || exit
echo -e '\n' | python setup_rouge.py > /dev/null
python setup.py install > /dev/null
cd ..

pip uninstall py-rouge
pip install rouge

echo "Installing requirements (transformers) ..."
cd transformers || exit
pip install -e . > /dev/null
cd ..

wandb login

exit 0
