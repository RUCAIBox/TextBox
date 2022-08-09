#!/bin/bash

function brewinstall () {
    if [ -x "$(command -v brew)" ];  then
        for pkg in "$@"; do
            brew install $pkg
        done
    else
        echo -e "\033[0;31mFailed to install packages because homebrew not found.\033[0m"
    fi
}

BASEDIR=$(dirname "$0")
F2RDIR=~/.files2rouge/data
cd "$BASEDIR" || exit

echo "
████████╗███████╗██╗  ██╗████████╗██████╗  ██████╗ ██╗  ██╗
╚══██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔═══██╗╚██╗██╔╝
   ██║   █████╗   ╚███╔╝    ██║   ██████╔╝██║   ██║ ╚███╔╝
   ██║   ██╔══╝   ██╔██╗    ██║   ██╔══██╗██║   ██║ ██╔██╗
   ██║   ███████╗██╔╝ ██╗   ██║   ██████╔╝╚██████╔╝██╔╝ ██╗
   ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═════╝  ╚═════╝ ╚═╝  ╚═╝
"

read -p "A modified version of transformers will be installed to python environment. Create a new conda environment (TextBox)? (y/n) " yn

case $yn in
    [yY] ) echo "Creating conda environment named TextBox (python=3.8) ..."
       conda create -n TextBox python=3.8
       conda activate TextBox
       ;;
    [nN] ) ;;
esac

echo "Installation may take a few minutes."
echo -e "\033[0;32mInstalling torch ...\033[0m"
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

echo -e "\033[0;32mInstalling requirements ...\033[0m"
pip install -r requirements.txt

echo -e "\033[0;32mInstalling requirements (fast-bleu) ...\033[0m"
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [ -x "$(command -v gcc-11 &> /dev/null)" ]; then
        brewinstall 'gcc@11'
    fi
    pip install fast-bleu --install-option="--CC=$(which gcc-11)" --install-option="--CXX=$(which g++-11)" > /dev/null
else
    pip install fast-bleu > /dev/null
fi

echo -e "\033[0;32mInstalling requirements (fastseq) ...\033[0m"
pip install git+https://github.com/microsoft/fastseq.git > /dev/null

echo -e "\033[0;32mInstalling requirements (rouge) ...\033[0m"
pip install -U git+https://github.com/pltrdy/pyrouge > /dev/null
git clone https://github.com/pltrdy/files2rouge.git  > /dev/null
cd files2rouge || exit
[ -d "~/.files2rouge" ] && mv ~/.files2rouge ~/.files2rouge.bak\
	&& echo -e "\033[1;33mRenaming ~/.files2rouge to ~/.files2rouge.bak\033[0m"
echo -e '\n' | python setup_rouge.py > /dev/null
python setup.py install > /dev/null
cd ..
F2RExpDIR=$F2RDIR/WordNet-2.0-Exceptions
rm $F2RDIR/WordNet-2.0.exc.db
rm $F2RExpDIR/WordNet-2.0.exc.db
perl $F2RExpDIR/buildExeptionDB.pl $F2RExpDIR exc $F2RExpDIR/WordNet-2.0.exc.db 
ln -s $F2RExpDIR/WordNet-2.0.exc.db $F2RDIR/WordNet-2.0.exc.db
chmod +rx $F2RDIR/WordNet-2.0.exc.db
chmod +rx $F2RExpDIR/WordNet-2.0.exc.db

pip uninstall py-rouge
pip install rouge > /dev/null

echo -e "\033[0;32mInstalling requirements (libxml) ...\033[0m"
if [[ "$OSTYPE" == "darwin"* ]]; then
    brewinstall libxml2 cpanminus
    cpanm --force XML::Parser
else
    if [ -x "$(command -v apt-get)" ];  then sudo apt-get install libxml-parser-perl
    elif [ -x "$(command -v yum)" ];    then sudo yum install -y "perl(XML::LibXML)"
    else echo -e '\033[0;31mFailed to install libxml. See https://github.com/pltrdy/files2rouge/issues/9 for more information.\033[0m' && exit;
    fi
fi

echo -e "\033[0;32mInstalling requirements (transformers) ...\033[0m"
cd transformers || exit
pip install -e . > /dev/null
cd ..

wandb enabled
wandb login

exit 0
