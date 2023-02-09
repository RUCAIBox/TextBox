#!/bin/bash

echo "
████████╗███████╗██╗  ██╗████████╗██████╗  ██████╗ ██╗  ██╗
╚══██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔═══██╗╚██╗██╔╝
   ██║   █████╗   ╚███╔╝    ██║   ██████╔╝██║   ██║ ╚███╔╝
   ██║   ██╔══╝   ██╔██╗    ██║   ██╔══██╗██║   ██║ ██╔██╗
   ██║   ███████╗██╔╝ ██╗   ██║   ██████╔╝╚██████╔╝██╔╝ ██╗
   ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═════╝  ╚═════╝ ╚═╝  ╚═╝
"

echo "Installation may take a few minutes."

if [[ `pip show torch 2> /dev/null` == "" ]]; then
    echo -e "\033[0;32mInstalling torch ...\033[0m"
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
fi

echo -e "\033[0;32mInstalling requirements ...\033[0m"
pip install -r requirements.txt

echo -e "\033[0;32mInstalling requirements (rouge) ...\033[0m"
pip install -U git+https://github.com/pltrdy/pyrouge.git
git clone https://github.com/pltrdy/files2rouge.git 
cd files2rouge
python setup_rouge.py
python setup.py install

pip uninstall py-rouge
pip install rouge