conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt 

cd transformers
pip install -e .
cd ..

pip install -U git+https://github.com/pltrdy/pyrouge
git clone https://github.com/pltrdy/files2rouge.git 
cd files2rouge
python setup_rouge.py
python setup.py install

pip uninstall rouge
pip install py-rouge
