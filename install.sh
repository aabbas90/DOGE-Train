#conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
# conda install pytorch torchvision cudatoolkit=11.3 -c pytorch #to support sm_80 on A100.
# Need python>=3.10
# conda install pytorch==2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pytorch-lightning==2.0.2
pip install torch_geometric
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1%2Bcu118.html

# Only required to generate random problem instances:
conda install -c conda-forge ecole

# pip install pyscipopt
# pip install ecole
pip install tqdm
# conda install h5py

pip install torchmetrics
pip install gurobipy
pip install tensorboardX

cd external/yacs
python setup.py install
cd ..

cd external/BDD
python setup.py install
cd ..