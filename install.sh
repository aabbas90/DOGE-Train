#conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
#conda install pytorch torchvision cudatoolkit=11.3 -c pytorch #to support sm_80 on A100.
conda install pyg -c pyg -c conda-forge

pip install pytorch-lightning==1.5.10
conda install -c conda-forge ecole pyscipopt
conda install -c conda-forge tqdm
# conda install h5py

conda install -c conda-forge torchmetrics
