# Think Before You Grasp: A Multimodal Vision-Language System for Strategic Part Grasping in Clutter
This is the official repository for the paper: **Think Before You Grasp: A Multimodal Vision-Language System for Strategic Part Grasping in Clutter**.


## Setup
###  Installation

- Ubuntu 23.04
- Torch 1.13.1, Torchvision 0.14.1
- Pybullet (simulation environment)
- Cuda 11.8
- GTX 3090*2[if you want to use the complete version]

```

conda create -n thinkgrasp python=3.8
conda activate thinkgrasp

pip install -r requirements.txt
pip install langsam.txt
pip install vlp_requirements.txt


python setup.py develop

cd models/graspnet/pointnet2
python setup.py install

cd ../knn
python setup.py install
```

###  Potential Issues of Installation
- When installing graspnetAPI, the following problem might occur:
```
× python setup.py egg_info did not run successfully.
│ exit code: 1
╰─> [18 lines of output]
The 'sklearn' PyPI package is deprecated, use 'scikit-learn'
rather than 'sklearn' for pip commands.
```
solution:
```
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
```
- Check the compatible version of torch and torchvision of your machine (especially the cuda vision) if the following problem occurs:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
solution: to install torch with the right cuda version, e.g.
```
# CUDA 11.8
pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

if you still face error please try to :
```
sudo apt-get install python3-dev

conda install gxx_linux-64

conda install gcc_linux-64

pip install ray

pip install wandb

pip install https://github.com/IDEA-Research/GroundingDINO/archive/refs/tags/v0.1.0-alpha2.tar.gz

git clone https://github.com/IDEA-Research/GroundingDINO.git

cd GroundingDINO

pip install -e .
```
```
cd langsam

git clone https://github.com/IDEA-Research/GroundingDINO.git

cd GroundingDINO

pip install -e .
```

change the float to float64


install CUDA 11.8 [link](https://drive.google.com/file/d/1weqwQq0Agm5Xp3q7CT_eo8gyvvd_JGAk/view?usp=sharing)
download the file and 
```
sudo bash cuda_11.8.0_520.61.05_linux.run
```

add these code in ~/.bashrc
```
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

```
export PATH=$PATH:/usr/local/cuda-11.8/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
export CUDADIR=/usr/local/cuda-11.8
```

if you want to use VLP
```
cd VLP
wget https://github.com/Cheems-Seminar/grounded-segment-any-parts/releases/download/v1.0/swinbase_part_0a0000.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
need to download pth in som[downloaddata]

### Assets
We provide the processed object models in this [link](https://drive.google.com/drive/folders/10Kyzzhgcnn1WUlQAhUDk9EBmCzk4p-Ar?usp=sharing). Please download the folder and put it in the `assets` folder.

### Pretrained Model by VLG
We provide the pretrained model in this [link](https://drive.google.com/drive/folders/19vsPWWdDoPDuGoRv5tFezaLEtzCPpsgv?usp=sharing). 

## Evaluation VLG
To test the pre-trained model, simply change the location of `--model_path`:

```python
python test.py --load_model True --model_path 'PATH OF YOUR CHECKPOINT FILE'
```

## Citation

If you find this work useful, please consider citing:
