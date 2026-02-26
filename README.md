## Move to Understand a 3D Scene: Bridging Visual Grounding and Exploration for Efficient and Versatile Embodied Navigation

<p align="left">
    <a href='https://www.arxiv.org/abs/2507.04047'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://www.arxiv.org/abs/2507.04047'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://mtu3d.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <!-- <a href='https://huggingface.co/spaces/li-qing/PQ3D-Demo'> -->
      <!-- <img src='https://img.shields.io/badge/Demo-HuggingFace-yellow?style=plastic&logo=AirPlay%20Video&logoColor=yellow' alt='HuggigFace'> -->
    <!-- </a> -->
    <a href='https://huggingface.co/bigai/MTU3D'>
      <img src='https://img.shields.io/badge/Model-Checkpoints-orange?style=plastic&logo=Google%20Drive&logoColor=orange' alt='Checkpoints'>
    </a>
</p>

[Ziyu Zhu](https://zhuziyu-edward.github.io/), 
[Xilin Wang](),
[Yixuan Li](),
[Zhuofan Zhang](https://tongclass.ac.cn/author/zhuofan-zhang/), 
[Xiaojian Ma](https://jeasinema.github.io/), 
[Yixin Chen](https://yixchen.github.io/),
[Baoxiong Jia](https://buzz-beater.github.io),
[Wei Liang](),
[Qian Yu](),
[Zhidong Deng](https://www.cs.tsinghua.edu.cn/csen/info/1165/4052.htm)📧,
[Siyuan Huang](https://siyuanhuang.com/)📧,
[Qing Li](https://liqing-ustc.github.io/)📧

This repository is the official implementation of the Arxiv paper "Move to Understand a 3D Scene: Briding Visual Grounding and Exploration for Efficient and Versatile Embodied Navigation".

[Paper](https://www.arxiv.org/abs/2507.04047) |
[arXiv](https://www.arxiv.org/abs/2507.04047) |
[Project](https://mtu3d.github.io) |
[Checkpoints](https://huggingface.co/bigai/MTU3D)

<div align=center>
<img src='https://mtu3d.github.io/mtu3d-teaser.png' width=100%>
</div>

### News
- [ 2025.07 ] Release training and evaluation.
- [ 2025.07 ] Release data and checkpoints.
- [ 2025.08 ] Release data collection scripts.
<!-- - [ 2024.07 ] Our huggingface DEMO is here [DEMO](https://huggingface.co/spaces/li-qing/PQ3D-Demo), welcome to try our model!
- [ 2024.07 ] Release codes of model! TODO: Clean up training and evaluation -->

### Abstract

 Embodied scene understanding requires not only comprehending visual-spatial information that has been observed but also determining where to explore next in the 3D physical world. 
  Existing 3D Vision-Language (3D-VL) models primarily focus on grounding objects in static observations from 3D reconstruction, such as meshes and point clouds, but lack the ability to actively perceive and explore their environment. To address this limitation, we introduce **M**ove **t**o **U**nderstand (**MTU3D**), a unified framework that integrates active perception with 3D vision-language learning, enabling embodied agents to effectively explore and understand their environment. . Extensive evaluations across various embodied navigation and question-answering benchmarks show that MTU3D outperforms state-of-the-art reinforcement learning and modular navigation approaches by 14\%, 27\%, 11\%, and 3\% in success rate on HM3D-OVON, GOAT-Bench, SG3D, and A-EQA, respectively. MTU3D's versatility enables navigation using diverse input modalities, including categories, language descriptions, and reference images. The deployment on a real robot demonstrates MTU3D's effectiveness in handling real-world data. These findings highlight the importance of bridging visual grounding and exploration for embodied intelligence.

### Install
1. Install conda package
```
conda create -n envname python=3.8
conda activate envname
pip3 install torch==2.0.0
pip3 install torchvision==0.15.1
python3 -m pip install nvidia-cudnn-cu11==8.7.0.84
pip3 install -r requirements.txt
```

2. Install Minkowski Engine
```
git clone https://github.com/NVIDIA/MinkowskiEngine.git
sudo apt install python3-distutils
conda install openblas-devel -c anaconda
cd MinkowskiEngine
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

3. Install FastSAM, link is here [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM.git)

put checkpoint to `./hm3d-online/FastSAM/FastSAM-x.pt`

```
cd hm3d-online
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
cd FastSAM
pip install -r requirements.txt
cd ../..
```

4. Install HabitatSim and HabitatLab
```
conda install habitat-sim=0.2.3 headless -c conda-forge -c aihabitat -y
git clone --branch v0.2.3 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
```

### Prepare data
1. download sceneverse data from [scene_verse_base](https://github.com/scene-verse/sceneverse?tab=readme-ov-file) and change `data.scene_verse_base` to sceneverse data directory.
2. download stage1 data for embodied segmentation training from [stage1](https://huggingface.co/datasets/bigai/MTU3D) and change `data.embodied_base` to download data directory.
3. download feature saved from stage1 from [stage1_feat](https://huggingface.co/datasets/bigai/MTU3D) and change `data.embodied_feat` to download data directory.
4. download vle data from [vle_stage2](https://huggingface.co/datasets/bigai/MTU3D) and change `data.embodied_vle` to download data directory.
5. change `embodied_scan_dir` in hm3d-online/*-nav.py to stage1 data directory.
6. download hm3d data from [hm3d](https://aihabitat.org/datasets/hm3d/) and change `hm3d_data_base_path` in hm3d-online/*.-nav.py.
7. download embodied navigation benchmark data from [embodied-bench](https://huggingface.co/datasets/bigai/MTU3D) and change `data_set_path` and `navigation_data_path` in hm3d-online/*.nav.py.


### Prepare checkpoints
1. download [mtu3d-ckeckpoints](https://huggingface.co/bigai/MTU3D), and change `pq3d_stage1_path` and `pq3d_stage2_path` in hm3d-online/*-nav.py.


### Run MTU3D for training
Stage 1 low-level percetpion training
```
python3 run.py --config-path configs/embodied-pq3d-final --config-name embodied_scan_instseg.yaml
```
Stage 2 vision-langauge-exploration pre-training
```
python3 run.py --config-path configs/embodied-pq3d-final --config-name embodied_vle.yaml 
```
Stage 3 navigation dataset specific fine-tuning
```
python3 run.py --config-path configs/embodied-pq3d-final --config-name embodied_vle.yaml data.train=[{specific_dataset}] pretrain_ckpt_path={stage2_pretrained_path}
```
For multi-gpu training usage, we use four GPU in our experiments.
```
python launch.py --mode ${launch_mode} \
    --qos=${qos} --partition=${partition} --gpu_per_node=4 --port=29512 --mem_per_gpu=80 \
    --config {config}  \
```
To debug, use
```
python3 ... debug.flag=True debug.debug_size=10
```

### Run MTU3D for evaluation
```
mkdir output_dirs
export PYTHONPATH=./:./hm3d-online:./hm3d-online/FastSAM
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export YOLO_VERBOSE=False
```
#### Evaluation for HM3D-ovon
Change path in hm3d-nav.py.
Edit run_nav.sh.
```
bash run_nav.sh
```

#### Evaluation for Goat-bench
Change path in goat-nav.py.
Edit run_nav.sh.
```
bash run_nav.sh
```

#### Evaluation for SG3D
Change path in sg3d-nav.py.
Edit run_nav.sh.
```
bash run_nav.sh
```

### Data Collection
We provide data collection scripts in **vle_collection** folder.

### Acknowledgement
We would like to thank the authors of [Vil3dref](https://github.com/cshizhe/vil3dref), [Mask3d](https://github.com/JonasSchult/Mask3D), [Openscene](https://github.com/pengsongyou/openscene), [Xdecoder](https://github.com/microsoft/X-Decoder), and [3D-VisTA](https://github.com/3d-vista/3D-VisTA) for their open-source release.


### Citation:
```
@article{zhu2025mtu,
  title = {Move to Understand a 3D Scene: Bridging Visual Grounding and Exploration for Efficient and Versatile Embodied Navigation},
  author = {Zhu, Ziyu and Wang, Xilin and Li, Yixuan and Zhang, Zhuofan and Ma, Xiaojian and Chen, Yixin and Jia, Baoxiong and Liang, Wei and Yu, Qian and Deng, Zhidong and Huang, Siyuan and Li, Qing},
  journal = {International Conference on Computer Vision (ICCV)},
  year = {2025}  
}
```
