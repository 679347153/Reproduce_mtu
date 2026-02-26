In contrast to the original repo, this implementation relies on Habitat-Sim v0.3.0 for data collection. Therefore, setting up a new conda environment is required.

# Guide
1. Collect under **vle_collection** folder.
```sh
cd vle_collection
```
2. Create conda environment.
```sh
conda env create -f environment.yaml -n envname
conda activate envname
conda install habitat-sim==0.3.0 withbullet -c conda-forge -c aihabitat
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab  # reinstall habitat_lab under new env
cd ..
pip install -r requirements.txt
```
3. Link dataset folder.
```sh
mkdir data
ln -s /path/to/your/datasets/ data
```
4. Run batch scripts to collect data. Currently support datasets: **['goat', 'ovon', 'sg3d']**.
```sh
python batch_script/{dataset_name}_batch_data_construct.py
```
# 