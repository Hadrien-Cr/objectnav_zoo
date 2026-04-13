```
export PYTHONNOUSERSITE=1
export HABITAT_DATA=<path/to/habitat/data>
conda env create -n objectnav_zoo -f environment.yml

ln -s $HABITAT_DATA . # create a symlink from the data folder to the root

export CUDA_HOME=$CONDA_PREFIX
export ZOO_ROOT=$(pwd)
```

```
# Prepare Submodules
git submodule update --init --recursive third_party/habitat-lab # uses v0.2.5
git submodule update --init --recursive third_party/MiDaS 
git submodule update --init --recursive third_party/SuperGluePretrainedNetwork
git submodule update --init --recursive third_party/Detic
git submodule update --init --recursive third_party/Grounded-Segment-Anything

# Habitat-Lab
pip install -e third_party/habitat-lab/habitat-lab -v
pip install -e third_party/habitat-lab/habitat-baselines -v

# Detic
pip install -r third_party/Detic/requirements.txt

# detectron2
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git' -v

# Grounded-SAM
cd $ZOO_ROOT/third_party/Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO --no-build-isolation
cd $ZOO_ROOT

# requirements
pip install -e .
pip install -r requirements.txt 
```

```
# Download checkpoints for Detic
mkdir $ZOO_ROOT/third_party/Detic/models
wget --no-check-certificate https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -P $ZOO_ROOT/third_party/Detic/models/

# Download checkpoints for MiDaS
wget --no-check-certificate https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt -P $ZOO_ROOT/third_party/MiDaS/weights/


# Download checkpoints for Grounded-SAM
mkdir $ZOO_ROOT/third_party/Grounded-Segment-Anything/checkpoints/
wget --no-check-certificate -P $ZOO_ROOT/third_party/Grounded-Segment-Anything/checkpoints/ https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
wget --no-check-certificate -P $ZOO_ROOT/third_party/Grounded-Segment-Anything/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget --no-check-certificate -P $ZOO_ROOT/third_party/Grounded-Segment-Anything/checkpoints/ https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
wget --no-check-certificate -P $ZOO_ROOT/third_party/Grounded-Segment-Anything/checkpoints/ https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
mkdir -p $ZOO_ROOT/data/checkpoints
```

```
# Download ProcTHOR-HAB scene data & ObjectNav Dataset
git clone git@hf.co:datasets/hssd/ai2thor-hab $HABITAT_DATA/scene_datasets/ai2thor-hab
# go to https://www.dropbox.com/scl/fi/noizniosf3sjaolq54a6v/objectnav_procthor-hab_v0.2.5.zip?rlkey=g5q3mmpin8fqu66jqrkfqsy6l&dl=0 
# and paste to $HABITAT_DATA/datasets/objectnav/procthor-hab
unzip $HABITAT_DATA/datasets/objectnav/procthor-hab/objectnav_procthor-hab_v0.2.5.zip -d $HABITAT_DATA/datasets/objectnav/procthor-hab
```

```
# Download HSSD scene data & ObjectNav Dataset
git clone git@hf.co:datasets/hssd/hssd-hab $HABITAT_DATA/scene_datasets/hssd-hab
# go to https://www.dropbox.com/scl/fi/n5m00eoydfedi0de1nh34/objectnav_hssd-hab_v0.2.5.zip?rlkey=zvosfrxu99si8xkfmkd623b4s&dl=0
# and paste to $HABITAT_DATA/datasets/objectnav/procthor-hab
unzip $HABITAT_DATA/datasets/objectnav/hssd-hab/objectnav_hssd-hab_v0.2.5.zip -d $HABITAT_DATA/datasets/objectnav/procthor-hab
```

```
# Download MP3D scene data & PointNav Dataset
python download_mp.py --task habitat -o $HABITAT_DATA/data/scene_datasets/mp3d_data
unzip $HABITAT_DATA/scene_datasets/mp3d_habitat.zip  -d $HABITAT_DATA/scene_datasets/
wget http://dl.fbaipublicfiles.com/habitat/mp3d/config_v1/mp3d.scene_dataset_config.json -P $HABITAT_DATA/scene_datasets/mp3d
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip -P $HABITAT_DATA/datasets/pointnav/mp3d/v1
unzip $HABITAT_DATA/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip -d third_party/habitat-lab/datasets/pointnav/mp3d/v1
```

```
# Download HM3D scene data & InstanceNav Dataset
# See https://github.com/matterport/habitat-matterport-3dresearch for scene datasets
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v3/instance_imagenav_hm3d_v3.zip
unzip instance_imagenav_hm3d_v3.zip
mkdir -p $HABITAT_DATA/datasets/instance_imagenav/hm3d/v3
mv instance_imagenav_hm3d_v3/* $HABITAT_DATA/datasets/instance_imagenav/hm3d/v3
rm -r instance_imagenav_hm3d_v3 instance_imagenav_hm3d_v3.zip
```

```
# Download Gibson & ObjectNav Dataset
download_mp.py --task_data gibson -o .
```
