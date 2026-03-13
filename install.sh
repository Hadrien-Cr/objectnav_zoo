######################
# Environment setup 
#####################
read -p "Press Enter to continue to Environment Setup" enter
conda env create -n objectnav_zoo -f src/objectnav_zoo/environment.yml
conda activate objectnav_zoo

export CUDA_HOME=$CONDA_PREFIX
export ZOO_ROOT=$(pwd)
export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
export CMAKE_POLICY_VERSION_MINIMUM=3.5

######################
# Package install
#####################
read -p "Press Enter to continue to Package install" enter
read 
python -m pip install -e src/objectnav_zoo

# submodules init
git submodule update --init --recursive src/third_party/habitat-lab 
git submodule update --init --recursive src/third_party/MiDaS 
git submodule update --init --recursive src/third_party/contact_graspnet 

# submodules for depth estimation
git submodule update --init --recursive src/objectnav_zoo/objectnav_zoo/perception/superglue/SuperGluePretrainedNetwork
# submodules for detection 
git submodule update --init --recursive src/objectnav_zoo/objectnav_zoo/perception/detection/detic/Detic
git submodule update --init --recursive src/objectnav_zoo/objectnav_zoo/perception/detection/grounded_sam/Grounded-Segment-Anything

# habitat-lab
pip install -e src/third_party/habitat-lab/habitat-lab -v
# habitat-baselines
pip install -e src/third_party/habitat-lab/habitat-baselines -v

# detectron2 
read -p "Press Enter to continue to detectron2 install" enter
pip install "setuptools<70" -U
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git' -v

# Detic
read -p "Press Enter to continue to Detic install" enter
cd $ZOO_ROOT
git submodule update --init --recursive src/objectnav_zoo/objectnav_zoo/perception/detection/detic/Detic
pip install -r $ZOO_ROOT/src/objectnav_zoo/objectnav_zoo/perception/detection/detic/Detic/requirements.txt

mkdir $ZOO_ROOT/src/objectnav_zoo/objectnav_zoo/perception/detection/detic/Detic/models
if ! [ -f $ZOO_ROOT/src/objectnav_zoo/objectnav_zoo/perception/detection/detic/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth ]; then
    wget --no-check-certificate https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -P $ZOO_ROOT/src/objectnav_zoo/objectnav_zoo/perception/detection/detic/Detic/models/
fi

# MiDaS
read -p "Press Enter to continue to MiDaS install" enter
if ! [ -f $ZOO_ROOT/src/third_party/MiDaS/weights/dpt_beit_large_512.pt ]; then
    wget --no-check-certificate https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt -P $ZOO_ROOT/src/third_party/MiDaS/weights/
fi

# Grounded SAM 
read -p "Press Enter to continue to Grounded SAM install" enter
cd $ZOO_ROOT/src/objectnav_zoo/objectnav_zoo/perception/detection/grounded-sam/Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO --no-build-isolation
pip install --upgrade diffusers[torch]
cd grounded-sam-osx && bash install.sh
python -m pip install opencv-python pycocotools matplotlib ipykernel

mkdir $ZOO_ROOT/src/objectnav_zoo/objectnav_zoo/perception/detection/grounded-sam/Grounded-Segment-Anything/checkpoints/
wget --no-check-certificate -P $ZOO_ROOT/src/objectnav_zoo/objectnav_zoo/perception/detection/grounded-sam/Grounded-Segment-Anything/checkpoints/ https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
wget --no-check-certificate -P $ZOO_ROOT/src/objectnav_zoo/objectnav_zoo/perception/detection/grounded-sam/Grounded-Segment-Anything/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget --no-check-certificate -P $ZOO_ROOT/src/objectnav_zoo/objectnav_zoo/perception/detection/grounded-sam/Grounded-Segment-Anything/checkpoints/ https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
wget --no-check-certificate -P $ZOO_ROOT/src/objectnav_zoo/objectnav_zoo/perception/detection/grounded-sam/Grounded-Segment-Anything/checkpoints/ https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth
mkdir -p $ZOO_ROOT/data/checkpoints

pip install pre-commit
python -m pre_commit install
python -m pre_commit install-hooks
