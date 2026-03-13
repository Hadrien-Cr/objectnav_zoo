objectnav_zoo is a collection of ObjectNav Baselines, with an enphasis on sim, scene representation and online 3D perception

Forked from the repo home-robot.

## Installation

Run [install.sh](install.sh)

## Data Download

### HM3D v0.2/v0.1 (Required by: Mod-IIN)

See https://github.com/matterport/habitat-matterport-3dresearch

### Habitat ImageNav Challenge Dataset (Required by: Mod-IIN)

'''
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v3/instance_imagenav_hm3d_v3.zip
unzip instance_imagenav_hm3d_v3.zip
mkdir -p data/datasets/instance_imagenav/hm3d/v3
mv instance_imagenav_hm3d_v3/* data/datasets/instance_imagenav/hm3d/v3
rm -r instance_imagenav_hm3d_v3 instance_imagenav_hm3d_v3.zip
'''

### Gibson (Required by: SemExp)

Use the invocation python download_mp.py --task_data gibson -o . with the received script to download the data (39.09GB). Matterport3D webpage: link.

## Roadmap
- [ ] Cross-Dataset Evaluation 
- [ ] Automatic Tuning 
- [x] Integrate [Mod-IIN](https://github.com/facebookresearch/home-robot/tree/main/projects/mod_IIN)
- [ ] Integrate [SemExp](https://github.com/devendrachaplot/Object-Goal-Navigation)
- [ ] Integrate [3DAwareNav](https://github.com/jzhzhang/3DAwareNav)
- [ ] Integrate [Modular GOAT - Modular CoW](https://theophilegervet.github.io/projects/goat/)
- [ ] Integrate [Concept Graphs](https://concept-graphs.github.io/)
- [ ] Integrate [MTU3D](https://github.com/MTU3D/MTU3D)
- [ ] Integrate [3DMem](https://github.com/UMass-Embodied-AGI/3D-Mem)
