<h2 align="center" width="100%">
Generalizable 3D Gaussian Splatting via Multi-View Stereo and Consistency Constraints
</h2>

## 🌟 Abstract
Recent neural rendering methods still struggle with fine-grained detail reconstruction and scene generalization, 
especially when handling complex geometries and low-texture regions. To address these challenges, 
we propose a 3D Gaussian Splatting (3DGS) framework enhanced by Multi-View Stereo (MVS), 
aiming to improve both rendering quality and cross-scene adaptability. 

## 🔨 Installation
### Clone our repository

  ```
  git clone https://github.com/yangyongjuan/MVS-APFA-GS.git 
  cd MVS-APFA-GS
  ```

### Set up the python environment

  ```
  conda create -n MVS-APFA-GS python=3.7.13
  conda activate MVS-APFA-GS
  pip install -r requirements.txt
  pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 -f https://download.pytorch.org/whl/torch_stable.html
  ```

### Install [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) renderer
  ```
  pip install lib/submodules/diff-gaussian-rasterization
  pip install lib/submodules/simple-knn
  ```

## 📦 Datasets

+ DTU
  ```
  mvs_training
      ├── dtu                   
          ├── Cameras                
          ├── Depths   
          ├── Depths_raw
          └── Rectified
  ```

+ Download [NeRF Synthetic], [Real Forward-facing], and [Tanks and Temples]datasets.


## 🚂 Training
### Train generalizable model

  To train a generalizable model from scratch on DTU, specify ``data_root`` in ``configs/dtu_pretrain.yaml`` first and then run:
  ```
  python train_net.py --cfg_file configs/dtu_pretrain.yaml train.batch_size 4
  ```

### Per-scene optimization
   ```
  bash scripts/llff_ft.sh
  bash scripts/nerf_ft.sh
  bash scripts/tnt_ft.sh
  ```

## 🎯 Evaluation

### Evaluation on DTU

  Use the following command to evaluate the pretrained model on DTU:
  ```
  python run.py --type evaluate --cfg_file configs/dtu_pretrain.yaml mvsgs.cas_config.render_if False,True mvsgs.cas_config.volume_planes 48,8 mvsgs.eval_depth True
  ```
  The rendered images will be saved in ```result/dtu_pretrain```. 

### Evaluation on Real Forward-facing, NeRF Synthetic and Tanks and Temples

  ```
  python run.py --type evaluate --cfg_file configs/llff_eval.yaml
  python run.py --type evaluate --cfg_file configs/nerf_eval.yaml
  python run.py --type evaluate --cfg_file configs/tnt_eval.yaml
  ```


## 📝 Citation
If you find our work useful for your research, please cite our paper.


## 😃 Acknowledgement
This project is built on source codes shared by [MVSGaussian](https://github.com/TQTQliu/MVSGaussian), [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [GaussianPro](https://github.com/kcheng1021/GaussianPro) and [LLFF](https://github.com/Fyusion/LLFF). Many thanks for their excellent contributions!
