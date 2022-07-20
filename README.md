<p align='center'>
  <b>
    <a href="https://arxiv.org/abs/2204.13952">ArXiv</a>
    | 
    <a href="#Get-Start">Get Start</a>
  </b>
</p> 


# Deep-Geometry-Post-Processing

The source code for our paper "[Deep Geometry Post-Processing for Decompressed Point Clouds](https://arxiv.org/abs/2204.13952)" (ICME2022 oral)

We propose a novel learning-based post-processing method to enhance the decompressed point clouds. Our model is able to significantly improve the geometry quality of the decompressed point clouds by predicting the occupancy probability of each voxel.

* **Display:**

<p align='center'>  
  <img src='https://user-images.githubusercontent.com/47820962/179795330-181c914f-d764-4f5e-bb95-a94a51ee5e74.png' width='700'/>
</p>
<p align='center'> 
  <b>Left:</b> the ground truth point clouds; <b>Middle:</b> the decompressed point clouds obtained by G-PCC; <b>Right:</b> the refined point clouds obtained by our model.
</p>

Experimental results show that the proposed method can significantly improve the quality of the decompressed point clouds, achieving <b>9.30dB BDPSNR gain</b> on three representative datasets on average.


## Get Start

### 1) Installation

**Requirements**

* Python 3
* pytorch (1.7.1)
* CUDA

**Conda installation**

```bash
# 1. Create a conda virtual environment.
conda create -n torch17 python=3.6
source activate torch17
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2

# 2. Install dependency
pip install -r requirement.txt
```


### 2) Running

#### Generating training dataset

The longdress and loot sequences in the [8iVFB](http://plenodb.jpeg.org/pc/8ilabs/) dataset are used for training. We randomly select 60 frames from the two sequences to construct the training set. The latest version of [MPEG-TMC13 (V14.0)](https://github.com/MPEGGroup/mpeg-pcc-tmc13) is used to obtain the decompressed point clouds at different bit rates.

The decoded point clouds are first divided into small non-overlapped cubes.

``` bash
python util/split_point_cloud.py --source_dir=your_path_of_decoded_point_clouds --save_dir=./traindata
```

#### Training

``` bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 train.py \
--config ./config/multi_scale.yaml \
--name baseline
```

We provide **the pre-trained weights**  of our model **[here](https://drive.google.com/drive/folders/1e20NiP1eyWkDnaxp4lZ6JI8Ym_f5nYxE?usp=sharing)**.

#### Testing

We evaluate the performance of our proposed model in the [8iVFB](http://plenodb.jpeg.org/pc/8ilabs/) dataset, [MVUB](http://plenodb.jpeg.org/pc/microsoft/) dataset, and [ODHM](https://mpeg-pcc.org/index.php/pcc-content-database/owlii-dynamic-human-textured-mesh-sequence-dataset/) dataset. Except for the two training point cloud sequences, the other sequences of the above three datasets are used for testing.

``` bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 6789 test.py \
--config ./config/multi_scale_all.yaml \
--name baseline \
--test_list ./test/example.txt
```
The geometry refined point clouds will be saved in the eval_result folder in default.

## Citation

```tex
@article{fan2022deep,
  title={Deep Geometry Post-Processing for Decompressed Point Clouds},
  author={Fan, Xiaoqing and Li, Ge and Li, Dingquan and Ren, Yurui and Gao, Wei and Li, Thomas H},
  journal={arXiv preprint arXiv:2204.13952},
  year={2022}
}
```

## Acknowledgement 

Some dataset preprocessing methods are derived from [PCGCv1](https://github.com/NJUVISION/PCGCv1).
