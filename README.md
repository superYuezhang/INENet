# INENet: Inliers Estimation Network with Similarity Learning for Partial Overlapping Registration

by Yue Wu, Yue Zhang, Xiaolong Fan, Maoguo Gong, Qiguang Miao, Wenping Ma, and details are in [paper](https://ieeexplore.ieee.org/document/9915616).

## Usage

Clone the repository.

Change the "DATA_DIR" parameter in line 14 of the "data_utils.py" file to its own data set folder path.

Run the "main.py" file.

## Requirement

​	h5py=3.7.0

​	open3d=0.15.2

​	pytorch=1.11.0

​	pytorch3d=0.6.2

​	scikit-learn=1.1.1

​	transforms3d=0.4.1

​	tensorboardX=1.15.0

​	tqdm

​	numpy

## Dataset

​		(1) [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)

​		(2) [3DMatch](https://3dmatch.cs.princeton.edu/)

​		(3) [S3DIS](https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip)

## Citation

If you find the code or trained models useful, please consider citing:

```
@article{2022inenet,
  author={Wu, Yue and Zhang, Yue and Fan, Xiaolong and Gong, Maoguo and Miao, Qiguang and Ma, Wenping},  
  journal={IEEE Transactions on Circuits and Systems for Video Technology},  
  title={INENet: Inliers Estimation Network with Similarity Learning for Partial Overlapping Registration},  
  year={2022}, 
  volume={},  
  number={}, 
  pages={1-1},  
  doi={10.1109/TCSVT.2022.3213592}
}
```

## Acknowledgement

Our code refers to [PointNet](https://github.com/fxia22/pointnet.pytorch), [DCP](https://github.com/WangYueFt/dcp) and [MaskNet](https://github.com/vinits5/masknet). We want to thank the above open-source projects.
