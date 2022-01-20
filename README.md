# Optimization-based-Inverse-Rendering
## Advisor: Dr. Xiaohu Guo, University of Texas at Dallas
## Research Internship, Summer 2019

### Introduction
* An attempt to reproduce the data generation part of this [paper](https://ieeexplore.ieee.org/abstract/document/8360505).
* The github repository for the aforementioned paper can be found [here](https://github.com/Juyong/3DFace).

### Data
* As mentioned in the paper the optimization makes use [Basel Face Model(BFM)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads).
* The mean shape and albedo parameters along with the principal components and standard deviations for identity and albedo can be found in BFM.
* The prinicipal components for the expression can be found in the CoarseData dataset which can be downloaded from the author's repository.
* The standard deivation for expression can be found in `std_exp.txt`
* The inner mouth vertices of the BFM model has been discarded based on [3DDFA](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm). The indices of the vertices ignored are stored in `model_info.mat`
* 300-W Dataset is used for the optimization process. The dataset also provides facial landmarks.

### Overview of files in this repository
* There are three implementations of the optimization algorithm.
* `Optimization based Inverse Rendering-PyTorch.ipynb/.py` - Pytorch implementation. The rasterization process alone uses numpy as the pytorch version of it is slow. Since torch tensor is converted to numpy, it gets detached from the graph and therefore torch.backward() cannot be used for calculating gradients and jacobain. To use the pytorch implementation convert all the numpy parts to torch tensor.
* `Optimization based Inverse Rendering-Numpy.ipynb/.py` - Numpy Implementation. Although the rendering and rasterization process is fast the autograd part of the code used to calculate the jacobian for Gauss-Newton method is extremely slow! This is because numpy runs on CPU.
* `Optimization based Inverse Rendering-MXNet.ipynb/.py` - MXNet Implementation. To overcome the hurdle in the numpy implementation, MXNet provides GPU accelerated numpy but doesn't have a jacobian function as in numpy. The code is not complete since the jacobian computation proves to be tricky.

### References
* Guo, Yudong, et al. "CNN-based real-time dense face reconstruction with inverse-rendered photo-realistic face images." IEEE transactions on pattern analysis and machine intelligence 41.6 (2018): 1294-1307.
* Rendering and rasterization code based on [this implemention](https://github.com/YadiraF/face3d)
* Jacobian matrix computation function in pytorch is based on [this implementation](https://github.com/ast0414/adversarial-example/blob/master/craft.py) and in numpy is based on [this implementation](https://stackoverflow.com/questions/49553006/compute-the-jacobian-matrix-in-python)
