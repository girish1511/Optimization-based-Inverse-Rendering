# Optimization-based-Inverse-Rendering
# Advisor: Prof. Xiaohu Guo, University of Texas at Dallas

**Introduction**
* An attempt to reproduce the data generation part of this [paper](https://ieeexplore.ieee.org/abstract/document/8360505).
* The github repository for the aforementioned paper can be found [here](https://github.com/Juyong/3DFace).

**Data**
* As mentioned in the paper the optimization makes use [Basel Face Model(BFM)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads).
* The mean shape and albedo parameters along with the principal components and standard deviations for identity and albedo can be found in BFM.
* The prinicipal components for the expression can be found in the CoarseData dataset which can be downloaded from the author's repository.
* The standard deivation for expression can is given in std_exp.txt
* The inner mouth vertices of the BFM model has been discarded based on [3DDFA](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm). After downloaing the 3DDFA_Release, the indices of the vertices ignored are stored in model_info.mat
