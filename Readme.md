# Codes for "Fine-grained Analysis of Stability and Generalization for Stochastic Bilevel Optimization" accepted by IJCAI 2024.

# Code & Data Acknowledgement

## Hyperparameter Optimization

As for hyperparameter optimization task, we employed the codes from [1] (https://githubfast.com/baofff/stability_ho) with several modifications (e.g., remove the re-initialization operation).

## Meta Learning

(2) As for meta learning task, we employed the codes from [2] (https://githubfast.com/JunjieYang97/stocBiO) .



# Data Source

# MNIST data set for data cleaning 

MNIST data [3] can be downloaded from the Python library "torch.utils.data"

# Omniglot data set for one-shot learning

Omniglot data [4] can be downloaded from https://githubfast.com/brendenlake/omniglot



## Reference

[1] Stability and generalization of bilevel programming in hyperparameter optimization

```
@article{bao2021stability,
  title={Stability and generalization of bilevel programming in hyperparameter optimization},
  author={Bao, Fan and Wu, Guoqiang and Li, Chongxuan and Zhu, Jun and Zhang, Bo},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={4529--4541},
  year={2021}}
```

[2] Bilevel Optimization: Nonasymptotic Analysis and Faster Algorithms

```
@inproceedings{ji2021bilevel,
	author = {Ji, Kaiyi and Yang, Junjie and Liang, Yingbin},
	title = {Bilevel Optimization: Nonasymptotic Analysis and Faster Algorithms},
	booktitle={International Conference on Machine Learning (ICML)},
	year = {2021}}
```

[3]  MNIST dataset

```
@article{lecun1998mnist,
  title={The MNIST database of handwritten digits},
  author={LeCun, Yann},
  journal={http://yann. lecun. com/exdb/mnist/},
  year={1998}}
```

[4]  Omniglot dataset

```
@article{
author = {Brenden M. Lake  and Ruslan Salakhutdinov  and Joshua B. Tenenbaum },
title = {Human-level concept learning through probabilistic program induction},
journal = {Science},
volume = {350},
number = {6266},
pages = {1332-1338},
year = {2015}}
```

## If you are interested in this work, please refer to https://www.ijcai.org/proceedings/2024/609 
## and cite as

@inproceedings{zhang2024genbo,
  title     = {Fine-grained Analysis of Stability and Generalization for Stochastic Bilevel Optimization},
  author    = {Zhang, Xuelin and Chen, Hong and Gu, Bin and Gong, Tieliang and Zheng, Feng},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}}
  pages     = {5508--5516},
  year      = {2024}
}




