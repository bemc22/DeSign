# Designed Dithering Sign Activation for Binary Neural Networks
[![DOI:10.1109/JSTSP.2024.3467926](https://zenodo.org/badge/DOI/10.1109/JSTSP.2024.3467926.svg)](https://doi.org/10.1109/JSTSP.2024.3467926)
[![arXiv](https://img.shields.io/badge/arXiv-2405.02220-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2405.02220)

## Abstract

Binary Neural Networks emerged as a cost-effective and energy-efficient solution for computer vision tasks by binarizing either network weights or activations. However, common binary activations, such as the Sign activation function, abruptly binarize the values with a single threshold, losing fine-grained details in the feature outputs. This work proposes an activation that applies multiple thresholds following dithering principles, shifting the Sign activation function for each pixel according to a spatially periodic threshold kernel. Unlike literature methods, the shifting is defined jointly for a set of adjacent pixels, taking advantage of spatial correlations. Experiments over the classification task demonstrate the effectiveness of the designed dithering Sign activation function as an alternative activation for binary neural networks, without increasing the computational cost. Further, DeSign balances the preservation of details with the efficiency of binary operations.


### Implementations

| Method        | Link |
| -----------   | ----------- |
| BNN+DeSign    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bemc22/DeSign/blob/main/BNN/notebooks/demo_train.ipynb)       |
| ReCU+DeSign   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bemc22/DeSign/blob/main/ReCU/notebooks/run_recu.ipynb)         |

### Demo DeSign Activation

| Library        | Link |
| -----------   | ----------- |
| Tensorflow    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bemc22/DeSign/blob/main/BNN/notebooks/demo_activation.ipynb)       |
| Pytorch    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bemc22/DeSign/blob/main/ReCU/notebooks/demo_activation.ipynb)       |



## How to cite
If this code is useful for your and you use it in an academic work, please consider citing this paper as


```bib
@article{monroy2024design,
  author={Monroy, Brayan and Estupinan, Juan and Gelvez-Barrera, Tatiana and Bacca, Jorge and Arguello, Henry},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={Designed Dithering Sign Activation for Binary Neural Networks}, 
  year={2024},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/JSTSP.2024.3467926}
}
```
