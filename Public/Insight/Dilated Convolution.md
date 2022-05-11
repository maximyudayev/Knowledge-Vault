Created: 16-04-2022 07:43
Status: #insight #done
Tags: [[Machine Learning]]

# Dilated Convolution
A dilated convolution learns long-term patterns in sequential data, increasing the [[Receptive Field]] with a dilation of a power of 2 (i.e 1, 2, 4, ..., 512). 

The general form is is used for causal processing:

$$\hat y_t=(f_{in}*_{d}p)[t]=\sum\limits_{i=0}^{k-1}p[i] f_{in_{t-di}}$$

Acausal version is obtained after padding inputs.
## References
1. Bai, S., Kolter, Z. J., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. _arXiv preprint [arXiv:1803.01271](https://arxiv.org/abs/1803.01271)_