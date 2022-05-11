Created: 15-04-2022 10:52
Status: #insight #done
Tags: [[Machine Learning]]

# Temporal Convolutional Network
TCNs capture long-range temporal dependencies using layered dilated convolutions as an alternative to [[Recurrent Neural Network]]. It learns long-term patterns in sequential data. Unlike RNNs, its architecture is not limited to sequential computations and an infinite [[Receptive Field]]. TCN's receptive field depends on the number of dilated layers. From experience, 10-layer architecture gives best results [1]. Varying the number of layers allows varying the memory size of the model. Compared to other [[Neural Network]] architectures, requires much less memory and easier [[Data Orchestration]].

It uses [[Dilated Convolution]] with an exponentially increasing [[Receptive Field]] by stacking them into layers with an increasing dilation factor as a power of 2 (i.e 1, 2, 4, ..., 512). Causal and acausal versions exist, producing a single output $\hat y_t$ at time $t$ and only using past samples up to $x_t$, or producing an equal length output sequence $\hat Y$ as the length of input sequence $X$, respectively. The latter is often used in sequence models by RNNs for sequence tasks.

In the original form, computes output in a causal fashion with $\hat y_t$ using only past samples, before time $t$.

$$\hat y_t=(f_{in}*_{d}p)[t]=\sum\limits_{i=0}^{k-1}p[i] f_{in_{t-di}}$$

Depending on the implementation, $\hat Y$ sequence is recomputed at each new sample shifted into the input FIFO, acausal [2], 
![[Dilated Temporal Convolution 2.png]]
or is itself also a FIFO, causal [3].
![[Dilated Temporal Convolution 1.png]]

The latter can be used for causal segmentation, where effectively the input sequence $X$ is fed through a sparsely connected neural network with $T$ input neurons and 1 output neuron at $\hat y_t$, exponentially decreasing in hidden layer size.

The former is often used with multiple stages and for learning more complex temporal patterns: in that case, the acausal approach is used that zero-pads inputs and computes the whole $\hat Y$ sequence at each new sample $x$. The final layer then optionally pools, convolves, or manipulates in a diferent way $\hat Y\in\Re^{T\times l}$ if needed to produce a set of class predictions $\hat y_t$ at current time instance $t$, instead of a window of size $T$.

To prevent doing linear combination with 0 activations, falling outside of window $T$ for a given dilated convolution layer, the size of window $T$ must be selected large and proportionate to the depth of the TCN stage.
$T=1+(k-1)\sum\limits_{i=0}^{n}2^{i}=2^{n+1}-1$, for causal and
$T=1$, for acausal networks.
$n$ - depth of the TCN block.

Each TCN layer performs a dilated convolution, activation, $1\times 1$ convolution and a residual connection, before feeding the output to the next layer.
![[MS-TCN Multi-Stage Temporal Convolutional Network for Action Segmentation 1.png]]
## References
1. [[Skeleton-Based Action Segmentation with Multi-Stage Spatial-Temporal Graph Convolutional Neural Networks]]
2. [[MS-TCN Multi-Stage Temporal Convolutional Network for Action Segmentation]]
3. Bai, S., Kolter, Z. J., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. _arXiv preprint [arXiv:1803.01271](https://arxiv.org/abs/1803.01271)._