Created: 15-04-2022 11:45
Status: #summary #todo
Tags: [[Temporal Convolutional Network]] [[Freezing of Gait]] [[Graph Convolutional Network]] [[Machine Learning]] [[Action Segmentation]]

# Skeleton-Based Action Segmentation with Multi-Stage Spatial-Temporal Graph Convolutional Neural Networks
- First stage is ST-GCN that approprietly models spatial hierarchy across joints.
- Combines [[Spatial-Temporal Graph Convolutional Network]] and Multi-Stage [[Temporal Convolutional Network]].
- Spatial graph convolutions learn spatial patterns.
- Dilated temporal convolutions learn long-term temporal patterns.
- Multiple TCN stages refine original predictions and learn higher-order sequence of actions defining more complex actions.

## Architecture
$f_{in}$ - input data
$f_{adj}$ - remapped input to network feature map size
$f_{gcn}$ - output of the ST-GCN stage
$\hat{Y}$ - output sequence

$T$ - number of samples
$N$ - number of joints
$C_{in}$ - number of input channels per joint (IMU data)
$C$ - number of feature channels in the neural network
$l$ - number of segmentation classes
### Input Remapping
Each $\Re^{1\times N\times C_{in}}$ sample is passed through [[Batch Normalization]] and then mapped to the network's feature map dimensions, $\Re^{1\times N\times C}$, after sampling and is pushed into a $T$ sized FIFO buffer. Mapping is a 1x1 2D Convolution (or a fully-connected layer applied across $N$ and $T$ dimensions of the input tensor) with $W_{in}\in\Re^{1\times 1\times C_{in}\times C}$ and $b\in\Re^C$.

$$f_{in}\in\Re^{T\times N\times C_{in}} \xrightarrow{1x1\quad Conv} f_{adj}\in\Re^{T\times N\times C}$$

Total \#MAC: $N\times C_{in}\times C+C$ (per sample).
Total \#params: $C_{in}\times C+C$.
### Graph Convolution
[[Spatial-Temporal Graph Convolutional Network]] learns a representation on a graph using: 
- the set of graph nodes $V$ over $N$ skeleton joints, across $T$ time samples.
- two sets of edges across these nodes, one for spatial connection between nodes across the same time sample, and other for temporal connection between the same node across time.
- an adjacency matrix, describing the graph structure.

$$f_{gcn}(v_{ti})=\sum\limits_{v_{tj}\in B(v_{ti})}\frac{1}{Z_{ti}(v_{tj})}f_{in}(v_{tj})w(l_{ti}(v_{tj}))$$

$v_{tj}\in B(v_{ti})$ - nodes within sampling area of the node $v_{ti}$
$l_{ti}$ - function that maps each node to a unique weight vector $w$
$Z_{ti}$ - normalizing term across graph partitions

$$f_{gcn}=\sum\limits_{p} A_{p}f_{adj}W_{p}M_{p}$$


$W_{p}\in\Re^{1\times 1\times C\times C}$ - weight matrix
$A_{p}\in \{0,1\}^{N\times N}$ - adjacency matrix of spatial connections between joints
$p$ - graph partition
$D_{p}$ - diagonal node degree matrix

$D_{p}^{-\frac{1}{2}}A_{p}D_{p}^{-\frac{1}{2}}$ symmetrically normalizes $A_{p}$.
### Output
$\hat{Y}\in\Re^{T\times l}$ sequence of probabilities of classes. In the case of MS-GCN, $l=2$, to classify only FOG.


## Notes
Window $T$ must be selected wide enough to accomodate for depth of the dilated convolutions
## References
1. Filtjens, B., Vanrumste, B., & Slaets, P. (2022). Skeleton-Based Action Segmentation with Multi-Stage Spatial-Temporal Graph Convolutional Neural Networks. _arXiv preprint [arxiv:2202.01727](https://arxiv.org/abs/2202.01727)._
2. [[Spatial-Temporal Graph Convolutional Network]]
3. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. _arXiv preprint [arxiv:1609.02907](https://arxiv.org/abs/1609.02907)._