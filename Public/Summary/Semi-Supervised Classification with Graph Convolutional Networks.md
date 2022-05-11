Created: 25-04-2022 14:47
Status: #summary #done
Tags: [[Machine Learning]] [[Graph Convolutional Network]]

# Semi-Supervised Classification with Graph Convolutional Networks
- Scalable approach for semi-supervised learning on graph-structured data, using CNNs operated directly on graphs.
- The model scales linearly in the number of graph edges that encodes local graph structure and node features.
- The graph structure is directly encoded using an NN.
- The method uses a single weight matrix per layer and deals with varying node degrees through an appropriate normalization of the adjacency matrix.
- Graph-Laplacian regularization methods are limited because they simply encode similarities between nodes.

![[Semi-Supervised Classification with Graph Convolutional Networks 1.png]]

![[Graph Convolutional Network#^dea5b0]] 

However, since $\pmb{A}$ is not normalized, the scale of the feature vectors will change, hence noramlization step must be taken such that each row of $\pmb{A}$ sums to 1. Diagonal node degree matrix $\pmb{D}$ is used for normalization as $\pmb{D}^{-1}\pmb{A}$. Symmetric normalization produces some interesting dynamics, so one arrives at $\pmb{D}^{-\frac{1}{2}}\pmb{A}\pmb{D}^{-\frac{1}{2}}$ instead.

$$\pmb{H}^{(l+1)}=f(\pmb{H}^{(l)},\pmb{A})=\sigma(\pmb{D}^{-\frac{1}{2}}\pmb{A}\pmb{D}^{-\frac{1}{2}}\pmb{H}^{(l)}\pmb{W}^{(l)})$$

## References
1. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. _arXiv preprint [arxiv:1609.02907](https://arxiv.org/abs/1609.02907)._
2. Kipf, T. (2016). _How powerful are Graph Convolutional Networks?_ Github.io. http://tkipf.github.io/graph-convolutional-networks/