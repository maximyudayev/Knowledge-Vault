Created: 25-04-2022 14:47
Status: #summary 
Tags: [[Machine Learning]] [[Graph Convolutional Network]]

# Semi-Supervised Classification with Graph Convolutional Networks
- Scalable approach for semi-supervised learning on graph-structured data, using CNNs operated directly on graphs.
- The model scales linearly in the number of graph edges that encodes local graph structure and node features.
- The graph structure is directly encoded using an NN.
- The method uses a single weight matrix per layer and deals with varying node degrees through an appropriate normalization of the adjacency matrix.
- Graph-Laplacian regularization methods are limited because they simply encode similarities between nodes.
## References
1. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. _arXiv preprint [arxiv:1609.02907](https://arxiv.org/abs/1609.02907)._