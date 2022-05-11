Created: 17-04-2022 12:40
Status: #insight #done
Tags: [[Machine Learning]]

# Spatial-Temporal Graph Convolutional Network
ST-GCN extends [[Graph Convolutional Network]] by extending the notion of the node neighborhood to nodes that are connected to the same root node, but also in the former and in the future time frames.
![[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition#^ce4b45]] 

For a well temporally structured graph, one with the same sequence of nodes across time, the label mapping function is: 
![[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition#^d970a4]] 

However, in the mathematical implementation, this mapping of neighborhood to its member nodes is done implicitly by the corresponding adjacency matrix $\pmb{A}$ when multiplied with the spatial-temporal input graph tensor which is represented as:
![[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition 5.png]]

The effect that needs to be achieved is to convolve features of each node in the selected node neighborhood with the shared weight kernel, and to aggregate the results channel-wise. This produces a new feature vector for the root node of the selected node neighborhood.
![[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition 6.png]]

The selection of features associated with the node neighborhood on the graph tensor is done by the adjacency matrix. Here, the convolution operation reuses each of the $C'$ weight kernels across the spatial-temporal node neighborhood, and then sums up the values channel-wise to produce the new feature vector for the root node.

![[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition#Concept]] ^c98285
## References
1. [[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition]].
2. [[Graph Convolutional Network]].
3. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. _arXiv preprint [arxiv:1609.02907](https://arxiv.org/abs/1609.02907)._