Created: 27-04-2022 15:50
Status: #insight #done
Tags: [[Machine Learning]]

# Graph Convolutional Network
Graph Convolutional Network generalizes [[Convolutional Neural Network]] to arbitrary structure graphs. It reuses the same weight kernel across the whole graph and updates features on nodes while retaining the structure of the graph unchanged.
It is a linear combination of node features in the node neighborhood of the root node with a shared weight matrix $\pmb{W}\in\Re^{D\times D'}$, where each node takes on the role of the root node one after another. Hence it is a multiplication of the input graph matrix $\pmb{f}_{in}\in\Re^{N\times D}$, of $N$ nodes and $D$ node features, with the adjacency matrix $\pmb{A}\in\{0,1\}^{N\times N}$ and the weight matrix $\pmb{W}$. 
The multiplication with the weight matrix $\pmb{W}$ produces partial sums of new features on each node, which are then selectively accumulated with the help of the adjacency matrix $\pmb{A}$.
It is equivalent to sequentially taking 1 node neighborhood at a time, multiplying each node's features with the weight kernel, and accumulating nodes partial sums channel-wise.
^dea5b0

![[Semi-Supervised Classification with Graph Convolutional Networks 1.png]]
## References
1. [[Semi-Supervised Classification with Graph Convolutional Networks]].
2. [[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition]].
3. [[Everything is Connected - Deep Learning on Graphs]].