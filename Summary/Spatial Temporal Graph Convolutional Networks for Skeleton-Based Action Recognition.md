Created: 17-04-2022 12:57
Status: #summary #todo
Tags: [[Graph Convolutional Network]] [[Machine Learning]] [[Action Segmentation]]

# Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
- [[Graph Convolutional Network]] generalizes [[Convolutional Neural Network]] to arbitrary structure graphs. Modeling dynamic graphs still needs to be explored.
- In spatial perspective, convolution is applied directly on the graph nodes and its neighbors; in spectral perspective, locality of convolution is considered in spectral analysis.
- It learns body part information implicitly through locality of graph convolutions and temporal dynamics.
- Since no manual part assignment is done, model is potent to learn better action representation.
- Modeled as time-series of 2D or 3D joint locations.
- Human actions inferred from spatial-temporal patterns.
- Multiple layers are stacked to integrate data along spatial and temporal dimensions.
- Joints move as parts of body parts, making the movements spatially constrained with respect to these relationships in local regions.
- Stacking multiple GCN layers generates higher-level feature maps.
- Spatial temporal graph is used to form a hierarchical representation of the skeleton sequences.
- 9 GCN layers: 3 of 64 channels, 3 of 128 channels, and 3 of 256 channels. All with temporal kernel size of 9, a residual connection, and 0.5 dropout probability.
- 4th and 7th temporal convolutional layers have stride of 2 for pooling.
- Global pooling at the end is done to get 256 feature vector for the input sequence and fed through the SoftMax.
- Weight matrix in GCN is shared among nodes: varying node degrees are dealt with by appropriate normalization of the adjacency matrix.
## Skeleton Graph Construction
Graph is comprised of a set of nodes $V=\{v_{ti}|t=1,...,T;i=1,...,N\}$, which are interconnected with a set of edges $E$, split across 2 subsets: $E_{S}=\{v_{ti}v_{tj}|(i,j)\in H\}$ - intra-frame skeleton connections, $E_{F}=\{v_{ti}v_{t+1)i}\}$ - inter-frame same joint connections. $E_{F}$ represents trajectory of a joint $i$ over time.
$F(v_{ti})$ is the feature vector of the node $v_{i}$ in time frame $t$, along with the confidence estimate.
Graph nodes are connected within each frame according to the skeleton structure. The same joint is connected to itself across all the time frames.
## Spatial Graph Convolutional Neural Network
Sampling function $p$ spits out a node in the neighborhood of $v_{ti}$ that's a member of the set $B(v_{ti})=\{v_{tj}|d(v_{ti},v_{tj})\leq D\}$, where $d(v_{ti},v_{tj})$ is the shortests distance between 2 nodes and $D$ is set to 1 in this work.

Weight function $w$ orders the weights according to the partitioning strategy of neighbor set $B(v_{ti})$ into $K$ subsets, with each of $K$ subsets having a label. $l_{ti}:B(v_{ti})\rightarrow\{0,...,K-1\}$ maps a node in a neighborhood to its subset label. The weight function is then $w(v_{ti},v_{tj})=w'(l_{ti}(v_{tj})): B(v_{ti})\rightarrow\Re^c$.

The spatial graph convolution is then $f_{out}(v_{ti})=\sum\limits_{v_{tj}\in B(v_{ti})}\frac{1}{Z_{ti}(v_{tj})}f_{in}(p(v_{ti},v_{tj}))w(v_{ti},v_{tj})$, where $Z_{ti}(v_{tj})=|\{v_{tk}|l_{ti}(v_{tk})=l_{ti}(v_{tj})\}|$ is the cardinality of the subset that balances the contributions of different subsets to the output.
## Spatial Temporal Modeling
Modeling spatial temporal dynamics in the sequence is extended from the spatial graph convolution by extending the concept of neighborhood, including now the temporally connected joints $B(v_{ti})=\{v_{qj}|d(v_{ti},v_{tj})\leq K; |q-t|\leq \lfloor\Gamma/2\rfloor\}$,  where $\Gamma$ is the temporal kernel size that controls the temporal range to be included in the neighborhood. $l_{ST}(v_{qj})=l_{ti}(v_{tj})+(q-t+\lfloor\Gamma/2\rfloor)\times K$ is the well-defined label mapping function for the spatial temporal graph convolution.
## Partitioning Strategies
Label mapping depends on the partitioning strategy.
### Uni-labeling
The subset is the while neighbor set itself. Inner product between each node's feature vector and the weight vector.
$K=1$; $l_{ti}(v_{tj})=0,\forall i,j\in V$.
### Distance
Partitioning based on the shortest distance $d(\cdot, v_{ti})$ to node $v_{ti}$, the root. If $D=1$ only immediate connections of $v_{ti}$ are in the subset, of which there are 2 (and hence 2 weight vectors): for $d=0$, node $v_{ti}$ itself, and for $d=1$, direct neighbors.
$K=2$; $l_{ti}(v_{tj})=d(v_{tj}, v_{ti})$.
### Spatial Configuration
Neighbor set is split into 3 subsets: root node itself, nodes closer to the center of gravity of the skeleton than the root node, and nodes farther away from the center of gravity of the skeleton than the root node.
$$l_{ti}(v_{tj})=
\begin{cases}
  0 & \textbf{if $r_{j}=r_{i}$}\\
  1 & \textbf{if $r_{j}\lt r_{i}$}\\
  2 & \textbf{if $r_{j}\gt r_{i}$}\\
\end{cases}$$
$r_{i}$ is the average distance from the center of gravity to joint $i$ across all time frames of the training set.
A more advanced partitioning scheme is expected to lead to better performance.
### Edge Importance Weighting
Since a joint can be a member of multiple body parts, it needs to have different importance weightings in those body parts, with respect to the movement dynamics. A mask $M$ is added to each spatial temporal graph convolution layer to scale the contribution of a joint's feature on its neighbors based on the learned weighting of the spatial edges in $E_{S}$.
### Implementation
Uni-labeling, single-frame ST-GCN can be implemented as:
$\pmb{f}_{out}=\pmb{\Lambda}^{-\frac{1}{2}}(\pmb{A}+\pmb{I})\pmb{\Lambda}^{-\frac{1}{2}}\pmb{f}_{in}\pmb{W}$

$\pmb{A}$ - adjacency matrix of intra-frame connections
$\pmb{I}$ - identity matrix of intra-frame self-connections
$\pmb{W}$ - weight matrix of stacked vectors of multiple output channels
$\pmb{f}_{in}$ - $(C,V,T)$ size input tensor
$\pmb{\Lambda}^{ii}=\sum\limits_{j}(A^{ij}+I^{ij})$ - degree matrix

Graph convolution is done by $1\times\Gamma$ 2D convolution and multiplication with the normalized adjacency matrix on the second dimension.

For partitioning strategies other than uni-labeling, the adjacency matrix is dismantled as $\pmb{A}+\pmb{I}=\sum\limits_{j}\pmb{A}_{j}$, which yields $\pmb{f}_{out}=\sum\limits_{j}\pmb{\Lambda}_{j}^{-\frac{1}{2}}\pmb{A}_{j}\pmb{\Lambda}_{j}^{-\frac{1}{2}}\pmb{f}_{in}\pmb{W}_{j}$, where  $\pmb{\Lambda}_{j}^{ii}=\sum\limits_{k}(A_{j}^{ik}+\alpha)$ and $\alpha=0.001$ to avoid empty rows in $\pmb{A}_{j}$.
$\pmb{\hat{A}} = \pmb{\Lambda}_{j}^{-\frac{1}{2}}\pmb{A}_{j}\pmb{\Lambda}_{j}^{-\frac{1}{2}}$ is a renormalized adjacency matrix. ~~(Why?)~~

Edge importance weighting is implemented as $\pmb{A}_{j}\otimes\pmb{M}_{j}$, an element-wise product of the subset's adjacency matrix with a learnable matrix $\pmb{M}_{j}$, unique to each subset's adjacency matrix and initialized as all-ones matrix.

Graph structure remains the same, only node features are updated. Each node's features are updated with respect to the immediately connected nodes in its neighborhood.
The GCN reuses the same weight matrix across all nodes in the layer, but the variability comes from the renormalized adjacency matrix and the mask matrix, which are designed to exploit intrinsic relationships between the nodes and the partition scheme, and the mapping function that reorders nodes to be multiplied with.

Spatial partitioning shows the best results.

## References
1. Yan, S., Xiong, Y., & Lin, D. (2018). Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. _arXiv preprint [arxiv:1801.07455](https://arxiv.org/abs/1801.07455)._
2. [[Semi-Supervised Classification with Graph Convolutional Networks]]