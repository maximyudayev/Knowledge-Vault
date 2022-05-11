Created: 17-04-2022 12:57
Status: #summary #todo
Tags: [[Graph Convolutional Network]] [[Machine Learning]] [[Action Segmentation]]

# Remarks
1. It is most likely that PyTorch implementations stream the full pre-recorded dataset through the network one layer at a time. This simplifies the implementation for researchers since no management of FIFOs is needed.
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
  4th and 7th temporal convolutional layers have stride of 2 for pooling, and also have a $1\times 1$ 2D convolution in the residual connection to match the layer's output dimension.
  Global pooling at the end is done to get 256 feature vector for the input sequence and fed through the SoftMax. ^4e9dea
- Weight matrix in GCN is shared among nodes: varying node degrees are dealt with by appropriate normalization of the adjacency matrix.
## Skeleton Graph Construction
Graph is comprised of a set of nodes $V=\{v_{ti}|t=1,...,T;i=1,...,N\}$, which are interconnected with a set of edges $E$, split across 2 subsets: $E_{S}=\{v_{ti}v_{tj}|(i,j)\in H\}$ - intra-frame skeleton connections, $E_{F}=\{v_{ti}v_{t+1)i}\}$ - inter-frame same joint connections. $E_{F}$ represents trajectory of a joint $i$ over time.
$F(v_{ti})$ is the feature vector of the node $v_{i}$ in time frame $t$, along with the confidence estimate.
Graph nodes are connected within each frame according to the skeleton structure. The same joint is connected to itself across all the time frames.
## Spatial Graph Convolutional Neural Network
Sampling function $p$ spits out a node in the neighborhood of $v_{ti}$ that's a member of the set $B(v_{ti})=\{v_{tj}|d(v_{ti},v_{tj})\leq D\}$, where $d(v_{ti},v_{tj})$ is the shortests distance between 2 nodes and $D$ is set to 1 in this work.

Weight function $w$ orders the weights according to the partitioning strategy of neighbor set $B(v_{ti})$ into $K$ subsets, with each of $K$ subsets having a label. 

$$l_{ti}:B(v_{ti})\rightarrow\{0,...,K-1\}$$  ^1210b3

maps a node in a neighborhood to its subset label. The weight function is then $w(v_{ti},v_{tj})=w'(l_{ti}(v_{tj})): B(v_{ti})\rightarrow\Re^c$.

The spatial graph convolution is then $f_{out}(v_{ti})=\sum\limits_{v_{tj}\in B(v_{ti})}\frac{1}{Z_{ti}(v_{tj})}f_{in}(p(v_{ti},v_{tj}))w(v_{ti},v_{tj})$, where $Z_{ti}(v_{tj})=|\{v_{tk}|l_{ti}(v_{tk})=l_{ti}(v_{tj})\}|$ is the cardinality of the subset that balances the contributions of different subsets to the output.
## Spatial Temporal Modeling
Modeling spatial temporal dynamics in the sequence is extended from the spatial graph convolution by extending the concept of neighborhood, including now the temporally connected joints 

$$B(v_{ti})=\{v_{qj}|d(v_{ti},v_{tj})\leq K; |q-t|\leq \lfloor\Gamma/2\rfloor\}$$ ^ce4b45

where $\Gamma$ is the temporal kernel size that controls the temporal range to be included in the neighborhood. This is equivalent to a 3D tensor of $n\times\Gamma\times C$, where $n$ is the degree of the node $v_{i}$ (the number of neighbor nodes it is connected to).

$$l_{ST}(v_{qj})=l_{ti}(v_{tj})+(q-t+\lfloor\Gamma/2\rfloor)\times K$$  ^d970a4

is the well-defined label mapping function for the spatial temporal graph convolution.

This mapping function is implicitly done by multiplication of the input tensor with the adjacency matrix.
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
$\pmb{f}_{out}=\pmb{D}^{-\frac{1}{2}}(\pmb{A}+\pmb{I})\pmb{D}^{-\frac{1}{2}}\pmb{f}_{in}\pmb{W}$

$\pmb{A}$ - adjacency matrix of intra-frame connections
$\pmb{I}$ - identity matrix of intra-frame self-connections
$\pmb{W}$ - weight matrix of stacked vectors of multiple output channels
$\pmb{f}_{in}$ - $(C,V,T)$ size input tensor
$\pmb{D}^{ii}=\sum\limits_{j}(A^{ij}+I^{ij})$ - diagonal node degree matrix (sum of all connections of a node), where for a undirectional graph the self-connection of a node is added as 2 to the degree value.

Graph convolution is done by $1\times\Gamma$ 2D convolution and multiplication with the normalized adjacency matrix on the second dimension.

For partitioning strategies other than uni-labeling, the adjacency matrix is dismantled as $\pmb{A}+\pmb{I}=\sum\limits_{j}\pmb{A}_{j}$, which yields $\pmb{f}_{out}=\sum\limits_{j}\pmb{D}_{j}^{-\frac{1}{2}}\pmb{A}_{j}\pmb{D}_{j}^{-\frac{1}{2}}\pmb{f}_{in}\pmb{W}_{j}$, where  $\pmb{D}_{j}^{ii}=\sum\limits_{k}(A_{j}^{ik}+\alpha)$ and $\alpha=0.001$ to avoid empty rows in $\pmb{A}_{j}$.
$\pmb{\hat{A}} = \pmb{D}_{j}^{-\frac{1}{2}}\pmb{A}_{j}\pmb{D}_{j}^{-\frac{1}{2}}$ is a renormalized adjacency matrix. ~~(Why?)~~

Edge importance weighting is implemented as $\pmb{A}_{j}\otimes\pmb{M}_{j}$, an element-wise product of the subset's adjacency matrix with a learnable matrix $\pmb{M}_{j}$, unique to each subset's adjacency matrix and initialized as all-ones matrix.

Graph structure remains the same, only node features are updated. Each node's features are updated with respect to the immediately connected nodes in its neighborhood.
The GCN reuses the same weight matrix across all nodes in the layer, but the variability comes from the renormalized adjacency matrix and the mask matrix, which are designed to exploit intrinsic relationships between the nodes and the partition scheme, and the mapping function that reorders nodes to be multiplied with.

Spatial partitioning shows the best results.
### In-depth Operator Analysis
#### Concept
A spatial temporal graph for the skeleton-based action segmentation can be thought of as a 3D tensor with length $\Gamma$ along the time axis, number of joints $N$ and number of features per joint $C$ along the other $2$ axis. This is because skeleton graph data is consistent and measurements across nodes come in the same order, frame after frame: each row of a resulting tensor corresponds to a single node across time.
ST-GCN takes the frame at $t$, $\lfloor\Gamma/2\rfloor$ frames into the future and $\lfloor\Gamma/2\rfloor$ frames into the past: in realtime applications, this can be thought of as processing latency of $\lfloor\Gamma/2\rfloor$ cycles (in the units of the sampling frequency of the capture device/dataset).

The drawing shows the tensor equivalent representation of the spatial-temporal graph.
Highlighted nodes are the timeframe at time $t$ at which the output $\pmb{f}_{out,t}\in\Re^{N\times C}$ is produced.
![[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition 5.png]]

The concept of applying ST-GCN is identical to GCN, but it extends the node neighborhood to the temporal dimension as well. To update a node's features, its neighbors are independently convolved with the shared weight kernel and later summed together across the new channels.
![[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition 6.png]]

$\sum\limits_{j}\pmb{\hat{A}}_{j}\pmb{f}_{in}\pmb{W}_{j}$ can be thought of as, for each partition $j$, a $1\times 1$ 2D convolution of the input $\pmb{f}_{in}\in\Re^{N\times T\times C}$ with the weight matrix $\pmb{W}_{j}\in\Re^{1\times 1\times C\times C}$ to produce an intermediate buffer $\pmb{f}^{'}_{j}\in\Re^{N\times T\times C}$, where each slice of the tensor is reused $\Gamma$ times. 

![[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition 1.png]]

Window of size $\Gamma$ slides over time and selectively weights and accumulates the partial convolution results of the graph based on the masked normalized adjacency matrix $\pmb{\hat{A}}_{j}\in\Re^{N\times N}$ which spatially accumulates interconnected nodes across neighborhoods for each time step separately.
However, in the same partition, each frame needs to be spatially combined based on the adjacency matrix only once since a frame's spatial accumulation is time-invariant: hence this step can be done on each incoming frame before pushing it into the corresponding $j$-th FIFO.

![[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition 2.png]]

The resulting $\pmb{f}^{''}_{j}\in\Re^{N\times\Gamma\times C}$ is then reduced by simply accumulating along $\Gamma$, the time dimension.

![[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition 3.png]]

The partial output matrix of partition $j$, $\pmb{f}^{'''}_{j}\in\Re^{N\times 1\times C}$, is then summed with the others to produce the final output $\pmb{f}_{out,t}\in\Re^{N\times 1\times C}$.
$\pmb{f}_{out,t}$ is fed through [[Batch Normalization]]$\rightarrow$[[Dropout Layer]]$\rightarrow$[[Residual Connection]]$\rightarrow$[[ReLU]].
The result is then pushed into the output FIFO of size $\Gamma$ to be consumed by the follow-up ST-GCN unit.

![[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition 4.png]]
#### Datapath Implication
This means that each $N\times 1\times C$ input sample, fed in at the sampling frequency of the dataset or the capturing system, is 2D convolved with $\pmb{W}_{j}$ and pushed into the corresponding $j$-th FIFO. The last $\Gamma$ samples are then used to produce an output sample.

For each partition, the 2D convolution of$\pmb{f}_{in}\in\Re^{N\times T\times C}$ with the weight matrix $\pmb{W}_{j}\in\Re^{1\times 1\times C\times C}$ is equivalent to doing a linear combination of features across every node and across all time stamps, separately. It produces partial accumulations/contributions of each node across space and time.

The multiplication with the $N\times N$ adjacency matrix (can be already weighted by the mask matrix M) is equivalent to accumulating partial contributions of nodes across space and time of only those joints that are directly connected in the neighborhood.

The final accumulation across the time axis (second dimension of size Gamma) produces the final output at time $t$ for the partition $j$ of size N x 1 x C, equivalent to summing together at time $t$, for each node, partial contributions of nodes in the neighborhood coming from $\lfloor\Gamma/2\rfloor$ former and $\lfloor\Gamma/2\rfloor$ future time frames.

Lastly, the partial results of each partition $j$ are summed and pushed into the output FIFO, consumed by the next ST-GCN layer.

Tadaa! No TCN needed, straight-forward implementation using standard differentiable tensor operators.

The residual connections in ST-GCN layers 4 and 7 are fed through a $1\times 1$ 2D convolution to match the layer's output dimensions.
### Experiments
#### Kinetics
Computer vision based. 400 human action classes. 300'000 videos of 10 seconds. Uses OpenPose pose estimator to estimate 2D locations of 18 joints from pictures, with a confidence metric for each.
$T$ is set to 300.
#### NTU-RGB+D
60 human action classes. 56'000 videos. Uses Kinect with depth estimator to estimate 3D locations of 25 joints from pictures.
## References
1. Yan, S., Xiong, Y., & Lin, D. (2018). Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. _arXiv preprint [arxiv:1801.07455](https://arxiv.org/abs/1801.07455)._
2. [[Semi-Supervised Classification with Graph Convolutional Networks]].
3. [[Graph Convolutional Network]].