Created: 15-04-2022 11:45
Status: #summary #todo
Tags: [[Temporal Convolutional Network]] [[Freezing of Gait]] [[Graph Convolutional Network]] [[Machine Learning]] [[Action Segmentation]]

# TODO:
Summarize the TCN block operations.

# Skeleton-Based Action Segmentation with Multi-Stage Spatial-Temporal Graph Convolutional Neural Networks
- Each FIFO input sample is batch normalized and remapped to the number of channels of the network.
- First stage is ST-GCN that approprietly models spatial hierarchy across joints.
- Combines [[Spatial-Temporal Graph Convolutional Network]] and Multi-Stage [[Temporal Convolutional Network]].
- Spatial graph convolutions learn spatial patterns.
- Dilated temporal convolutions learn long-term temporal patterns.
- Multiple TCN stages refine original predictions and learn higher-order sequence of actions defining more complex actions.
## Architecture
$\pmb{f}_{in}$ - input data
$\pmb{f}_{adj}$ - remapped input to network feature map size
$\pmb{f}_{gcn}$ - output of the ST-GCN stage
$\pmb{\hat{Y}}$ - output sequence

$T$ - number of samples
$N$ - number of joints
$C_{in}$ - number of input channels per joint (IMU data)
$C$ - number of feature channels in the neural network
$l$ - number of segmentation classes
### Input Normalization
Each sample $f_{in_{t}}\in\Re^{N\times 1\times C_{in}}$ is passed through [[Batch Normalization]].

<u>Total:</u> 
1. __\#MAC:__ $N\times C_{in}\times(1+1)$ (per time sample); bias addition can be preloaded in the accumulator to avoid separate additions.
2. __\#params:__ $1+1$ (tensor-wise normalization), $C_{in}\times(1+1)$ (channel-wise normalization).
3. __\#memory:__ data can be streamed into the next layer, no need to save intermediate data.

<u>Note:</u> 
1. Aforementioned MAC and params totals are applicable to only inference, after the BN parameters have been learned and frozen.
2. The normalization can be fused with the next convolution layer to reduce the number of operations and stored parameters.
### Input Remapping
 Each sample is then mapped to the network's feature map dimensions, $f_{adj_{t}}\in\Re^{N\times 1\times C}$, after sampling and is pushed into a $T$ sized FIFO buffer. Mapping is a $1\times 1$ 2D Convolution (or a fully-connected layer applied across $N$ and $T$ dimensions of the input tensor) with $\pmb{W}_{in}\in\Re^{1\times 1\times C_{in}\times C}$ and $\pmb{b}\in\Re^C$.

$$\pmb{f}_{in}\in\Re^{N\times T\times C_{in}} \xrightarrow{1x1\quad 2D\ Conv} \pmb{f}_{adj}\in\Re^{N\times T\times C}$$

<u>Total:</u> 
1. __\#MAC:__ $N\times(C_{in}\times C+C)$ (per sample); bias addition can be preloaded in the accumulator to avoid separate additions.
2. __\#params:__ $C_{in}\times C+C$. 
3. __\#memory:__ $N\times C_{in}$ (per sample); $N\times\Gamma\times C_{in}$ for the $\pmb{f}_{adj}$ FIFO.

<u>Note:</u> 
1. Because the model does inference at each new sample each sample remains in the FIFO and is reused for $\Gamma$ time steps (in the units of the sampling frequency). Hence, it is more efficient to normalize and remap each sample only once since adjusted FIFO samples are time-invariant.
2. The previous normalization can be fused with this convolution layer to reduce the number of operations and stored parameters.
### ST-GCN Block
#### Graph Convolution
[[Spatial-Temporal Graph Convolutional Network]] learns a representation on a graph using: 
- the set of graph nodes $V$ over $N$ skeleton joints, across $T$ time samples.
- two sets of edges across these nodes, one for spatial connection between nodes across the same time sample, and other for temporal connection between the same node across time.
- an adjacency matrix, describing the graph structure.

$$f_{gcn}(v_{ti})=\sum\limits_{v_{tj}\in B(v_{ti})}\frac{1}{Z_{ti}(v_{tj})}f_{in}(v_{tj})w(l_{ti}(v_{tj}))$$

$v_{tj}\in B(v_{ti})$ - nodes within sampling area of the node $v_{ti}$.
$l_{ti}$ - function that maps each node to a unique weight vector $w$.
$Z_{ti}$ - normalizing term across graph partitions.

Since it is a CNN generalization, the weights are reused across the input data. The spatial-temporal graph convolution uses an indexing function to order the nodes correctly:
![[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition#^1210b3]] ![[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition#^d970a4]]

This spatial-temporal mapping is done by the adjacency matrix $\pmb{A}\in\{0,1\}^{N\times N}$ when multiplied by the 3D graph tensor $\pmb{f}_{in}\in\Re^{N\times T\times C}$.

#### ST-GCN General Layer
$$\pmb{f}_{gcn}=\sum\limits_{j}\pmb{\hat{A}}_{j}\pmb{f}_{adj}\pmb{W}_{j}$$

$\pmb{W}_{j}\in\Re^{1\times 1\times C_{in}\times C}$ - weight matrix.
$\pmb{\hat{A}}_{j}\in\Re^{N\times N}=\pmb{D}_{j}^{-\frac{1}{2}}\pmb{A}_{j}\pmb{D}_{j}^{-\frac{1}{2}}\otimes\pmb{M}_{j}$ - symmetrically normalized adjacency matrix of spatial connections between joints of that partition weighted by the edge importance matrix of that partition (element-wise product).
$\pmb{A}_{j}\in\{0,1\}^{N\times N}$ - original adjacency matrix of the partition.
$\pmb{D}_{j}$ - diagonal node degree matrix (sum of all connections of a node), where for a undirectional graph the self-connection of a node is added as 2 to the degree value.
$\pmb{M}_{j}\in\Re^{N\times N}$ - learnable weight matrix.
$j$ - node neighborhood partition.

The [[Spatial-Temporal Graph Convolutional Network]] layers are stacked into a block of 9:
![[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition#^4e9dea]]

##### Layer 1
First ST-GCN layer in the code does not have a residual connection.
1. __\#MAC:__ 
	1. $j\times N\times C_{1}\times (C_{1}+1)$, convolution of node features of incoming sample (bias can be preloaded into the accumulator to avoid separate additions).
	2. $j\times N\times C_{1}\times N$, spatial accumulation of partial sums of new node features of incoming frame (multiplication with adjacency matrix)$\rightarrow$pushed into FIFO.
	3. $j\times N\times C_{1}\times\Gamma$, temporal accumulation of partial sums of spatial accumulations on whole FIFO.
	4. $j\times N\times C_{1}$, accumulation across partitions.
	5. $N\times C_{1}\times (1+1)$, batch normalization (bias can be preloaded into the accumulator to avoid separate additions).
	6. __?__ [[Dropout Layer]] (50% chance of discarding).
	7. $N\times C_{1}$ comparisons ([[ReLU]]$=\max(0, x)$).
2. __\#params:__ 
	1. $j\times C_{1}\times (C_{1}+1)$ (convolution).
	2. $C_{1}+C_{1}$ (batch normalization).
3. __\#memory:__ 
	1. $j\times N\times\Gamma\times C_{1}$ words (internal $j$ FIFOs).

##### Layer 2
Input frame produced by the previous layer is pushed into the $\lceil\Gamma/2\rceil$ FIFO before any processing steps for later use of the last element in the residual connection.
1. __\#MAC:__ 
	1. $j\times N\times C_{1}\times (C_{1}+1)$, convolution of node features of incoming sample (bias can be preloaded into the accumulator to avoid separate additions).
	2. $j\times N\times C_{1}\times N$, spatial accumulation of partial sums of new node features of incoming frame (multiplication with adjacency matrix)$\rightarrow$pushed into FIFO.
	3. $j\times N\times C_{1}\times\Gamma$, temporal accumulation of partial sums of spatial accumulations on whole FIFO.
	4. $j\times N\times C_{1}$, accumulation across partitions.
	5. $N\times C_{1}\times (1+1)$, batch normalization (bias can be preloaded into the accumulator to avoid separate additions).
	6. __?__ [[Dropout Layer]] (50% chance of discarding).
	7. $N\times C_{1}$ additions ([[Residual Connection]]).
	8. $N\times C_{1}$ comparisons ([[ReLU]]$=\max(0, x)$).
2. __\#params:__ 
	1. $j\times C_{1}\times (C_{1}+1)$ (convolution main branch).
	2. $C_{1}+C_{1}$ (batch normalization).
4. __\#memory:__ 
	1. $j\times N\times\Gamma\times C_{1}$ words (internal $j$ FIFOs).
	2. $N\times \lceil\Gamma/2\rceil\times  C_{1}$ words (first half of the FIFO, but unprocessed for use for the residual connection).

##### Layer 3
1. __\#MAC:__ 
	1. $j\times N\times C_{1}\times (C_{1}+1)$, convolution of node features of incoming sample (bias can be preloaded into the accumulator to avoid separate additions).
	2. $j\times N\times C_{1}\times N$, spatial accumulation of partial sums of new node features of incoming frame (multiplication with adjacency matrix)$\rightarrow$pushed into FIFO.
	3. $j\times N\times C_{1}\times\Gamma$, temporal accumulation of partial sums of spatial accumulations on whole FIFO.
	4. $j\times N\times C_{1}$, accumulation across partitions.
	5. $N\times C_{1}\times (1+1)$, batch normalization (bias can be preloaded into the accumulator to avoid separate additions).
	6. __?__ [[Dropout Layer]] (50% chance of discarding).
	7. $N\times C_{1}$ additions ([[Residual Connection]]).
	8. $N\times C_{1}$ comparisons ([[ReLU]]$=\max(0, x)$).
3. __\#params:__ 
	1. $j\times C_{1}\times (C_{1}+1)$ (convolution main branch).
	2. $C_{1}+C_{1}$ (batch normalization).
4. __\#memory:__ 
	1. $j\times N\times\Gamma\times C_{1}$ words (internal $j$ FIFOs).
	2. $N\times \lceil\Gamma/2\rceil\times  C_{1}$ words (first half of the FIFO, but unprocessed for use for the residual connection).

##### Layer 4
The FIFO is twice larger to allow dilation during temporal accumulation of spatial partial sums.
1. __\#MAC:__ 
	1. $j\times N\times C_{2}\times (C_{1}+1)$, convolution of node features of incoming sample (bias can be preloaded into the accumulator to avoid separate additions).
	2. $j\times N\times C_{2}\times N$, spatial accumulation of partial sums of new node features of incoming frame (multiplication with adjacency matrix)$\rightarrow$pushed into FIFO.
	3. $j\times N\times C_{2}\times\Gamma$, temporal accumulation of partial sums of spatial accumulations on whole FIFO.
	4. $j\times N\times C_{2}$, accumulation across partitions.
	5. $N\times C_{2}\times (1+1)$, batch normalization (bias can be preloaded into the accumulator to avoid separate additions).
	6. __?__ [[Dropout Layer]] (50% chance of discarding).
	7. $N\times C_{2}\times (C_{1}+1)$, convolution of node features of incoming sample to adjust dimensions in the residual branch (bias can be preloaded into the accumulator to avoid separate additions).
	8. $N\times C_{2}$ additions ([[Residual Connection]]).
	9. $N\times C_{2}$ comparisons ([[ReLU]]$=\max(0, x)$).
2. __\#params:__ 
	1. $j\times C_{2}\times (C_{1}+1)$ (convolution main branch).
	2. $C_{2}\times (C_{1}+1)$ (convolution residual branch).
	3. $C_{2}+C_{2}$ (batch normalization).
3. __\#memory:__ 
	1. $j\times N\times(2\times\Gamma-1)\times C_{2}$ words (internal $j$ FIFOs).
	2. $N\times \lceil(2\times\Gamma-1)/2\rceil\times  C_{1}$ words (first half of the FIFO, but unprocessed for use for the residual connection).

##### Layer 5
1. __\#MAC:__ 
	1. $j\times N\times C_{2}\times (C_{2}+1)$, convolution of node features of incoming sample (bias can be preloaded into the accumulator to avoid separate additions).
	2. $j\times N\times C_{2}\times N$, spatial accumulation of partial sums of new node features of incoming frame (multiplication with adjacency matrix)$\rightarrow$pushed into FIFO.
	3. $j\times N\times C_{2}\times\Gamma$, temporal accumulation of partial sums of spatial accumulations on whole FIFO.
	4. $j\times N\times C_{2}$, accumulation across partitions.
	5. $N\times C_{2}\times (1+1)$, batch normalization (bias can be preloaded into the accumulator to avoid separate additions).
	6. __?__ [[Dropout Layer]] (50% chance of discarding).
	7. $N\times C_{2}$ additions ([[Residual Connection]]).
	8. $N\times C_{2}$ comparisons ([[ReLU]]$=\max(0, x)$).
2. __\#params:__ 
	1. $j\times C_{2}\times (C_{2}+1)$ (convolution main branch).
	2. $C_{2}+C_{2}$ (batch normalization).
3. __\#memory:__ 
	1. $j\times N\times\Gamma\times C_{2}$ words (internal $j$ FIFOs).
	2. $N\times \lceil\Gamma/2\rceil\times  C_{2}$ words (first half of the FIFO, but unprocessed for use for the residual connection).

##### Layer 6
1. __\#MAC:__ 
	1. $j\times N\times C_{2}\times (C_{2}+1)$, convolution of node features of incoming sample (bias can be preloaded into the accumulator to avoid separate additions).
	2. $j\times N\times C_{2}\times N$, spatial accumulation of partial sums of new node features of incoming frame (multiplication with adjacency matrix)$\rightarrow$pushed into FIFO.
	3. $j\times N\times C_{2}\times\Gamma$, temporal accumulation of partial sums of spatial accumulations on whole FIFO.
	4. $j\times N\times C_{2}$, accumulation across partitions.
	5. $N\times C_{2}\times (1+1)$, batch normalization (bias can be preloaded into the accumulator to avoid separate additions).
	6. __?__ [[Dropout Layer]] (50% chance of discarding).
	7. $N\times C_{2}$ additions ([[Residual Connection]]).
	8. $N\times C_{2}$ comparisons ([[ReLU]]$=\max(0, x)$).
2. __\#params:__ 
	1. $j\times C_{2}\times (C_{2}+1)$ (convolution main branch).
	2. $C_{2}+C_{2}$ (batch normalization).
3. __\#memory:__ 
	1. $j\times N\times\Gamma\times C_{2}$ words (internal $j$ FIFOs).
	2. $N\times \lceil\Gamma/2\rceil\times  C_{2}$ words (first half of the FIFO, but unprocessed for use for the residual connection).

##### Layer 7
The FIFO is twice larger to allow dilation during temporal accumulation of spatial partial sums.
1. __\#MAC:__ 
	1. $j\times N\times C_{3}\times (C_{2}+1)$, convolution of node features of incoming sample (bias can be preloaded into the accumulator to avoid separate additions).
	2. $j\times N\times C_{3}\times N$, spatial accumulation of partial sums of new node features of incoming frame (multiplication with adjacency matrix)$\rightarrow$pushed into FIFO.
	3. $j\times N\times C_{3}\times\Gamma$, temporal accumulation of partial sums of spatial accumulations on whole FIFO.
	4. $j\times N\times C_{3}$, accumulation across partitions.
	5. $N\times C_{3}\times (1+1)$, batch normalization (bias can be preloaded into the accumulator to avoid separate additions).
	6. __?__ [[Dropout Layer]] (50% chance of discarding).
	7. $N\times C_{3}\times (C_{2}+1)$, convolution of node features of incoming sample to adjust dimensions in the residual branch (bias can be preloaded into the accumulator to avoid separate additions).
	8. $N\times C_{3}$ additions ([[Residual Connection]]).
	9. $N\times C_{3}$ comparisons ([[ReLU]]$=\max(0, x)$).
2. __\#params:__ 
	1. $j\times C_{3}\times (C_{2}+1)$ (convolution main branch).
	2. $C_{3}\times (C_{2}+1)$ (convolution residual branch).
	3. $C_{3}+C_{3}$ (batch normalization).
3. __\#memory:__ 
	1. $j\times N\times(2\times\Gamma-1)\times C_{3}$ words (internal $j$ FIFOs).
	2. $N\times \lceil(2\times\Gamma-1)/2\rceil\times  C_{2}$ words (first half of the FIFO, but unprocessed for use for the residual connection).

##### Layer 8
1. __\#MAC:__ 
	1. $j\times N\times C_{3}\times (C_{3}+1)$, convolution of node features of incoming sample (bias can be preloaded into the accumulator to avoid separate additions).
	2. $j\times N\times C_{3}\times N$, spatial accumulation of partial sums of new node features of incoming frame (multiplication with adjacency matrix)$\rightarrow$pushed into FIFO.
	3. $j\times N\times C_{3}\times\Gamma$, temporal accumulation of partial sums of spatial accumulations on whole FIFO.
	4. $j\times N\times C_{3}$, accumulation across partitions.
	5. $N\times C_{3}\times (1+1)$, batch normalization (bias can be preloaded into the accumulator to avoid separate additions).
	6. __?__ [[Dropout Layer]] (50% chance of discarding).
	7. $N\times C_{3}$ additions ([[Residual Connection]]).
	8. $N\times C_{3}$ comparisons ([[ReLU]]$=\max(0, x)$).
2. __\#params:__ 
	1. $j\times C_{3}\times (C_{3}+1)$ (convolution main branch).
	2. $C_{3}+C_{3}$ (batch normalization).
3. __\#memory:__ 
	1. $j\times N\times\Gamma\times C_{3}$ words (internal $j$ FIFOs).
	2. $N\times \lceil\Gamma/2\rceil\times  C_{3}$ words (first half of the FIFO, but unprocessed for use for the residual connection).

##### Layer 9
Last layer of the ST-GCN block. Followed up by the average pooling of the output $N\times C_{3}$ frame across the $N$ dimension and mapping of the resulting $C_{3}$ vector to the size of the input of the TCN blocks through a [[Fully Connected Network]].
1. __\#MAC:__ 
	1. $j\times N\times C_{3}\times (C_{3}+1)$, convolution of node features of incoming sample (bias can be preloaded into the accumulator to avoid separate additions).
	2. $j\times N\times C_{3}\times N$, spatial accumulation of partial sums of new node features of incoming frame (multiplication with adjacency matrix)$\rightarrow$pushed into FIFO.
	3. $j\times N\times C_{3}\times\Gamma$, temporal accumulation of partial sums of spatial accumulations on whole FIFO.
	4. $j\times N\times C_{3}$, accumulation across partitions.
	5. $N\times C_{3}\times (1+1)$, batch normalization (bias can be preloaded into the accumulator to avoid separate additions).
	6. __?__ [[Dropout Layer]] (50% chance of discarding).
	7. $N\times C_{3}$ additions ([[Residual Connection]]).
	8. $N\times C_{3}$ comparisons ([[ReLU]]$=\max(0, x)$).
	9. $C_{3}\times N$ additions and $C_{3}$ divisions (average pooling across nodes).
	10. $C_{TCN}\times (C_{3}+1)$ (convolution/FCN remapping to the size of the TCN blocks).
	11. __?__ [[Softmax Probability]] (converting predictions to probabilities.
2. __\#params:__ 
	1. $j\times C_{3}\times (C_{3}+1)$ (convolution main branch).
	2. $C_{3}+C_{3}$ (batch normalization).
3. __\#memory:__ 
	1. $j\times N\times\Gamma\times C_{3}$ words (internal $j$ FIFOs).
	2. $N\times \lceil\Gamma/2\rceil\times  C_{3}$ words (first half of the FIFO, but unprocessed for use for the residual connection).

##### Total
1. __\#MAC:__ 
	1. Total per each layer above.
2. __\#params:__ 
	1. Total per each layer above.
	2. $j\times N\times N$ words (normalized, importance weighted adjacency matrices, for each partition $j$). 
3. __\#memory:__ 
	1. Total per each layer above.

<u>Note:</u> 
1. The values are computed according to the approach elaborated in [[Spatial-Temporal Graph Convolutional Network]] for the optimal reuse of data that avoids recomputations. That is:
	1. Each sample is fed to the $1\times 1$ 2D convolution, with the result stored in one of the corresponding $j$ FIFOs of size $\Gamma$.
	2. The $\Gamma$ sized FIFO performs accumulation of partial sums as dictated by the normalized weighted adjacency matrix $\pmb{\hat{A}}_{j}$.
2. During inference the adjacency matrix $\pmb{\hat{A}}_{j}$ does not have to be recomputed since the weighted importance matrix $\pmb{M}_{j}$ is frozen and the two can be precomputed and stored in memory. 

### TCN Block
_Continues here_
### Output
$\hat{Y}\in\Re^{T\times l}$ sequence of probabilities of classes. In the case of MS-GCN, $l=2$, to classify only FOG.
## Notes
Window $T$ must be selected wide enough to accomodate for depth of the dilated convolutions
## References
1. Filtjens, B., Vanrumste, B., & Slaets, P. (2022). Skeleton-Based Action Segmentation with Multi-Stage Spatial-Temporal Graph Convolutional Neural Networks. _arXiv preprint [arxiv:2202.01727](https://arxiv.org/abs/2202.01727)._
2. [[Spatial-Temporal Graph Convolutional Network]]