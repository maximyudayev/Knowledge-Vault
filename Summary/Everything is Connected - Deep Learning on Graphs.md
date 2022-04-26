Created: 26-04-2022 12:08
Status: #summary 
Tags: [[mlcon 2.0]] [[Machine Learning]] [[Conferences]]

# Everything is Connected - Deep Learning on Graphs
- World is graphs - the main modality of data.
- Image is a projection of the complex intra-connected environment that we make to easier perceive the data.
- Human perception is graph-structured. It is hence the most likely path toward [[General Artificial Intelligence]].
- Graph representation learning is processing data that lives on graphs to produce knowledge.
- [[Graph Convolutional Network]]'s update features of nodes on a graph as a function of the information on a node and all the neighboring nodes: updating a node by taking into account itself and its relationships the nodes connected to it.
- GNNs allow you to take into account irregularly structured data and the local structure that data exhibits as context. Capitalizing on the intrinsic connections within data, without the need for the ML model to implicitly learn it.
- Updates are done per each node neighborhood in isolation, stacking layers and non-linearities to learn more high-level and sophisticated features.
- Based on the updated features on the network, a classification can be done by feeding separate node features to classify nodes, the entire (sub)graphs to classify graphs, or to classify properties on the edges or if certain edges even exist by a classifier that uses incident nodes' features and edge features themselves.
- Node-, edge-, or graph-level predictions.
	- Node classification.
	- Graph classification.
	- Link prediction (knowledge graphs and recommendations systems). 
- Google uses GNNs in combination with RL to design its next TPUv5 chip (place and route).
- Google maps use case.
	- Path finder spits out a candidate route through the road network. The route is partitioned. Partition of the road is fed to the GNN. GNN estimates based on histroical and current travel conditions the travel time. These estimates are aggregated over the entire route for the final prediction.
		- Get candidate route from the path finder module.
		- Partition the road network into supersegments, sampled according to the density of the traffic (sized equally w.r.t. traffic density).
		- Run GNN over each supersegment graph to estimate the time of arrival.
		- Aggregate travel times.
- Transformers are a special case of attentional GNNs that operate over the entire graph. [[Transformers are Graph Neural Networks]].
## References
1. [Everything is connected: deep learning on graphs](https://www.youtube.com/watch?v=tL7-Lc38m24)