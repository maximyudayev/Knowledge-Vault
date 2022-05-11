Created: 15-04-2022 10:36
Status: #summary #todo
Tags: [[Machine Learning]] [[Temporal Convolutional Network]] [[Action Segmentation]]

# Remarks
- Authors and forks stream the full recording sequence through the model layer by layer, not using a sliding window.
- Farha uses causal TCN in his which lets him stream the full recorded sequence through the TCN stages, layer-by-layer.
# MS-TCN Multi-Stage Temporal Convolutional Network for Action Segmentation
- Multiple [[Temporal Convolutional Network]] stages are joined to refine predictions.
- Has a large [[Receptive Field]] with less parameters than a [[Recurrent Neural Network]].
- More computationally efficient than a sequential RNN/LSTM.
- Stage consists of layers of [[Dilated Convolution]].
- 10 layers per stage produce the best results.
- Each layer is followed by an activation function and a residual connection.
- Layers are progressively dilated by a factor of 2: 1, 2, 4, ..., 512.
- A stage outputs [[Softmax Probability]], which is the input of the next stage.
- Convolutional layers are temporal window size agnostic, unlike fully-connected layers.
- Pooling reduces temporal resolution, hence not used.
- The very first layer of a TCN is a 1D convolutional layer that maps input feature size to match the network's feature map, which makes the same model scalable to a variety of input data configuration (operates on embedding: encoder 1D convolutional layer can be retrained for overly divergent tasks, instead of the entire network).
- The very last layer of a TCN is a 1D convolutional layer that maps the internal feature map to the output features, followed by a [[Softmax Probability]] function that produces probabilities of classification classes.
- Multi-stage architecture provides more context for more accurate predictions and less oversegmentation, thanks to learning sequence dependencies of actions at growingly higher abstractions.
- 64 filters per TCN layer are used, with kernels of size 3.

## Description
The very first layer of a TCN stage is a $1\times 1$ convolution that remaps the number of input channels to the number of channels in the network, namely 64.
Each stage consists of $L$ layers, namely 10, where each layer consists of a [[Dilated Convolution]]->[[ReLU]]->$1\times 1$ Convolution->[[Residual Connection]]. These $L$ layers are stacked together into a single stage, followed by the [[Softmax Probability]] unit to produce probabilities.
![[MS-TCN Multi-Stage Temporal Convolutional Network for Action Segmentation 1.png]]
## References
1. Farha, Y. A., & Gall, J. (2019). MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation. _2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_. https://doi.org/10.1109/cvpr.2019.00369