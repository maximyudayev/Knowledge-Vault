Created: 15-04-2022 10:36
Status: #summary #done
Tags: [[Machine Learning]] [[Temporal Convolutional Network]] [[Action Segmentation]]

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
## References
1. Farha, Y. A., & Gall, J. (2019). MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation. _2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_. https://doi.org/10.1109/cvpr.2019.00369