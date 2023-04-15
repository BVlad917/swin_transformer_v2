# Swin Transformer V2

Implementation of the [Swin Transformer V2](https://arxiv.org/abs/2111.09883).

## Swin Transformer V2
<img width="513" alt="image" src="https://user-images.githubusercontent.com/72063186/229312414-2a9a6d40-97a9-4293-a9c6-b97184e9ed2a.png">

This implementation is heavily based on the previously implemented [Swin Transformer V1](https://github.com/BVlad917/swin_transformer.git).

In essence, the second version of the Swin Transformer brings 3 key changes as far as the architecture is concerned:
1. __Post-Normalization.__ Swin Transformer V1 used __Pre-Normalization__, meaning that in a Swin Transformer Block we 
would have the inputs of the attention mechanism normalized, but the outputs could have arbitrary scale and these
outputs would be directly fed through a residual connection. This would cause activations in the layer layers to be 
much larger than the activations on the earlier layers and training would become unstable. To alleviate this, we 
move the layer normalization after the attention mechanism and the MLP layer so now the outputs of these layers 
which will be subsequently fed to the residual connection will be normalized.

2. __Scaled Cosine Attention.__ Swin Transformer V1 used the usual dot product attention pioneered in the original 
[Attention is all you need paper](https://arxiv.org/abs/1706.03762). This is the standard attention mechanism, but the 
authors of the Swin Transformer V2 paper found that if we use Scaled Cosine Attention the training will be more stable. 
This basically means that the query and keys will be normalized before applying scaled dot-product attention, 
everything remains the same. The authors note "The scaled cosine attention makes the computation irrelevant to 
amplitudes of block inputs, and the attention values are less likely to fall into extremes."

3. __Continuous Relative Position Bias with Log-spaced Coordinates.__ Swin Transformer V1 used relative positional 
bias instead of the usual absolute position bias used in the original 
[Vision Transformer](https://arxiv.org/abs/2010.11929). Instead of this, the Swin Transformer V2 uses Continuous 
Relative Position Bias with Log-spaced Coordinates which allows a smoother transition to higher resolution images
and higher window sizes after pre-training on a set image resolution and window size. Before this, in Swin Transformer
V1, the way this transition was made was through bicubic interpolation, which the authors found is suboptimal.

NB: Other changes such as supervised pre-training, zero-redundancy optimizer, activation checkpointing, and 
sequential self-attention are also used in Swin Transformer V2 but these changes are either not relevant to the 
architecture itself or have little impact so they are not discussed here.

