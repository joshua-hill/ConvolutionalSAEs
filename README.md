##Convolutional Sparse Auto-Encoders

We train a convolutional variation of sparse auto-encoders on the activations of AlexNet. This architecture should be able to ~localize the prescence of features in an image (?).

We implement the encoder and decoders of the SAE as conv-nets. Our encoder is of dimension (c x d x 1 x 1), transforming the dimension of the activation to (B, D, H, W). The decoder is of dimension (d x c x 1 x 1).


Inspired by post by Apollo Research: 
