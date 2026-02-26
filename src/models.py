"""
models.py

Contains everything related to the neural networks used in this work, the corresponding torch.Modules
and functions to create them, mainly the main neural network `MLP`, the gradient reversal layer,
and the dummy drop-in `Identity`

"""

import torch.nn as nn
import torch.autograd as autograd


def create_mlp(
    layer_dims=[],
    activation=nn.ReLU,
    use_dropout=False,
    dropout_p=0.2,
    use_batch_norm=False,
):
    """
    a helper function to create a fully-connected multi-layer perceptron.
    - `layer_dims`: a list of the dimensions of each layer
    - `activation`: the activation function to use between layers (default ReLU)
    - `use_dropout`: if dropouts should be introduced after each layer
    - `dropout_p`: probability parameter for the dropout
    - `use_batch_norm`: wether to use batchnorm after each layer

    returns a list of the defined layers that can be passed to nn.Sequential
    """
    layers = []
    for in_features, out_features in zip(layer_dims[:-1], layer_dims[1:]):
        # Add a linear layer
        layers.append(nn.Linear(in_features, out_features))

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        # Add activation if specified
        if activation is not None:
            layers.append(activation())

        # Add dropout if specified
        if use_dropout:
            layers.append(nn.Dropout(p=dropout_p))
    return layers


class GradReverse(autograd.Function):
    """
    The gradient reversal layer, acts as an identity layer on the forward pass,
    but returns the negated gradients on the backward pass.

    source for this implementation:
        https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/6

    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


class MLP(nn.Module):
    """
    The neural network to learn domain invariant representations.
    It consists of an autoencoder (encoder, decoder), a regression head, and an adversarial domain classifier.

    """

    def __init__(
        self,
        sample_dims,
        num_domains,
        latent_dims,
        domain_latent_dims,
        encoder_hidden_dims,
        decoder_hidden_dims,
        domain_classifier_hidden_dims,
        apply_transform,
        use_batch_norm=False,
        use_GeLU=False,
    ):
        """
        Create a new network that should learn domain invariant representations.
        - `sample_dims`: dimensions of the inputs
        - `num_domains`: the total number of different domains (actuators)
        - `latent_dims`: size of the latent dimension
        - `domain_latent_dims`: the number of latents that is allowed to contain domain-specific features
        - `encoder_hidden_dims`: list of the hidden dims of the encoder
        - `decoder_hidden_dims`: list of the hidden dims of the decoder
        - `domain_classifier_hidden_dims`: list of the hidden dims of the domain_classifier
        - `apply_transform`: if 'fft' or 'stft' is applied to the data
        - `use_batch_norm`: if the model shoudl use batch norm (currently unused)
        - `use_GeLU`: if the model should use GeLU instead of ReLU
        """
        super().__init__()
        self.domain_latent_dims = domain_latent_dims

        # adapt dimensions in case preprocessing is applied
        if apply_transform == "fft":
            sample_dims = sample_dims // 2

        if apply_transform == "stft":
            sample_dims = 2049

        sample_dims = 2049

        activation = nn.GELU if use_GeLU else nn.ReLU
        encoder_layers = create_mlp(
            layer_dims=[sample_dims, *encoder_hidden_dims],
            activation=activation,
            use_dropout=False,
            use_batch_norm=False,
        )
        encoder_layers.append(nn.Linear(encoder_hidden_dims[-1], latent_dims))
        encoder_layers.append(nn.BatchNorm1d(latent_dims))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = create_mlp(
            layer_dims=[latent_dims, *decoder_hidden_dims],
            activation=activation,
        )
        decoder_layers.append(nn.Linear(decoder_hidden_dims[-1], sample_dims))
        self.decoder = nn.Sequential(*decoder_layers)

        domain_classifier_layers = create_mlp(
            layer_dims=[
                latent_dims - domain_latent_dims,
                *domain_classifier_hidden_dims,
            ],
            activation=activation,
        )
        domain_classifier_layers.append(
            nn.Linear(domain_classifier_hidden_dims[-1], num_domains)
        )
        self.domain_classifier = nn.Sequential(*domain_classifier_layers)

        regressor_layers = create_mlp(layer_dims=[latent_dims, 32, 8])
        regressor_layers.append(nn.Linear(8, 1))
        self.regressor = nn.Sequential(*regressor_layers)

    def forward(self, xs):
        """
        forward pass through the network.

        Returns a tuple of:
            - latents
            - non_domain_latents (latents without domain-specific information)
            - xs_reconstructed
            - domain_predicted
            - regression_predicted
        """
        latents = self.encoder(xs)
        xs_reconstructed = self.decoder(latents)

        domain_predicted = self.domain_classifier(
            grad_reverse(latents[:, : -self.domain_latent_dims])
        )
        regression_predicted = self.regressor(latents).flatten()
        non_domain_latents = latents[:, : -self.domain_latent_dims]

        return (
            latents,
            non_domain_latents,
            xs_reconstructed,
            domain_predicted,
            regression_predicted,
        )


class Identity(nn.Module):
    """
    Dummy Module that creates identity mapping embeddings
    """

    def __init__(self):
        super().__init__()

    def forward(self, xs):

        # returns xs and more xs as dummy values so that the tuple unpacking does not break.
        return xs, xs, xs, xs, xs
