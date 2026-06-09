"""
losses.py

Contains the code for the different losses and their combinations used in the MorphEmbedd module.


The loss configuration was done in the code and not moved to its own config file, which is why not all configurations are present here.
"""

import torch


class CombinedLoss(torch.nn.Module):
    """
    The fully combined loss function

    The loss configuration was done in the code and not moved to its own config file, which is why not all configurations are present here.
    """

    def __init__(self, alpha, beta, gamma):
        """
        Initialize the loss with weights for the different components:
            - `alpha`: weight for the reconstruction loss
            - `beta`: weight for the domain loss
            - `gamma`: weight for the regression loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # weights to equalize the loss magnitude based on observed values from
        # initial training run
        self.z_reconstruction = 1
        self.z_domain = 1
        self.z_regression = torch.tensor(0.075)

        self.reconstruction_loss = torch.nn.MSELoss()
        self.domain_loss = torch.nn.CrossEntropyLoss()
        self.regression_loss = torch.nn.MSELoss()

    def forward(
        self,
        xs,
        xs_reconstructed,
        domains,
        domains_predicted,
        labels,
        labels_predicted,
        has_labels,
    ):
        """
        Calculate the different sublosses and the overall loss from all necessary inputs:
            - `xs`: the sample
            - `xs_reconstructed`: the reconstructed sample
            - `domains`: the actual domain
            - `domains_predicted`: the predicted domain
            - `labels`: the regression labels
            - `labels_predicted`: the predicted regression labels
            - `has_labels`: helper variable that indicates if regression labels are available or not
        """
        reconstruction = self.z_reconstruction * self.reconstruction_loss(
            xs, xs_reconstructed
        )
        domain = self.z_domain * self.domain_loss(domains_predicted, domains)
        regression = self.z_regression * self.regression_loss(
            has_labels * labels, has_labels * labels_predicted
        )  # only consider the labeled examples

        loss = (
            self.alpha * reconstruction + self.beta * domain + self.gamma * regression
        )

        losses = {
            "loss": loss,
            "reconstruction": reconstruction,
            "domain": domain,
            "regression": regression,
        }
        return loss, losses
