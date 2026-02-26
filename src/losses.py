"""
losses.py

Contains the code for the different losses and their combinations used in the MorphEmbedd module.


The loss configuration was done in the code and not moved to its own config file, which is why not all configurations are present here.
"""

import torch


def get_triplets(
    embeddings,
    labels,
    has_labels,
):
    """
    Generates triplets for each embeddings:
        - `embeddings`: the embeddings to find triplets for
        - `labels`: the labels of the embeddings
        - `has_labels`: indicator for each embedding if it has a label (for training with mix of unlabeled and labeled data)

    Generated with ChatGPT
    """
    device = embeddings.device
    batch_size = embeddings.size(0)
    anchor_list, positive_list, negative_list = [], [], []
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)

    for i in range(batch_size):
        if not has_labels[i]:
            continue
        anchor = embeddings[i]
        anchor_label = labels[i]

        # Positives: same label but not the anchor itself
        pos_mask = (labels == anchor_label) & (
            torch.arange(batch_size, device=device) != i
        )
        pos_indices = pos_mask.nonzero(as_tuple=False).view(-1)
        if pos_indices.numel() == 0:
            continue  # skip if no positive in batch
        # Negatives: different label
        neg_mask = labels != anchor_label
        neg_indices = neg_mask.nonzero(as_tuple=False).view(-1)

        if neg_indices.numel() == 0:
            continue  # skip if no negative
        # Pick one positive and one negative (randomly)
        pos_idx = pos_indices[torch.randint(0, pos_indices.numel(), (1,))]
        neg_idx = neg_indices[torch.randint(0, neg_indices.numel(), (1,))]

        anchor_list.append(anchor)
        positive_list.append(embeddings[pos_idx])
        negative_list.append(embeddings[neg_idx])

    # Stack to get tensors of shape [num_triplets, embedding_dim]
    if len(anchor_list) == 0:
        return None, None, None

    return (
        torch.stack(anchor_list),
        torch.stack(positive_list).squeeze(1),
        torch.stack(negative_list).squeeze(1),
    )


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
        # self.latent_loss = torch.nn.TripletMarginLoss()
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
        non_domain_latents,
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
            - `non_domain_latents`: the latent variables
        """
        reconstruction = self.z_reconstruction * self.reconstruction_loss(
            xs, xs_reconstructed
        )
        domain = self.z_domain * self.domain_loss(domains_predicted, domains)
        # anchors, positives, negatives = get_triplets(
        #     non_domain_latents, labels, has_labels
        # )
        # if anchors is not None:
        #    domain = self.latent_loss(anchors, positives, negatives)
        # else:
        #    domain = 0
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
