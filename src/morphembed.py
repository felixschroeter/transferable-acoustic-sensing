"""
Contains a LightningModule (MorphEmbedModel) that encapsulates the training, validation, and testing code.
It also handles logging and the evaluation
"""

import lightning
import torch
import mlflow
import pandas as pd

import evaluation
import models
import losses


class MorphEmbedModel(lightning.LightningModule):
    """
    lightning module that handles the train and evaluation loops
    """

    def __init__(self, model_name, model_kwargs, batch_size):
        """
        initialize the module with:
            - `model_name`: name of the model from config
            - `model_kwargs`: args from the config to pass on to the model
            - `batch_size`: the batch size (only required so the logging works correctly)
        """
        super().__init__()
        # save the hyperparameters to make the reloadable
        self.save_hyperparameters()

        self.model_name = model_name

        if model_name == "mlp":
            self.model = models.MLP(**model_kwargs)
        elif model_name == "identity_model":
            self.model = models.Identity()

        # hard-coded parameters -> move to config at some point
        self.num_domain_latents = 2
        self.alpha = 0.5
        self.beta = 0
        self.gamma = 0.5

        self.loss = losses.CombinedLoss(self.alpha, self.beta, self.gamma)

        # helper dict to collect outputs
        self.test_outputs = {}

        # set batch_size for logging
        self.batch_size = batch_size

    def training_step(self, batch):
        """
        Perform a training step for `batch`

        Returns training loss
        """
        # unpack batch
        xs, domains, labels, has_labels = (
            batch["x"],
            batch["domain"],
            batch["label"],
            batch["has_label"],
        )

        # forward pass
        (
            latents,
            non_domain_latents,
            xs_reconstructed,
            domains_predicted,
            labels_predicted,
        ) = self.model(xs)

        # compute loss
        loss, losses = self.loss(
            xs,
            xs_reconstructed,
            domains,
            domains_predicted,
            labels,
            labels_predicted,
            has_labels,
            non_domain_latents,
        )

        # log the losses
        losses = {"train_" + key: value for key, value in losses.items()}
        self.log_dict(losses, on_step=True, on_epoch=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch):
        """
        Validates the current model on one `batch`
        Return validation loss
        """
        # unpack batch
        xs, domains, labels, has_labels = (
            batch["x"],
            batch["domain"],
            batch["label"],
            batch["has_label"],
        )

        # forward pass
        (
            latents,
            non_domain_latents,
            xs_reconstructed,
            domains_predicted,
            labels_predicted,
        ) = self.model(xs)

        # calculate loss
        loss, losses = self.loss(
            xs,
            xs_reconstructed,
            domains,
            domains_predicted,
            labels,
            labels_predicted,
            has_labels,
            non_domain_latents,
        )

        # log the losses
        losses = {"val_" + key: value for key, value in losses.items()}
        self.log_dict(losses, on_step=True, on_epoch=True, batch_size=self.batch_size)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """
        For a `batch` with `batch_idx` from dataloader `dataloader_idx`, pass all samples through the model
        and collect their embeddings

        Got the idea to collect the outputs in a dict by dataloader from asking ChatGPT
        """
        # unpack batch
        xs, labels = batch["x"], batch["label"]

        # forward pass
        latents, non_domain_latents, xs_reconstructed, _, _ = self.model(xs)

        # add the embeddings to the outputs dictionary
        if dataloader_idx not in self.test_outputs:
            self.test_outputs[dataloader_idx] = {
                "embeddings": [],
                "labels": [],
            }
        self.test_outputs[dataloader_idx]["embeddings"].append(non_domain_latents)
        self.test_outputs[dataloader_idx]["labels"].append(labels)

    def on_test_epoch_end(self):
        """
        Run the evaluation on the embeddings.

        Is run after all test samples were passed through the test_step function.
        """

        # get the idx to map back the datasets to the tasks and fingers
        regression_idx = self.trainer.datamodule.test_regression_dataloader_idx
        classification_idx = self.trainer.datamodule.test_classification_dataloader_idx
        finger_idx = self.trainer.datamodule.test_dataset_finger_idx

        # create a dataframe from all the embeddings
        dataframes = []
        for dataloader_idx, outputs in self.test_outputs.items():
            # Concatenate all predictions and targets for this dataloader
            embeddings = torch.cat(outputs["embeddings"], dim=0).tolist()
            labels = torch.cat(outputs["labels"], dim=0).tolist()
            df = pd.DataFrame({"embeddings": embeddings, "labels": labels})
            df["finger_id"] = finger_idx[dataloader_idx]
            if dataloader_idx in regression_idx:
                df["task"] = "regression"
            elif dataloader_idx in classification_idx:
                df["task"] = "classification"
            else:
                raise NotImplementedError()
            dataframes.append(df)
        all_results_df = pd.concat(dataframes, axis=0)

        # log the dataframe to MLFlow and save it locally
        all_results_df.to_parquet("./temp/all_embeddings.parquet", index=False)
        mlflow.log_artifact("./temp/all_embeddings.parquet")
        mlflow.log_table(
            data=all_results_df,
            artifact_file="all_results_df.json",
        )  # log again as table so it also be inspected in the WebUI

        # create a PCA plot from the embeddings for the regression datasets
        pca_regression = evaluation.embedding_pca(
            all_results_df,
            "regression",
            "PCA of regression embeddings by finger and label",
        )
        mlflow.log_figure(pca_regression, "pca_regression.png")

        # create a PCA plot from the embeddings for the classification datasets
        pca_classification = evaluation.embedding_pca(
            all_results_df,
            "classification",
            "PCA of classification embeddings by finger and label",
        )
        mlflow.log_figure(pca_classification, "pca_classification.png")

        # create a T-SNE plot from the embeddings for the regression datasets
        tsne_regression = evaluation.embedding_tsne(
            all_results_df,
            "regression",
            "t-SNE of regression embeddings by finger and label",
        )
        mlflow.log_figure(tsne_regression, "tsne_regression.png")

        # create a T-SNE plot from the embeddings for the classification datasets
        tsne_classification = evaluation.embedding_tsne(
            all_results_df,
            "classification",
            "t-SNE of classification embeddings by finger and label",
        )
        mlflow.log_figure(tsne_classification, "tsne_classification.png")

        # run the evaluation pipeline and collect the results
        results_classification, results_regression = (
            evaluation.run_downstream_evaluation(all_results_df)
        )

        mlflow.log_table(
            data=results_classification,
            artifact_file="test_results_classification.json",
        )
        mlflow.log_table(
            data=results_regression, artifact_file="test_results_regression.json"
        )

    def configure_optimizers(self):
        """
        Initialize the optimizer used in the training

        returns the optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
