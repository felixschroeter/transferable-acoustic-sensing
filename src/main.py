"""
main.py

The main script to run the training loop for the neural network based approach,
or generate the normalized samples for the other approaches.

usage (from project root):
    - sh run_mlflow.sh  # start MLFlow server (required)
    - uv run src/main.py  # run training script

For advanced cli options for overriding the config on invokation, refer to the hydra.cc docs:
    https://hydra.cc/docs/intro/

"""

import pprint
import hydra
import torch
import lightning
import mlflow

from omegaconf import OmegaConf
from mlflow.pytorch import MlflowModelCheckpointCallback

import datamodule
import morphembed


# automatically loads the config from the files in src/conf
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:

    # print the config
    pprint.pprint(OmegaConf.to_container(cfg=cfg, resolve=True))

    # initialize the datamodule
    dm = datamodule.AudioDataModule(
        model_name=cfg["model"]["name"],
        data_basedir=cfg["data_basedir"],
        train_datasets=cfg["train_datasets"],
        train_regression_datasets=cfg["train_regression_datasets"],
        test_datasets=cfg["test_datasets"],
        test_classification_datasets=cfg["evaluate_classification_datasets"],
        test_regression_datasets=cfg["evaluate_regression_datasets"],
        datasets=cfg["datasets"],
        **cfg["data"]["kwargs"],
        test_size=cfg["training"]["test_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        persistent_workers=cfg["training"]["persistent_workers"],
        prefetch_factor=cfg["training"]["prefetch_factor"],
        pin_memory=cfg["training"]["pin_memory"],
        random_seed=cfg["random_seed"],
        apply_norm=cfg["apply_norm"],
    )

    torch.set_float32_matmul_precision("high")

    # initialize the model from the config
    model_kwargs = {
        **cfg["model"]["kwargs"],
        "sample_dims": cfg["data"]["kwargs"]["window_size"],
        "num_domains": len(cfg["train_datasets"]),
        "apply_transform": cfg["data"]["kwargs"]["apply_transform"],
    }
    model = morphembed.MorphEmbedModel(
        model_name=cfg["model"]["name"],
        model_kwargs=model_kwargs,
        batch_size=cfg["training"]["batch_size"],
    )

    # configure early stopping based on validation loss
    early_stopping = lightning.pytorch.callbacks.early_stopping.EarlyStopping(  # pyright: ignore[reportAttributeAccessIssue]
        monitor="val_loss", patience=20
    )

    # configure MLFLow logging
    mlflow.pytorch.autolog(checkpoint=True)  # pyright: ignore[notExportedIssue]
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow_checkpoint_callback = MlflowModelCheckpointCallback(
        monitor="val_loss", save_best_only=True
    )

    # configure the trainer
    trainer = lightning.Trainer(
        callbacks=[early_stopping, mlflow_checkpoint_callback],
        log_every_n_steps=100,
        default_root_dir=cfg["output_dir"],
        max_epochs=cfg["training"]["max_epochs"],
    )

    # start a new MLFlow run and start training
    with mlflow.start_run(
        description=f"{cfg['model']['name']} for {cfg['experiment_name']}"
    ):
        # log the config to MLFlow
        mlflow.log_text(OmegaConf.to_yaml(cfg), "hydra_config.yaml")

        # the identity model simply returns the inputs unprocessed and does not need to be trained
        if not cfg["model"]["name"] == "identity_model":
            # run training
            trainer.fit(model=model, datamodule=dm)

        # run evaluation
        trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
