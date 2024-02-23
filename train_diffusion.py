import argparse
from pathlib import Path

import torch
import yaml
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from lightning_modules import LigandPocketDDPM
from utils import merge_configs, merge_args_and_yaml

# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--resume', type=str, default=None)
    args = p.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert 'resume' not in config
    
    # Get main config
    ckpt_path = None if args.resume is None else Path(args.resume)
    if args.resume is not None:
        resume_config = torch.load(
            ckpt_path, map_location=torch.device('cpu'))['hyper_parameters']

        config = merge_configs(config, resume_config)

    args = merge_args_and_yaml(args, config)
    
    diffusion_fn = None

    histogram_file = Path(args.datadir, 'size_distribution.npy')
    histogram = np.load(histogram_file).tolist()
    diffusion = LigandPocketDDPM(
        configs=args,
        node_histogram=histogram,
    )
    
    logger = pl.loggers.WandbLogger(
        save_dir=args.logdir,
        project='diffsbdd-ec',
        group=args.wandb_params.group,
        name=args.run_name,
        id=args.run_name,
        resume='must' if args.resume is not None else False,
        entity=args.wandb_params.entity,
        mode=args.wandb_params.mode,
    )

    outdir = Path(args.logdir, args.run_name)
    checkpoint_best_model = ModelCheckpoint(
        dirpath=Path(outdir, 'checkpoint'),
        filename="best-model-{epoch:02d}",
        monitor="loss/val",
        save_top_k=1,
        save_last=True,
        mode="min",
    )
   

    if args.gpus == 0:
        trainer = pl.Trainer(
            max_epochs=args.n_epochs,
            logger=logger,
            callbacks=[checkpoint_best_model],
            enable_progress_bar=args.enable_progress_bar,
            num_sanity_val_steps=args.num_sanity_val_steps,
            accelerator='cpu'
        )
    elif args.gpus == 1:
        trainer = pl.Trainer(
            max_epochs=args.n_epochs,
            logger=logger,
            callbacks=[checkpoint_best_model],
            enable_progress_bar=args.enable_progress_bar,
            num_sanity_val_steps=args.num_sanity_val_steps,
            accelerator='gpu',
            devices=1
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args.n_epochs,
            logger=logger,
            callbacks=[checkpoint_best_model],
            enable_progress_bar=args.enable_progress_bar,
            num_sanity_val_steps=args.num_sanity_val_steps,
            accelerator='gpu',
            devices=args.gpus,
            strategy='ddp'
        )

    trainer.fit(model=diffusion, ckpt_path=ckpt_path)

    
