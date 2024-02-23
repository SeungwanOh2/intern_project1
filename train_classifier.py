import argparse
import pickle
import yaml
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from lightning_modules import LigandPocketDDPM, PocketClassifier
from utils import merge_configs, merge_args_and_yaml
    
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
    
    diffusion = LigandPocketDDPM.load_from_checkpoint(args.diffusion)
    diffusion.freeze()
    
    def diffusion_fn(ligand, pocket, timestep=None):
        return diffusion.ddpm.sample_noised(ligand, pocket, timestep=timestep)
    
    label_hist = None
    if args.label_hist is not None:
        with open(args.label_hist, 'rb') as f:
            label_hist = pickle.load(f)
            
    classifier = PocketClassifier(
        configs=args,
        diffusion=diffusion_fn,
        label_hist=label_hist
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
    
    early_stopping = EarlyStopping(
        monitor="loss/val",
        patience=args.max_patience,
        verbose=False,
        mode="min")
    
    outdir = Path(args.logdir, args.run_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(outdir, 'checkpoint'),
        filename="best-model-epoch={epoch:02d}",
        monitor="loss/val",
        save_top_k=1,
        save_last=True,
        mode="min",
    )

    if args.gpus == 0:
        trainer = pl.Trainer(
            max_epochs=args.n_epochs,
            logger=logger,
            callbacks=[checkpoint_callback, early_stopping],
            enable_progress_bar=args.enable_progress_bar,
            num_sanity_val_steps=args.num_sanity_val_steps,
            accelerator='cpu'
        )
    elif args.gpus == 1:
        trainer = pl.Trainer(
            max_epochs=args.n_epochs,
            logger=logger,
            callbacks=[checkpoint_callback, early_stopping],
            enable_progress_bar=args.enable_progress_bar,
            num_sanity_val_steps=args.num_sanity_val_steps,
            accelerator='gpu',
            devices=1
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args.n_epochs,
            logger=logger,
            callbacks=[checkpoint_callback, early_stopping],
            enable_progress_bar=args.enable_progress_bar,
            num_sanity_val_steps=args.num_sanity_val_steps,
            accelerator='gpu',
            devices=args.gpus,
            strategy='ddp'
        )

    trainer.fit(model=classifier, ckpt_path=ckpt_path)
    
    # classifier.test_visualize()
        
        