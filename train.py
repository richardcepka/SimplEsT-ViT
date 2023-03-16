import copy
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from os.path import join as join_path
from statistics import median

import numpy as np
import torch
import torch.nn.functional as F
from ffcv.fields.basics import IntDecoder
from ffcv.fields.rgb_image import (CenterCropRGBImageDecoder,
                                   RandomResizedCropRGBImageDecoder)
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import (NormalizeImage, RandomHorizontalFlip, Squeeze,
                             ToDevice, ToTensor, ToTorchImage)

from models import SimplEsTViT, SimpleViT
from shampoo.shampoo import Shampoo
from shampoo.shampoo_utils import GraftingType
from tricks import SAM, compute_ema


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255


# Config
@dataclass
class Config:
    seed: int = 3407
    # wandb logging
    wandb_log: bool = False  # disabled by default
    wandb_project: str = 'nanovit'
    wandb_run_name: str = 'imagenet/shampoo/TAT-setup'
    # training
    epochs: int = 90
    num_warmup_steps: int = 10_000
    eval_step: int = 1251 * 3
    grad_accumulation_steps: int = 2
    train_metrics_window_size: int = 128
    grad_clip: float = 1.0
    lb_smooth: float = 0.1  # [0.0, 1.0]
    checkpointing: bool = True
    # mixed precision
    device: str = "cuda"
    dtype: str = "bfloat16"
    mixp_enabled: bool = True
    # data
    batch_size: int = 512
    n_workers: int = 16
    # model
    model_name: str = "espatat"  # "simplevit", "espatat"
    model_cfg: dict = field(
        default_factory=lambda: dict(
            image_size=224,
            patch_size=16,
            num_classes=1000,
            dim=384,
            depth=12,
            heads=6,
            mlp_dim=384 * 4,
            drop_p=0.2,
            c_val_0_target=0.9,  # TAT 
            gamma_max_depth=0.005,  # E-SPA
            att_bias=True,
            ff_biase=True,
        )
    )
    # optimizer
    opt_name: str = "shampoo"
    opt_cfg: dict = field(
        default_factory=lambda: dict(
            lr=0.0005,  # 0.0005
            weight_decay=0.00001,  # 0.00005
            precondition_frequency=25,
            start_preconditioning_step=25,
        )
    ) 
    # tricks
    ema: bool = False
    ema_step: int = 5
    sam: bool = False
    sam_step: int = 1  
    sam_rho: float = 0.05  # if sam_step=10 then rho=0.5

    def to_dict(self):
        return asdict(self)

# Data
def create_train_loader(path, batch_size, in_memory, dtype, device, num_workers=16):
    _res_tuple = (224, 224)
    image_pipeline = [
        RandomResizedCropRGBImageDecoder(_res_tuple),
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, dtype)
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True)
    ]

    order = OrderOption.RANDOM if in_memory else OrderOption.QUASI_RANDOM
    loader = Loader(path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    os_cache=in_memory,
                    drop_last=True,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    })
    return loader

def create_val_loader(path, batch_size, dtype, device, num_workers=16):
    _res_tuple = (224, 224)
    _ratio = 224 / 256
    cropper = CenterCropRGBImageDecoder(_res_tuple, ratio=_ratio)
    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, dtype)
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True)
    ]

    loader = Loader(path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=OrderOption.SEQUENTIAL,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    })
    return loader

# Optimizer
def build_optimizer(model, name, opt_cfg: dict, sam=False, sam_rho=0.05):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optim_groups = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)]},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    if name == "adam":
        optimizer = torch.optim.Adam(optim_groups, **opt_cfg, fused=True)
    elif name == "shampoo":
        optimizer = Shampoo(
            optim_groups, **opt_cfg,
            betas=(0.9, 0.999),
            max_preconditioner_dim=8192,
            use_decoupled_weight_decay=False,
            grafting_type=GraftingType.ADAM,
            grafting_epsilon=1e-08,
            grafting_beta2=0.999,
            epsilon=1e-12
        )
    return SAM(optimizer, rho=sam_rho) if sam else optimizer 
    
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

# Model
def build_model(model_name, model_cfg: dict):
    if model_name == "espatat":
        return SimplEsTViT(**model_cfg)
    elif model_name == "simplevit":
        return SimpleViT(**model_cfg)

# Utils
def loopy(dl):
    while True:
        for x in dl: yield x 

def acc(output, target): return torch.sum(output.argmax(1) == target) / len(target)

def set_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_number_of_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)    

@torch.no_grad()
def estimate_metrics(model, loaders, ctx):
    metrics = {}
    model.eval()
    for split, loader in loaders:
        running_loss, running_acc = 0, 0
        for x, y in loader:
            with ctx:
                output = model(x)
                loss = F.cross_entropy(output, y)
            running_loss += loss.item()
            running_acc += acc(output, y).item()
        metrics[f'{split}/loss'] = running_loss / len(loader)
        metrics[f'{split}/acc'] = running_acc / len(loader)
    model.train()
    return metrics


def main(cfg):
    
    # logging
    if cfg.wandb_log:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=cfg.to_dict())
        cfg = wandb.config

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    set_seed(cfg.seed)

    if cfg.checkpointing and (not os.path.exists("checkpoints")): os.makedirs("checkpoints")

    _in_memory = True
    trainloader = create_train_loader(
        join_path('data', 'ffcv_imagenet', 'train'), 
        ctx.batch_size, _in_memory, ptdtype, ctx.device, num_workers=ctx.n_workers)
    testloader = create_val_loader(
        join_path('data', 'ffcv_imagenet', 'train'), 
        ctx.batch_size, ptdtype, ctx.device, num_workers=ctx.n_workers)

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.dtype]
    ctx = torch.autocast(enabled=cfg.mixp_enabled, device_type=cfg.device.split(":")[0], dtype=ptdtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == 'float16'))

    model = build_model(cfg.model_name, cfg.model_cfg).to(cfg.device)
    if cfg.ema: ema_model = copy.deepcopy(model)

    optimizer = build_optimizer(model, cfg.opt_name, cfg.opt_cfg, sam=cfg.sam, sam_rho=cfg.sam_rho)
    num_steps = (cfg.epochs * len(trainloader)) // cfg.grad_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.num_warmup_steps, num_steps, num_cycles=0.5, last_epoch=-1)

    model = torch.compile(model)  # mode="max-autotune"
    model.train()

    print(
        f"Number of parameters: {get_number_of_params(model)}",
        f"Number of steps: {num_steps}",
        sep="\n"
    )
        
    dataloader_loopy = loopy(trainloader)
    def get_batch():
        return next(dataloader_loopy)
   
    x, y = get_batch()
    train_metrics = defaultdict(list)

    t_all = time.time()
    for step in range(1, num_steps + 1):
        # _________________________
        # forward-backward
        def fw_bw(x, y):
            optimizer.zero_grad(set_to_none=True)
            _loss = 0
            for _ in range(cfg.grad_accumulation_steps):
                with ctx:
                    output = model(x)
                    loss = F.cross_entropy(output, y, label_smoothing=cfg.lb_smooth) 
                    loss = loss / cfg.grad_accumulation_steps

                # async prefetch next batch while model is doing the forward pass on the GPU
                # will be in GPU memory during evaluation :(
                x, y = get_batch()
                
                _loss += loss.item()
                scaler.scale(loss).backward()

            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip) 
            return _loss, x, y

        if isinstance(optimizer, SAM):
            step_type = ("first", "second") if (step % cfg.sam_step) == 0 else ("skip",)
            for s in step_type:
                _loss, x, y = fw_bw(x, y)
                scaler.step(optimizer, step_type=s)
                scaler.update()
        else:
            _loss, x, y = fw_bw(x, y)
            scaler.step(optimizer)
            scaler.update()

        train_metrics["loss"].append(_loss)
        scheduler.step()
        # __________________________

        # _________________________
        # aggregate training metrics
        if (step % cfg.train_metrics_window_size) == 0: 
            agg_loss = median(train_metrics["loss"])
            
            print(
                f"step {step}/{num_steps}: train/loss-med{cfg.train_metrics_window_size} {agg_loss:.4f}", 
                sep=", "
            ) 
            if cfg.wandb_log:
                wandb.log(
                    {
                    f"train/loss-media-{cfg.train_metrics_window_size}": agg_loss,
                    "lr": scheduler.get_last_lr()[0]
                    }, 
                step=step)
            train_metrics = defaultdict(list)
        # _________________________

        if cfg.ema and (step % cfg.ema_step) == 0: compute_ema(model, ema_model, smoothing=0.99)
        # _________________________
        # evaluate model and validation dataset
        if (step % cfg.eval_step) == 0 or step == num_steps:
            metrics = estimate_metrics(
                ema_model if cfg.ema else model, [('val', testloader)], cfg.device, ctx
            )
            metrics['time'] = time.time() - t_all
            print(
                f"step {step}/{num_steps}: val/acc {metrics['val/acc']:.4f}",
                f"val/loss {metrics['val/loss']:.4f}",  
                f"time {metrics['time']:.4f}s",
                sep=", "
            )
            if cfg.wandb_log: wandb.log(metrics, step=step)
            if cfg.checkpointing:
                torch.save(
                    {
                    'step': step,
                    'model_state_dict': (ema_model if cfg.ema else model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': metrics['val/acc'],
                    }, 
                    f'checkpoints/model{step}.pth'
                )
                if cfg.wandb_log:
                    model_artifact = wandb.Artifact(f"model-checkpoint-{step}", type="model")
                    model_artifact.add_file(f'checkpoints/model{step}.pth')
                    wandb.save(f'checkpoints/model{step}.pth')
                    wandb.run.log_artifact(model_artifact)
        # _________________________

if __name__ == "__main__":
    main(Config())
