import copy
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import partial
from os.path import join as join_path
from statistics import median

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from models import SimplEsTViT, SimpleViT
from shampoo.shampoo import Shampoo
from shampoo.shampoo_utils import GraftingType
from tricks import SAM, compute_ema, mix_mixup


# Config
@dataclass
class Config:
    seed: int = 3407
    # wandb logging
    wandb_log: bool = False  # disabled by default
    wandb_project: str = 'nanovit'
    wandb_run_name: str = ''
    # training
    epochs: int = 90
    num_warmup_steps: int = 10_000
    eval_step: int = 1251 * 3
    grad_accumulation_steps: int = 2
    train_metrics_window_size: int = 128
    grad_clip: float = 1.0
    lb_smooth: float = 0.  # [0.0, 1.0]
    checkpointing: bool = True
    # mixed precision
    device: str = "cuda"
    dtype: str = "bfloat16"
    mixp_enabled: bool = True
    # data
    batch_size: int = 512
    data_name: str = "imagenet"  # "cifar10", "cifar100", "imagenet200", "imagenet"
    augment: bool = True
    num_classes: int = 1000  # 10, 100, 200, 1000
    # model
    model_name: str = "espatat"  # "simplevit", "espatat"
    model_cfg: dict = field(
        default_factory=lambda: dict(
            image_size=224,  # 32, 32, 64, 224
            patch_size=16,   # 4, 4, 8, 16
            num_classes=1000,  # 10, 100, 200, 1000
            dim=384,
            depth=12,
            heads=6,
            mlp_dim=384 * 4,
            drop_p=0.,
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
            lr=0.0003,  # 0.0005
            weight_decay=0.00001,  # 0.00005
            precondition_frequency=25,
            start_preconditioning_step=25,
        )
    ) 
    # tricks
    mixup: bool = True
    ema: bool = False
    ema_step: int = 5
    sam: bool = False
    sam_step: int = 1  
    sam_rho: float = 0.05  # if sam_step=10 then rho=0.5

    def to_dict(self):
        return asdict(self)

# Data
def get_data(batch_size, data_name='cifar10', augment=False) -> dict:
    def _get_image_folder(data_name, root, train, download, transform=None):
        return datasets.ImageFolder(root=join_path(root, data_name, 'train' if train else 'val'), transform=transform)
    
    dataset = {
        'cifar10': datasets.CIFAR10, 
        'cifar100': datasets.CIFAR100, 
        'imagenet200': partial(_get_image_folder, data_name='tiny-imagenet-200'),
        'imagenet': partial(_get_image_folder, data_name='imagenet'),

    }[data_name]
    mean = {
        'cifar10': (0.4914, 0.4822, 0.4465), 
        'cifar100': (0.5071, 0.4867, 0.4408), 
        'imagenet200': (0.4802, 0.4481, 0.3975), 
        'imagenet': (0.485, 0.456, 0.406)
    }[data_name]
    std = {
        'cifar10': (0.2023, 0.1994, 0.2010), 
        'cifar100': (0.2675, 0.2565, 0.2761), 
        'imagenet200': (0.2770, 0.2691, 0.2821), 
        'imagenet': (0.229, 0.224, 0.225)
    }[data_name]
    size = {
        'cifar10': 32, 
        'cifar100': 32, 
        'imagenet200': 64, 
        'imagenet': 224
    }[data_name]
    root = 'data'
    
    # Define transforms
    t_augment_list = [] if not augment else [
        transforms.RandomCrop(size=size, padding=4) if data_name != 'imagenet' else transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=10)
    ]

    t_eval_list = [] if data_name != 'imagenet' else [
        transforms.Resize(256), 
        transforms.CenterCrop(size)
    ]

    t_esential_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    train_transform = transforms.Compose(t_augment_list + t_esential_list)
    test_transform = transforms.Compose(t_eval_list + t_esential_list)

    # Define trainloaders
    trainloader = torch.utils.data.DataLoader(
        dataset(root=root, train=True, download=True, transform=train_transform),
        batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4, drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        dataset(root=root, train=False, download=True, transform=test_transform),
        batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4
    )
    return trainloader, testloader

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
def estimate_metrics(model, loaders, device, ctx):
    metrics = {}
    model.eval()
    for split, loader in loaders:
        running_loss, running_acc = 0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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

    trainloader, testloader = get_data(cfg.batch_size, cfg.data_name, cfg.augment)
    
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
        x, y = next(dataloader_loopy)
        x, y = x.to(cfg.device, non_blocking=True), y.to(cfg.device, non_blocking=True)
        if cfg.mixup: x, y = mix_mixup(x, y, cfg.num_classes, alpha=0.2)
        return x, y
   
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
