# based on: https://github.com/karpathy/nanoGPT/blob/master/bench.py
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from train import build_model, build_optimizer, get_number_of_params

# -----------------------------------------------------------------------------
batch_size = 1024
image_size = 224
num_classes = 1000
seed = 3407
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16'  # 'float32' or 'bfloat16' or 'float16'
compile = True  # use PyTorch 2.0 to compile the model to be faster
profile = False  # use pytorch profiler, or just simple benchmarking?

model_name = "espatat"  # "simplevit", "espatat"
model_cfg = dict(
    image_size=image_size,
    patch_size=16, 
    num_classes=num_classes,
    dim=384,
    depth=12,
    heads=6,
    mlp_dim=384 * 4,
    drop_p=0.,
    c_val_0_target=0.9,  # TAT 
    gamma_max_depth=0.005,  # E-SPA
    att_bias = True,
    ff_biase = True
)   

opt_name = "shampoo"  # "adam"
opt_cfg = dict(
    lr=0.0007,
    weight_decay=0.,
    epsilon=1e-6,
    precondition_frequency=1,
    start_preconditioning_step=10
)
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type.split(":")[0], dtype=ptdtype)

def get_timer():
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    return starter, ender

x = torch.normal(0, 1, size=(batch_size, 3, image_size, image_size), device=device)
y = torch.torch.randint(0, num_classes, (batch_size,), device=device, dtype=torch.long)
get_batch = lambda: (x, y)

model = build_model(model_name, model_cfg)
model = model.to(device)
print("Number of parameters: ", get_number_of_params(model))

optimizer = build_optimizer(model, opt_name, opt_cfg)

if compile:
    print("Compiling model...")
    model = torch.compile(model)  # mode="max-autotune"

if profile:
    # useful docs on pytorch profiler:
    # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    wait, warmup, active = 5, 5, 5
    num_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        record_shapes=False,
        profile_memory=True,
        with_stack=False,  # incurs an additional overhead, disable if not needed
        with_flops=True,
        with_modules=False,  # only for torchscript models atm
    ) as prof:

        for k in range(num_steps):
            X, Y = get_batch()
            with ctx:
                output = model(x)
                loss = F.cross_entropy(output, Y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")

            prof.step()  # notify the profiler at end of each step

else:
    # simple benchmarking
    torch.cuda.synchronize()
    for stage, num_steps in enumerate([10, 20]):  # burnin, then benchmark
        starter, ender = get_timer()
        starter.record()
        for k in range(num_steps):
            X, Y = get_batch()
            with ctx:
                output = model(x)
                loss = F.cross_entropy(output, Y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")

        ender.record()
        torch.cuda.synchronize()
        total_time = 1e-3 * starter.elapsed_time(ender)
        print(f"Step at {stage} took {total_time/num_steps:.4f} seconds")
    peak_memory = torch.cuda.max_memory_allocated(device=device)
    print(f"Peak GPU memory usage: {peak_memory / 1024**2:.2f} MB")