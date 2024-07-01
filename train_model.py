import os
import torch
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import matplotlib.pyplot as plt
import numpy as np
from model import GPT, GPTConfig
from loader import DataLoaderLite
from utils import LinearWarmupCosineAnnealingScheduler
import time

# DDP-based training loop

# setup the DDP environment
# `torchrun` command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE

ddp = int(os.environ.get('RANK', -1)) != -1 # is DDP run?
if ddp:
  assert torch.cuda.is_available(), "DDP requires CUDA"
  init_process_group(backend='nccl')
  ddp_rank = int(os.environ['RANK'])
  ddp_local_rank = int(os.environ['LOCAL_RANK'])
  ddp_world_size = int(os.environ['WORLD_SIZE'])
  device = f'cuda:{ddp_local_rank}' # cuda device
  torch.cuda.set_device(device)
  master_process = ddp_rank == 0 # this "master" process will do checkpointing, logging, etc.
else: 
  # non-DDP run
  ddp_rank = 0
  ddp_local_rank = 0
  ddp_world_size = 1
  master_process = True
  # auto-detect device
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

seedval = 1337
torch.manual_seed(seedval)
if torch.cuda.is_available():
  torch.cuda.manual_seed(seedval)

# batch size and gradient accumulator setup with DDP
total_batch_size = 524288 #2**19 = roughly the 0.5M token batch size listed in the GPT3 paper regarding the size of the 125M parameter GPT2
# each of the `WORLD_SIZE` processes will use the following B and T.
# thus, each forward pass through the model (during training) will process B * T * ddp_world_size tokens
B = 16 # "micro"-batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size should be divisble by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # = 1 for the above settings,
if master_process:
  print(f"total desired batch size: {total_batch_size}")
  print(f"required gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", master_process=master_process)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", master_process=master_process)


torch.set_float32_matmul_precision("high")

# create model
cfg = GPTConfig(vocab_size=50304) #overriding vocab_size with a power of 2
model = GPT(cfg)
model.to(device)
model.train()
#model = torch.compile(model) # most of the benefit here may be for GPUs
if ddp:
  model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

max_lr = 6e-4 * 2 # bumping up the max learning rate by a factor of 2
min_lr = 0.1 * max_lr
warmup_steps = 715
max_steps = 19073

# note, the fineweb dataset we are using contains 1e10 tokens. During each optimization step, we process 524288 tokens.
# 1e10 / 5.24288e5 = 19073 = max_steps. Hence, our training setup only achieves 1 pass through the dataset, i.e., 1 epoch.

optimizer = raw_model.configure_optimizers(weight_decay = 0.1, learning_rate=max_lr, device=device_type)
scheduler = LinearWarmupCosineAnnealingScheduler(max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps, max_steps=max_steps)

# logging
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # clearing file if it exists
  pass

for step in range(max_steps):
  t0 = time.time()
  is_last_step = (step == max_steps - 1)

  # validation
  if step % 250 == 0 or is_last_step:
    model.eval()
    
    val_loader.reset()
    with torch.no_grad():
      val_loss_accum = 0.0
      val_loss_steps = 20
      for _ in range(val_loss_steps):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
          logits, loss = model(x,y)
        loss = loss / val_loss_steps
        val_loss_accum += loss.detach()
    if ddp:
      dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
      print(f"validation loss: {val_loss_accum.item():.4f}")
      with open(log_file, "a") as f:
        f.write(f"{step} val {val_loss_accum.item():.6f}\n")

  # training 
  model.train()
  optimizer.zero_grad()
  loss_accum = 0.0
  for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    if ddp:
      model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
      logits, loss = model(x,y)
    loss = loss / grad_accum_steps # compensate for accumulation, the loss defaults to a mean reduction
    loss_accum += loss.detach()
    loss.backward()
  if ddp:
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  # lr
  lr = scheduler.get_lr(step)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  optimizer.step()
  if device_type == "cuda":
    torch.cuda.synchronize() 
  t1 = time.time()
  dt = t1-t0
  if master_process:
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    print(f'step {step} || loss: {loss_accum.item():.4f} || norm: {norm:.4f} || elapsed_time: dt={dt*1000:.4f}ms || tok/sec: {tokens_per_sec:.2f}')
    with open(log_file, "a") as f:
      f.write(f"{step} train {loss_accum.item():.6f}\n")
    if step > 0 and (step % 5000 == 0 or is_last_step):
      checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
      checkpoint = {
        'model': raw_model.state_dict(),
        'config': raw_model.config,
        'step': step,
        'val_loss': val_loss_accum.item()
      }
      # can also save optimizer.state_dict()
      torch.save(checkpoint, checkpoint_path)
if ddp:
  destroy_process_group()
