import math
import torch
import torch.nn.functional as F
import tiktoken

class LinearWarmupCosineAnnealingScheduler:
  def __init__(self, max_lr=3e-4, min_lr=0.0, warmup_steps=10, max_steps=50):
    self.max_lr = max_lr
    self.min_lr = min_lr
    self.warmup_steps = warmup_steps
    self.max_steps = max_steps

  def get_lr(self, it):
    # 1.) linear warmup from 0
    if it < self.warmup_steps:
      return (it+1)*self.max_lr / self.warmup_steps

    # 2.) it > lr_decay
    if it > self.max_steps:
      return self.min_lr
    
    # 3.) cosine decay
    decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return self.min_lr + coeff*(self.max_lr - self.min_lr)
  

def say_hello(model):
    model.eval()

    num_return_seqs = 5
    max_length = 30

    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm a language model,") # (B,)
    tokens = torch.tensor(tokens, dtype=torch.long) # (B,)
    tokens = tokens[None,:].repeat(num_return_seqs, 1) # (5, 8)
    x = tokens.to(device)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x) # (B,T,vocab_size)
        logits = logits[:, -1, :] #predictive distribution for the final token
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs,1) # (B,1)
        xcol = torch.gather(topk_indices, -1, ix) # (B,1)
        x = torch.cat((x,xcol), dim=1)
    # decoding the generated text
    for i in range(num_return_seqs):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)