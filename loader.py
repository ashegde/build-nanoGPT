import torch
import tiktoken

class DataLoaderLite:
  def __init__(self, B, T):
    self.B = B
    self.T = T

    with open('input.txt', 'r') as f:
      text = f.read()
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(text)
    self.tokens = torch.tensor(tokens)
    print(f'loaded {len(self.tokens)} tokens')
    print(f'1 epoch = {len(self.tokens) // (B*T)} batches')

    # state (for iterating)
    self.current_position = 0

  def next_batch(self):
    B, T = self.B, self.T
    buff = self.tokens[self.current_position : self.current_position+B*T+1]
    x = buff[:-1].view(B, T) # inputs
    y = buff[1:].view(B, T) # targets
    self.current_position += B*T

    # reset the position if we cannot construct the next batch of inputs and targets
    if self.current_position + (B*T+1) > len(self.tokens):
      self.current_position = 0
    return x, y