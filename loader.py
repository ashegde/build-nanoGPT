import torch
import tiktoken

class DataLoaderLite:
  def __init__(self, B, T, process_rank, num_processes):
    self.B = B
    self.T = T
    self.process_rank = process_rank
    self.num_processes = num_processes

    with open('input.txt', 'r') as f:
      text = f.read()
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(text)
    self.tokens = torch.tensor(tokens)
    print(f'loaded {len(self.tokens)} tokens')
    print(f'1 epoch = {len(self.tokens) // (B*T)} batches')

    # state (for iterating), strided across the processes
    self.current_position = self.B * self.T * self.process_rank

  def next_batch(self):
    B, T = self.B, self.T
    buff = self.tokens[self.current_position : self.current_position+B*T+1]
    x = buff[:-1].view(B, T) # inputs
    y = buff[1:].view(B, T) # targets
    self.current_position += B  * T * self.num_processes

    # reset the position if we cannot construct the next batch of inputs and targets
    if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
      self.current_position = B * T * self.process_rank
    return x, y