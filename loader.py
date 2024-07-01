import os
import torch
import tiktoken
import numpy as np

def load_tokens(filename):
  npt = np.load(filename)
  npt = npt.astype(int)
  ptt = torch.tensor(npt, dtype=torch.long)
  return ptt

# TODO include permutation / shuffling of data

class DataLoaderLite:
  def __init__(self, B, T, process_rank, num_processes, split, master_process):
    self.B = B
    self.T = T
    self.process_rank = process_rank
    self.num_processes = num_processes
    assert split in {'train', 'val'}

    # get the shard filenames
    data_root = "edu_fineweb10B"
    shards = os.listdir(data_root)
    shards = [s for s in shards if split in s]
    shards = sorted(shards)
    shards = [os.path.join(data_root, s) for s in shards]
    self.shards = shards
    assert len(shards) > 0, f"no shards found for the split {split}"
    if master_process: #only print once (on the master process)
      print(f"found {len(shards)} shards for the split {split}")

    # state and initialize at shard 0, we will process the shards sequentially
    # During each batch, B*T*num_processess tokens are processed.
    self.reset()

  def reset(self):
    self.current_shard = 0
    self.tokens = load_tokens(self.shards[self.current_shard])
    self.current_position = self.B * self.T * self.process_rank  # state (for iterating), strided across the processes

  def next_batch(self):
    B, T = self.B, self.T
    buff = self.tokens[self.current_position : self.current_position+B*T+1]
    x = (buff[:-1]).view(B, T) # inputs
    y = (buff[1:]).view(B, T) # targets
    self.current_position += B  * T * self.num_processes

    # reset the position if the remaining tokens do not exactly fit into the batch shape
    # I expect that this may result in not processing remaining tokens
    if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
      self.current_shard = (self.current_shard + 1) % len(self.shards)
      # load tokens from new shard
      self.tokens = load_tokens(self.shards[self.current_shard])
      self.current_position = B * T * self.process_rank
    return x, y
