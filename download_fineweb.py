import os 
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import time 


local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
# 1**10 token dataset. Break into 10**2 shards, each consisting of 10**8 tokens
shard_size = int(1e8)

# local directory
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# logging and start timer
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log_fineweb10B.txt")
with open(log_file, "w") as f: # clearing file if it exists
  pass
t0 = time.time()

# download dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # special token for end of text

# doc tokenizer helper -- tokenizes a single document and returns an numpy array of uint16 tokens
def tokenize(doc):
    tokens = [eot] #init
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all()
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents and write output shards
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # can we add new tokens to the current shard?
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)

            # create and/or update progress
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write what we can to the current shard and then start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            #
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1 # increment to the next shard
            progress_bar = None
            # move over the remaining tokens to the new shard
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder
    
    # write the remaining tokens to the final shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count]) 

t1 = time.time()
dt = t1-t0
with open(log_file, "a") as f:
    f.write(f"elapsed time {dt:.6f}\n")
