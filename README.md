This repo is entirely based on following along with Andrej Karpathy's tutorial on GPT2: https://www.youtube.com/watch?v=l8pRSuU81PU

The training and validation losses for the settings documented in the above scripts are:
![loss](https://github.com/ashegde/build-nanoGPT/assets/18709839/bdd447fe-ae76-4b7a-a0d8-8aea6537f8d8)

Training was conducted on a GPU instance from [Lambda Labs]([url](https://lambdalabs.com/)) containing 8x NVIDIA A100 SXM with 40 GB VRAM/GPU memory.
The cost for this instance was $10.32 / hr. Downloading the dataset via `download_fineweb.py` took around 0.5 hrs. Training the model took around 1-2 hrs. So, the total cost was around $30.
