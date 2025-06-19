# Installation

See all package versions in requirements.txt

```bash
conda create -n timer1 python=3.10.12
pip install requirements.txt
conda activate timer1
```

Note that we use `vllm==0.8.4 transformers==4.51.1 numba==0.61.2  trl==0.17.0 torch==2.6.0`, with CUDA version of `12.4`. 
The versions are important for bug-free training and inference!