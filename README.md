# cute-flash-attention

Implement a simple Flash Attention using Cute, with very little performance overhead. When using it, you need to first replace `-I/mnt/d/cuda/cutlass/*` in bench.py with the path to your own cutlass directory. This code is for educational purposes only, edge cases are not considered, and it only works in the scenario where `head_dim=64`.

|                      | 1x1024x32x64 |
| -------------------- | ------------ |
| cute-flash-attention | 182.21us     |
| flashinfer           | 194.4us      |
| flash-attention      | 184.48us     |
