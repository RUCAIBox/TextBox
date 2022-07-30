# Accelerate Example

## Configuration Example

```
In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU): 2
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
Do you want to use DeepSpeed? [yes/NO]: no
Do you want to use FullyShardedDataParallel? [yes/NO]: no
How many GPU(s) should be used for distributed training? [1]:4
Do you wish to use FP16 or BF16 (mixed precision)? [NO/fp16/bf16]: no
```

## Testing Example

```
Running:  accelerate-launch --config_file=None /.../accelerate/test_utils/test_script.py
stdout: **Initialization**
stdout: Testing, testing. 1, 2, 3.
stdout: Distributed environment: MULTI_GPU  Backend: nccl
stdout: Num processes: 4
stdout: Process index: 0
stdout: Local process index: 0
stdout: Device: cuda:0
stdout: 
stdout: 
stdout: **Test random number generator synchronization**
stdout: Distributed environment: MULTI_GPU  Backend: nccl
stdout: Num processes: 4
stdout: Process index: 2
stdout: Local process index: 2
stdout: Device: cuda:2
stdout: 
stdout: Distributed environment: MULTI_GPU  Backend: nccl
stdout: Num processes: 4
stdout: Process index: 3
stdout: Local process index: 3
stdout: Device: cuda:3
stdout: 
stdout: Distributed environment: MULTI_GPU  Backend: nccl
stdout: Num processes: 4
stdout: Process index: 1
stdout: Local process index: 1
stdout: Device: cuda:1
stdout: 
stdout: All rng are properly synched.
stdout: 
stdout: **DataLoader integration test**
stdout: 13  2 0 tensor([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
stdout:          14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
stdout:          28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
stdout:          42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
stdout:          56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
stdout:          70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
stdout:          84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
stdout:          98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
stdout:         112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
stdout:         126, 127], device='cuda:3') <class 'accelerate.data_loader.DataLoaderShard'>
stdout: tensor([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
stdout:          14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
stdout:          28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
stdout:          42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
stdout:          56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
stdout:          70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
stdout:          84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
stdout:          98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
stdout:         112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
stdout:         126, 127], device='cuda:1')tensor([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
stdout:          14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
stdout:          28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
stdout:          42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
stdout:          56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
stdout:          70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
stdout:          84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
stdout:          98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
stdout:         112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
stdout:         126, 127], device='cuda:2')  <class 'accelerate.data_loader.DataLoaderShard'><class 'accelerate.data_loader.DataLoaderShard'>
stdout: 
stdout: tensor([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
stdout:          14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
stdout:          28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
stdout:          42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
stdout:          56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
stdout:          70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
stdout:          84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
stdout:          98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
stdout:         112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
stdout:         126, 127], device='cuda:0') <class 'accelerate.data_loader.DataLoaderShard'>
stdout: Non-shuffled dataloader passing.
stdout: Shuffled dataloader passing.
stdout: Non-shuffled central dataloader passing.
stdout: Shuffled central dataloader passing.
stdout: 
stdout: **Training integration test**
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Training yielded the same results on one CPU or distributed setup with no batch split.
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: 
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: 
stdout: FP16 training check.Training yielded the same results on one CPU or distributes setup with batch split.FP16 training check.
stdout: 
stdout: FP16 training check.
stdout: 
stdout: FP16 training check.
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: 
stdout: Legacy FP16 training check.
stdout: Legacy FP16 training check.
stdout: Legacy FP16 training check.
stdout: Legacy FP16 training check.
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: BF16 training check.BF16 training check.BF16 training check.
stdout: 
stdout: 
stdout: BF16 training check.
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
stdout: Model dtype: torch.float32, torch.float32. Input dtype: torch.float32
Test is a success! You are ready for your distributed training!
```

