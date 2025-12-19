# Running train_gpt.py on a Single GPU

The script is designed for distributed training but can run on a single GPU using `torchrun`.

## Quick Start

```bash
torchrun --nproc_per_node=1 train_gpt.py
```

## How It Works

The script automatically adapts to single GPU:
- **world_size = 1**: Uses 1 GPU
- **grad_accum_steps = 8**: Since `grad_accum_steps = 8 // world_size`, you get 8 gradient accumulation steps
- **Effective batch size**: Original batch sizes are divided by `world_size * grad_accum_steps = 8`

## Default Batch Sizes (for 8 GPUs)

The default hyperparameters are:
- `train_bs_schedule`: `(131072, 262144, 393216)` tokens
- `val_batch_size`: `2097152` tokens

With **1 GPU**, these become:
- Training: `(16384, 32768, 49152)` tokens per step
- Validation: `262144` tokens per step

## Adjusting for Single GPU

If you want smaller batch sizes or different settings, you can modify the `Hyperparameters` class in `train_gpt.py`:

```python
@dataclass
class Hyperparameters:
    # data
    train_files: str = "data/fineweb10B/fineweb_train_*.bin"
    val_files: str = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens: int = 10485760
    
    # batch sizes - REDUCE THESE for single GPU
    train_bs_schedule: tuple = (4 * 2048 * 8, 8 * 2048 * 8, 12 * 2048 * 8)  # Smaller
    train_bs_extension: int = 12 * 2048 * 8
    train_max_seq_len: int = 128 * 16
    val_batch_size: int = 2 * 64 * 1024 * 8  # Smaller
    
    # ... rest of parameters
```

**Important**: Batch sizes must be divisible by `world_size * grad_accum_steps = 8`

## Required Environment Variables

`torchrun` automatically sets these, but if running manually:

```bash
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
```

## Data Path

Set the data path via environment variable:

```bash
DATA_PATH=/path/to/data torchrun --nproc_per_node=1 train_gpt.py
```

Or modify in code:
```python
data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)
args.val_files = os.path.join(data_path, args.val_files)
```

## Memory Considerations

Single GPU training will use:
- **Gradient accumulation**: 8 steps (simulates 8 GPUs)
- **Model**: ~350M parameters in bfloat16 ≈ 700MB
- **Activations**: Depends on batch size and sequence length
- **Optimizer states**: Additional memory for Adam/NorMuon

For a 24GB GPU, the default batch sizes should work, but you may need to reduce them if you run out of memory.

## Example: Reduced Batch Sizes for Single GPU

```python
@dataclass
class Hyperparameters:
    # ... other params ...
    
    # Smaller batch sizes for single GPU
    train_bs_schedule: tuple = (2 * 2048 * 8, 4 * 2048 * 8, 6 * 2048 * 8)  # 32768, 65536, 98304
    train_bs_extension: int = 6 * 2048 * 8
    val_batch_size: int = 1 * 64 * 1024 * 8  # 524288
```

## Troubleshooting

### "world_size must be a divisor of 8"
This assertion passes for world_size=1, 2, 4, or 8. If you see this error, check that `WORLD_SIZE` is set correctly.

### "Batch size must be divisible by world size"
Your batch sizes must be divisible by `world_size * grad_accum_steps = 8`. Make sure your batch sizes are multiples of 8.

### Out of Memory (OOM)
Reduce batch sizes in `Hyperparameters`:
- Reduce `train_bs_schedule` values
- Reduce `val_batch_size`
- Reduce `train_max_seq_len` (currently 2048)

### No data files found
Ensure your `.bin` files are in the correct location:
- Default: `data/fineweb10B/fineweb_train_*.bin` and `data/fineweb10B/fineweb_val_*.bin`
- Or set `DATA_PATH` environment variable

## Full Command Example

```bash
# With custom data path
DATA_PATH=/path/to/your/data \
torchrun --nproc_per_node=1 train_gpt.py

# Or from the nano/modded-nanogpt directory
cd nano/modded-nanogpt
torchrun --nproc_per_node=1 train_gpt.py
```

