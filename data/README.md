# FineWeb Data Processing for modded-nanogpt

This directory contains scripts for processing FineWeb data into the `.bin` format required by modded-nanogpt.

## Overview

The modded-nanogpt training code expects data in a specific binary format (`.bin` files) with:
- **Header**: 256 int32 values (1024 bytes)
  - `header[0] = 20240520` (magic number)
  - `header[1] = 1` (version)
  - `header[2] = number of tokens`
- **Tokens**: uint16 array (2 bytes per token, GPT-2 token IDs)

Each document is separated by the `<|endoftext|>` token (ID: 50256).

## Files

### `cached_fineweb10B.py`
Downloads pre-processed `.bin` files from HuggingFace. This is the **slow** method if you already have parquet files.

### `fineweb.py`
Original script that downloads FineWeb from HuggingFace and converts to `.bin` format. Uses the `datasets` library to stream data.

### `parquet_to_bin.py` ⭐ **NEW**
**Use this if you already have parquet files locally!**

Converts local parquet files directly to `.bin` format, avoiding slow downloads.

## Using `parquet_to_bin.py`

### Basic Usage

```bash
python parquet_to_bin.py \
    --parquet_dir /path/to/your/parquet/files \
    --output_dir fineweb10B
```

### Options

- `--parquet_dir`: Directory or glob pattern containing parquet files (required)
- `--output_dir`: Output directory for `.bin` files (default: `fineweb10B`)
- `--shard_size`: Tokens per shard (default: 100M tokens)
- `--text_field`: Field name in parquet containing text (default: `"text"`)
- `--num_workers`: Number of parallel workers (default: `cpu_count - 2`)
- `--parquet_pattern`: Glob pattern for parquet files (default: `*.parquet`)

### Examples

```bash
# Process all parquet files in a directory
python parquet_to_bin.py --parquet_dir /data/fineweb/parquet/

# Process specific files with a pattern
python parquet_to_bin.py --parquet_dir "/data/fineweb/*.parquet"

# Custom output directory and shard size
python parquet_to_bin.py \
    --parquet_dir /data/fineweb/ \
    --output_dir my_fineweb \
    --shard_size 50000000  # 50M tokens per shard

# Single file
python parquet_to_bin.py --parquet_dir /data/fineweb/file.parquet
```

### Output

The script creates files in the output directory:
- `fineweb_val_000000.bin` - First shard (validation set)
- `fineweb_train_000001.bin` - Second shard (training)
- `fineweb_train_000002.bin` - Third shard (training)
- ... and so on

Each shard contains approximately `shard_size` tokens (default: 100M tokens).

## How It Works

1. **Read Parquet Files**: Uses `pyarrow.parquet` to stream parquet files efficiently
2. **Tokenize**: Uses `tiktoken` with GPT-2 encoding to tokenize each document
3. **Add Separators**: Prepends `<|endoftext|>` token (50256) before each document
4. **Shard**: Splits tokens into shards of specified size
5. **Write Binary**: Writes header + tokens in the expected format

## Dependencies

Install required packages:

```bash
pip install tiktoken pyarrow pandas numpy tqdm
```

Or use the requirements file:

```bash
pip install -r requirements.txt
# Note: You may also need: pip install pyarrow pandas numpy tqdm
```

## Data Format Details

### Bin File Structure

```
[Header: 256 int32 = 1024 bytes]
  - bytes 0-3:   20240520 (magic number)
  - bytes 4-7:   1 (version)
  - bytes 8-11:  num_tokens (int32)
  - bytes 12-1023: reserved (zeros)

[Tokens: num_tokens * 2 bytes]
  - Each token is a uint16 (0-65535)
  - GPT-2 token IDs
  - Documents separated by token 50256 (<|endoftext|>)
```

### Parquet File Requirements

The parquet files should have a `text` field (or specify with `--text_field`) containing the document text. The script will:
- Skip empty or null text fields
- Tokenize each document separately
- Add the `<|endoftext|>` separator automatically

## Performance Tips

1. **Parallel Processing**: The script uses multiprocessing by default. Adjust `--num_workers` based on your system.
2. **Memory**: Large shard sizes use more memory. Default (100M tokens) uses ~200MB per shard buffer.
3. **I/O**: Processing is streaming, so it works with large parquet files without loading everything into memory.

## Troubleshooting

### "token dictionary too large for uint16"
This means a token ID exceeds 65535. GPT-2 tokenizer should never produce this, but if you're using a custom tokenizer, ensure vocab size ≤ 65536.

### "No parquet files found"
Check that:
- The path is correct
- Files have `.parquet` extension (or adjust `--parquet_pattern`)
- You have read permissions

### Field not found
If your parquet files use a different field name for text, specify it with `--text_field`.

## Comparison with Other Methods

| Method | Speed | Requires Download | Use Case |
|--------|-------|-------------------|----------|
| `cached_fineweb10B.py` | Slow (download) | Yes | Quick start, no parquet files |
| `fineweb.py` | Medium (download + process) | Yes | Want to process from HuggingFace |
| `parquet_to_bin.py` | Fast (local only) | No | **You already have parquet files** |

