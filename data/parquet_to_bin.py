"""
Convert FineWeb parquet files to .bin format for modded-nanogpt.

This script reads parquet files from a local directory and converts them to
the .bin format expected by modded-nanogpt, avoiding the need to download
pre-processed bin files from HuggingFace.

Usage:
    python parquet_to_bin.py --parquet_dir /path/to/parquet/files --output_dir data/fineweb10B

The bin file format:
    - Header: 256 int32 values (1024 bytes)
      - header[0] = 20240520 (magic number)
      - header[1] = 1 (version)
      - header[2] = number of tokens
    - Tokens: uint16 array (2 bytes per token)
"""
import os
import argparse
import glob
import multiprocessing as mp
import numpy as np
import pandas as pd
import tiktoken
import pyarrow.parquet as pq
from tqdm import tqdm


def write_datafile(filename, toks):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


def tokenize_document(text, enc, eot):
    """
    Tokenize a single document and return a numpy array of uint16 tokens.
    
    Args:
        text: The text content of the document
        enc: The tiktoken encoder
        eot: The end-of-text token ID
    
    Returns:
        numpy array of uint16 tokens
    """
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def process_parquet_file(parquet_file, enc, eot, text_field="text"):
    """
    Process a single parquet file and return all tokenized documents.
    
    Args:
        parquet_file: Path to the parquet file
        enc: The tiktoken encoder
        eot: The end-of-text token ID
        text_field: Name of the text field in the parquet file
    
    Returns:
        List of numpy arrays of tokens (one per document)
    """
    tokens_list = []
    try:
        parquet_file_obj = pq.ParquetFile(parquet_file)
        
        # Check if field exists
        schema = parquet_file_obj.schema_arrow
        field_names = [field.name for field in schema]
        
        if text_field not in field_names:
            available_fields = ", ".join(field_names)
            raise ValueError(f"Field '{text_field}' not found in {parquet_file}. Available fields: {available_fields}")
        
        # Process in batches to avoid loading entire file into memory
        for batch in parquet_file_obj.iter_batches(batch_size=1000):
            df_batch = batch.to_pandas()
            
            # Process each document in the batch
            for text in df_batch[text_field]:
                if pd.isna(text) or text is None:
                    continue
                text_str = str(text)
                if len(text_str.strip()) == 0:
                    continue
                
                # Tokenize the document
                tokens = tokenize_document(text_str, enc, eot)
                tokens_list.append(tokens)
    
    except Exception as e:
        print(f"Error processing {parquet_file}: {e}")
        raise
    
    return tokens_list


def tokenize_worker(args):
    """Worker function for parallel processing."""
    parquet_file, enc_name, text_field = args
    enc = tiktoken.get_encoding(enc_name)
    eot = enc._special_tokens['<|endoftext|>']
    return process_parquet_file(parquet_file, enc, eot, text_field)


def main():
    parser = argparse.ArgumentParser(
        description="Convert FineWeb parquet files to .bin format for modded-nanogpt"
    )
    parser.add_argument(
        "--parquet_dir",
        type=str,
        required=True,
        help="Directory containing parquet files (supports glob patterns like *.parquet)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="fineweb10B",
        help="Output directory for .bin files (default: fineweb10B)"
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=10**8,
        help="Size of each shard in tokens (default: 100M tokens)"
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Name of the text field in parquet files (default: 'text')"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes (default: cpu_count - 2)"
    )
    parser.add_argument(
        "--parquet_pattern",
        type=str,
        default="*.parquet",
        help="Glob pattern to match parquet files (default: *.parquet)"
    )
    
    args = parser.parse_args()
    
    # Find all parquet files
    if os.path.isfile(args.parquet_dir):
        # Single file
        parquet_files = [args.parquet_dir]
    elif os.path.isdir(args.parquet_dir):
        # Directory - find all parquet files
        pattern = os.path.join(args.parquet_dir, args.parquet_pattern)
        parquet_files = sorted(glob.glob(pattern))
    else:
        # Try as glob pattern
        parquet_files = sorted(glob.glob(args.parquet_dir))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found matching: {args.parquet_dir}")
    
    print(f"Found {len(parquet_files)} parquet file(s)")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']
    
    # Process parquet files (with optional parallelization)
    nprocs = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 2)
    
    all_tokens_list = []
    
    if nprocs > 1 and len(parquet_files) > 1:
        print(f"Processing {len(parquet_files)} parquet files with {nprocs} workers...")
        # Parallel processing
        worker_args = [(f, "gpt2", args.text_field) for f in parquet_files]
        with mp.Pool(nprocs) as pool:
            results = pool.map(tokenize_worker, worker_args)
        # Flatten results
        for tokens_list in results:
            all_tokens_list.extend(tokens_list)
    else:
        print(f"Processing {len(parquet_files)} parquet files sequentially...")
        for parquet_file in tqdm(parquet_files, desc="Processing parquet files"):
            tokens_list = process_parquet_file(parquet_file, enc, eot, args.text_field)
            all_tokens_list.extend(tokens_list)
    
    print(f"Total documents processed: {len(all_tokens_list)}")
    
    # Write shards
    shard_index = 0
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    
    print("Writing shards...")
    for tokens in tqdm(all_tokens_list, desc="Tokenizing and sharding"):
        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < args.shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(output_dir, f"fineweb_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = args.shard_size - token_count
            if progress_bar:
                progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder
    
    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(output_dir, f"fineweb_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])
    
    print(f"\nDone! Created {shard_index + 1} shard(s) in {output_dir}")
    print(f"First shard is validation (fineweb_val_000000.bin), rest are training shards")


if __name__ == "__main__":
    main()

