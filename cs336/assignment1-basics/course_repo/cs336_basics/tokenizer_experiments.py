# sample 10 docs from tinystories
# use tinystories tokenizer and tokenize, compute bytes to tokens ratio
import os
import time
import numpy as np
from cs336_basics.tokenizer import Tokenizer

def human_bytes(n: float) -> str:
      for unit in ("B", "KB", "MB", "GB", "TB"):
          if n < 1024:
              return f"{n:.2f} {unit}"
          n /= 1024
      return f"{n:.2f} PB"

def iter_docs(f, sep="<|endoftext|>", chunk_size=4096):
    buffer = ""
    while True:
        chunk = f.read(chunk_size)

        if not chunk:  # end
            if buffer: yield buffer
            break
        
        buffer += chunk
        parts = buffer.split(sep)
        for doc in parts[:-1]:  # last one is not guaranteed to be a doc
            yield doc
        buffer = parts[-1]

def iter_chunks(f, sep=b"<|endoftext|>", chunk_size: int = 1 * 1024 * 1024):
    """Stream (text, num_bytes) from a BINARY file, never splitting `sep`.
    Reading bytes gives exact progress for free (len of the slice); decoding
    is work we'd do anyway to feed the tokenizer, so it adds nothing."""
    buffer = b""
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            if buffer: yield buffer.decode("utf-8"), len(buffer)
            break
        buffer += chunk
        cut = buffer.rfind(sep)  # find last occurance of sep in buffer
        if cut == -1:
            continue
        cut += len(sep)
        yield buffer[:cut].decode("utf-8"), cut  # sep is ASCII, so the cut never splits a UTF-8 char
        buffer = buffer[cut:]

def tokenize_tinystories_test():
    tok = Tokenizer.from_files(
        "bpe_output/tinystories/vocab.json",
        "bpe_output/tinystories/merges.txt"
    )

    num_bytes = 0
    num_toks = 0
    i = 0
    with open("data/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8") as f:
        for doc in iter_docs(f):
            if i >= 10: break
            num_bytes += len(doc.encode("utf-8"))
            num_toks += len(tok.encode(doc))
            i += 1

    print(f"tinystories tokenizer compression ratio: {num_bytes / num_toks:.2f} bytes/tok")

def profile_tokenizer():
    tok = Tokenizer.from_files(
        "bpe_output/tinystories/vocab.json",
        "bpe_output/tinystories/merges.txt",
        special_tokens=["<|endoftext|>"]
    )

    with open("data/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8") as f:
        text = f.read()
    num_bytes = len(text.encode("utf-8"))
    # start = time.perf_counter()
    # tok.encode(text)
    # elapsed = time.perf_counter() - start
    # Tokenizing entire text we achieve 5.58 - 5.73 MB/sec

    with open("data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
        start = time.perf_counter()
        for chunk, _ in iter_chunks(f):  # using 128 MB chunks we achieve 5.51 MB/sec (basically same)
            tok.encode(chunk)
        elapsed = time.perf_counter() - start

    # print as MB per sec
    bps = num_bytes / elapsed
    mbps = bps / 1024 / 1024
    print(f"tinystories tokenizer throughput: {mbps:.2f} MB/sec")
    print(f"time to tokenize Pile dataset (825GB): {825 * 1024 / mbps / 3600:.2f} hrs")

def tokenize_tinystories(data_path: str, out_path: str | None, chunk_size: int = 1 * 1024 * 1024):
    # acutally write the fully tokenized train dataset as a np uint16 array
    tok = Tokenizer.from_files(
        "bpe_output/tinystories/vocab.json",
        "bpe_output/tinystories/merges.txt",
        special_tokens=["<|endoftext|>"]
    )

    total = os.path.getsize(data_path)
    done = 0
    num_toks = 0
    start = time.perf_counter()
    with open(data_path, "rb") as f, open(out_path or os.devnull, "wb") as fout:
        for text, nbytes in iter_chunks(f, chunk_size=chunk_size):
            ids = np.array(tok.encode(text), dtype=np.uint16)  # uint16 is smallest size that can go up to vocab size
            ids.tofile(fout)
            done += nbytes
            num_toks += ids.size
            elapsed = time.perf_counter() - start
            print(
                f"\r[tinystories] {done / total:6.1%} | "
                f"{human_bytes(done)}/{human_bytes(total)} | "
                f"{num_toks:,} toks | {human_bytes(done / elapsed)}/s | {elapsed:5.1f}s",
                end="", flush=True,
            )
    print()  # finish the live line

if __name__ == "__main__":
    tokenize_tinystories_test()
    profile_tokenizer()

    # surprisingly using 1MB chunks has 7.30 MB/s throughput vs 128MB chunks which is only 5.21 Mb/s
    tokenize_tinystories(
        data_path="data/TinyStoriesV2-GPT4-train.txt",
        out_path=None
    )
    # tokenize_tinystories(
    #     data_path="data/TinyStoriesV2-GPT4-train.txt",
    #     out_path="data/TinyStoriesV2-GPT4-train-tokenized.bin"
    # )

