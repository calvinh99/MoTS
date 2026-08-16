"""nanoBPE. Each version is a copy of the last with a small edit.

  v0→v1  GPT-2 pretok dict (stop scanning every byte of the corpus)
  v1→v2  inverted index + incremental pair counts
  v2→v3  lazy heap: heapify, pop, one heappush
  v3→v4  Pool pretok (merge loop unchanged)

  uv run python cs336_basics/bpe_reimplement.py 0 val
"""
import argparse
import heapq
import math
import multiprocessing as mp
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import BinaryIO

import regex as re

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = ["<|endoftext|>"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def merge_pair(tokens: list[bytes] | tuple[bytes, ...], pair: tuple[bytes, bytes]) -> list[bytes]:
    """Replace every adjacent `pair` in `tokens` with the concatenated bytes."""
    new_token = pair[0] + pair[1]
    merged = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            merged.append(new_token)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def pretokenize_chunk(input_path: str, start: int, end: int, special_tokens: list[str]) -> Counter:
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    special_pat = "|".join(re.escape(tok) for tok in special_tokens)
    pat = re.compile(GPT2_PAT)
    pretoken_freq = Counter()
    for part in re.split(special_pat, text):
        for match in pat.finditer(part):
            pretoken_freq[tuple(bytes([b]) for b in match.group().encode())] += 1
    return pretoken_freq


class InversePair:
    """Make a min-heap pop the lexicographically greater pair on frequency ties."""

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "InversePair") -> bool:
        return self.pair > other.pair


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def train_bpe_v0(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    load_start = time.perf_counter()
    text = open(input_path, encoding="utf-8", errors="ignore").read()
    # split on specials so we don't merge across documents — not GPT-2 pretok
    special_pat = "|".join(re.escape(tok) for tok in special_tokens)
    corpus = []
    for part in re.split(special_pat, text):
        if part:
            corpus.append([bytes([b]) for b in part.encode()])
    load_seconds = time.perf_counter() - load_start

    vocab = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode()
    merges = []
    merge_start = time.perf_counter()
    while len(vocab) < vocab_size:
        pair_freq = defaultdict(int)
        for tokens in corpus:
            for i in range(len(tokens) - 1):
                pair_freq[tokens[i], tokens[i + 1]] += 1
        max_pair = max(pair_freq, key=lambda pair: (pair_freq[pair], pair))
        corpus = [merge_pair(tokens, max_pair) for tokens in corpus]
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
    print(f"load corpus: {load_seconds:.3f}s  merge: {time.perf_counter() - merge_start:.3f}s")
    return vocab, merges


def train_bpe_v1(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretok_start = time.perf_counter()
    pretoken_freq = pretokenize_chunk(input_path, 0, os.path.getsize(input_path), special_tokens)
    pretok_seconds = time.perf_counter() - pretok_start

    vocab = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode()
    merges = []
    merge_start = time.perf_counter()
    while len(vocab) < vocab_size:
        pair_freq = defaultdict(int)
        for pretoken, count in pretoken_freq.items():
            for i in range(len(pretoken) - 1):
                pair_freq[pretoken[i], pretoken[i + 1]] += count
        max_pair = max(pair_freq, key=lambda pair: (pair_freq[pair], pair))
        new_freq = Counter()
        for pretoken, count in pretoken_freq.items():
            new_freq[tuple(merge_pair(pretoken, max_pair))] += count
        pretoken_freq = new_freq
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
    print(f"pretokenization: {pretok_seconds:.3f}s  merge: {time.perf_counter() - merge_start:.3f}s")
    return vocab, merges


def train_bpe_v2(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretok_start = time.perf_counter()
    pretoken_freq = pretokenize_chunk(input_path, 0, os.path.getsize(input_path), special_tokens)
    pretok_seconds = time.perf_counter() - pretok_start

    vocab = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode()
    merges = []
    lookup = defaultdict(set)
    pair_freq = defaultdict(int)
    for pretoken, count in pretoken_freq.items():
        for i in range(len(pretoken) - 1):
            lookup[pretoken[i], pretoken[i + 1]].add(pretoken)
            pair_freq[pretoken[i], pretoken[i + 1]] += count

    merge_start = time.perf_counter()
    while len(vocab) < vocab_size:
        max_pair = max(pair_freq, key=lambda pair: (pair_freq[pair], pair))
        pair_freq_delta = defaultdict(int)
        for pretoken in list(lookup[max_pair]):
            count = pretoken_freq[pretoken]
            new_pretoken = tuple(merge_pair(pretoken, max_pair))
            for i in range(len(pretoken) - 1):
                pair_freq_delta[pretoken[i], pretoken[i + 1]] -= count
                lookup[pretoken[i], pretoken[i + 1]].discard(pretoken)
            for i in range(len(new_pretoken) - 1):
                pair_freq_delta[new_pretoken[i], new_pretoken[i + 1]] += count
                lookup[new_pretoken[i], new_pretoken[i + 1]].add(new_pretoken)
            del pretoken_freq[pretoken]
            pretoken_freq[new_pretoken] += count
        for pair, delta in pair_freq_delta.items():
            if delta == 0:
                continue
            pair_freq[pair] += delta
            if pair_freq[pair] == 0:
                del pair_freq[pair]
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
    print(f"pretokenization: {pretok_seconds:.3f}s  merge: {time.perf_counter() - merge_start:.3f}s")
    return vocab, merges


def train_bpe_v3(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretok_start = time.perf_counter()
    pretoken_freq = pretokenize_chunk(input_path, 0, os.path.getsize(input_path), special_tokens)
    pretok_seconds = time.perf_counter() - pretok_start

    vocab = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode()
    merges = []
    lookup = defaultdict(set)
    pair_freq = defaultdict(int)
    for pretoken, count in pretoken_freq.items():
        for i in range(len(pretoken) - 1):
            lookup[pretoken[i], pretoken[i + 1]].add(pretoken)
            pair_freq[pretoken[i], pretoken[i + 1]] += count
    heap = [(-count, InversePair(pair)) for pair, count in pair_freq.items()]
    heapq.heapify(heap)

    merge_start = time.perf_counter()
    while len(vocab) < vocab_size:
        while True:
            neg_freq, inv_pair = heapq.heappop(heap)
            if pair_freq.get(inv_pair.pair, 0) == -neg_freq:
                max_pair = inv_pair.pair
                break
        pair_freq_delta = defaultdict(int)
        for pretoken in list(lookup[max_pair]):
            count = pretoken_freq[pretoken]
            new_pretoken = tuple(merge_pair(pretoken, max_pair))
            for i in range(len(pretoken) - 1):
                pair_freq_delta[pretoken[i], pretoken[i + 1]] -= count
                lookup[pretoken[i], pretoken[i + 1]].discard(pretoken)
            for i in range(len(new_pretoken) - 1):
                pair_freq_delta[new_pretoken[i], new_pretoken[i + 1]] += count
                lookup[new_pretoken[i], new_pretoken[i + 1]].add(new_pretoken)
            del pretoken_freq[pretoken]
            pretoken_freq[new_pretoken] += count
        for pair, delta in pair_freq_delta.items():
            if delta == 0:
                continue
            pair_freq[pair] += delta
            if pair_freq[pair] == 0:
                del pair_freq[pair]
            else:
                heapq.heappush(heap, (-pair_freq[pair], InversePair(pair)))
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
    print(f"pretokenization: {pretok_seconds:.3f}s  merge: {time.perf_counter() - merge_start:.3f}s")
    return vocab, merges


def train_bpe_v4(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretok_start = time.perf_counter()
    num_cores = os.cpu_count() or 1
    max_chunk_bytes = 1 * 1024 * 1024  # 1MB
    desired_num_chunks = max(num_cores, math.ceil(os.path.getsize(input_path) / max_chunk_bytes))
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, special_tokens[0].encode("utf-8"))
    jobs = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    num_processes = min(num_cores, len(jobs))
    with mp.Pool(num_processes) as pool:
        pretoken_freq = Counter()
        for chunk_freq in pool.starmap(pretokenize_chunk, jobs):
            pretoken_freq.update(chunk_freq)
    pretok_seconds = time.perf_counter() - pretok_start

    vocab = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode()
    merges = []
    lookup = defaultdict(set)
    pair_freq = defaultdict(int)
    for pretoken, count in pretoken_freq.items():
        for i in range(len(pretoken) - 1):
            lookup[pretoken[i], pretoken[i + 1]].add(pretoken)
            pair_freq[pretoken[i], pretoken[i + 1]] += count
    heap = [(-count, InversePair(pair)) for pair, count in pair_freq.items()]
    heapq.heapify(heap)

    merge_start = time.perf_counter()
    while len(vocab) < vocab_size:
        while True:
            neg_freq, inv_pair = heapq.heappop(heap)
            if pair_freq.get(inv_pair.pair, 0) == -neg_freq:
                max_pair = inv_pair.pair
                break
        pair_freq_delta = defaultdict(int)
        for pretoken in list(lookup[max_pair]):
            count = pretoken_freq[pretoken]
            new_pretoken = tuple(merge_pair(pretoken, max_pair))
            for i in range(len(pretoken) - 1):
                pair_freq_delta[pretoken[i], pretoken[i + 1]] -= count
                lookup[pretoken[i], pretoken[i + 1]].discard(pretoken)
            for i in range(len(new_pretoken) - 1):
                pair_freq_delta[new_pretoken[i], new_pretoken[i + 1]] += count
                lookup[new_pretoken[i], new_pretoken[i + 1]].add(new_pretoken)
            del pretoken_freq[pretoken]
            pretoken_freq[new_pretoken] += count
        for pair, delta in pair_freq_delta.items():
            if delta == 0:
                continue
            pair_freq[pair] += delta
            if pair_freq[pair] == 0:
                del pair_freq[pair]
            else:
                heapq.heappush(heap, (-pair_freq[pair], InversePair(pair)))
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
    print(
        f"pretokenization: {pretok_seconds:.3f}s ({num_processes} processes)  "
        f"merge: {time.perf_counter() - merge_start:.3f}s"
    )
    return vocab, merges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version", type=int, choices=[0, 1, 2, 3, 4])
    parser.add_argument("split", choices=["val", "valid", "train"])
    args = parser.parse_args()
    split = "valid" if args.split in ("val", "valid") else "train"
    path = DATA_DIR / f"TinyStoriesV2-GPT4-{split}.txt"
    vocab_size = {0: 256 + 1 + 2, 1: 256 + 1 + 100}.get(args.version, 10_000)
    trainer = [train_bpe_v0, train_bpe_v1, train_bpe_v2, train_bpe_v3, train_bpe_v4][args.version]
    print(f"v{args.version}  {path.name}  vocab_size={vocab_size}")
    vocab, merges = trainer(str(path), vocab_size, SPECIAL_TOKENS)
    print(f"vocab={len(vocab)}  merges={len(merges)}")
