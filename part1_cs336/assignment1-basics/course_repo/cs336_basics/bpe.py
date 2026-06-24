import argparse
import cProfile
import time
import resource
import pstats
import threading
import regex as re  # diff from import re?
import math
import multiprocessing as mp
import json
import os
from collections import Counter, defaultdict
from typing import BinaryIO
import logging
import heapq
import psutil

logger = logging.getLogger(__name__)

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    taken from pretokenization_example.py
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
    

def pretokenize(input_path: str, start, end, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # split based on special tokens
    special_token_PAT = "|".join(re.escape(tok) for tok in special_tokens) # we escape to prevent a special token's "|" from being interpreted as a regex pattern
    chunk_parts = re.split(special_token_PAT, chunk)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    PAT = re.compile(PAT)  # precompilng helps
    token_freq = Counter()

    # using finditer is more memory friendly than findall
    for chunk_part in chunk_parts:
        for pt in PAT.finditer(chunk_part):
            token_freq[pt.group()] += 1

    # convert each pretoken to list of bytes (utf-8)
    token_freq = {tuple([bytes([b]) for b in pt.encode("utf-8")]): freq for pt, freq in token_freq.items()}
    return token_freq

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    profile_mode: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """outputs vocab and ordered merges"""

    with open(input_path, "rb") as f:
        # say we have 8 cores and using a 200MB limit we get 12 chunks
        # but we can perfectly have each chunk be 250MB and get 8 chunks, this is faster cuz 1 process = 1 chunk
        # so we should have a max chunk limit (to avoid using too much memory) rather than a hardcoded chunk limit
        # this we set to higher, then if the num chunks split by this size is smaller than number of cores, we use number of cores
        # with 512MB, and 8 cores on my mac, at any given moment we are only using 4GB of RAM which is super doable
        MAX_CHUNK_BYTES = 512 * 1024 * 1024  # 512MB limit
        max_cores = 1 if profile_mode else os.cpu_count()
        desired_num_chunks = max(max_cores, math.ceil(os.path.getsize(input_path) / MAX_CHUNK_BYTES))
        logger.info("Target number of chunks: %d", desired_num_chunks)
        boundaries = find_chunk_boundaries(f, desired_num_chunks, special_tokens[0].encode("utf-8"))
        logger.info("Found %d chunks, avg chunk size: %.2fMB", len(boundaries) - 1, os.path.getsize(input_path) / (len(boundaries) - 1) / 1024 / 1024)

    # map
    jobs = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    if profile_mode:
        logger.info("Running in profile mode, using 1 process for pretokenization")
        pretoken_freqs = [pretokenize(*job) for job in jobs]
    else:
        num_processes = min(os.cpu_count(), len(boundaries)-1)
        logger.info("Using %d processes for pretokenization", num_processes)
        with mp.Pool(num_processes) as pool:
            pretoken_freqs = pool.starmap(pretokenize, jobs) # starmap so that tuples are unpacked

    # reduce
    pretoken_freq: dict[list[bytes], int] = Counter()
    for d in pretoken_freqs:
        pretoken_freq.update(d)

    # create initial vocab of 256 bytes & special tokens
    vocab = {i: bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)] = bytes(special_token.encode("utf-8"))
    merges = []

    # now merge loop w/ index
    lookup_pretoken: dict[tuple[bytes, bytes], set[tuple[bytes]]] = defaultdict(set)
    pair_freq: dict[tuple[bytes, bytes], int] = defaultdict(int)

    class InversePair:
        def __init__(self, pair: tuple[bytes, bytes]): self.pair = pair
        def __lt__(self, other): return self.pair > other.pair  # we turn lt into gt for minheap tie breaking
        
    pair_max_heap: list[tuple[int, InversePair]] = []  # for O(1) pop & O(logn) push of pair freqs rather than O(nlogn) sort through pair freqs

    # one-time construction of initial pair frequencies and lookup
    for pretoken, freq in pretoken_freq.items():
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            lookup_pretoken[pair].add(pretoken)  # we use set so that a pretoken with the same pair multiple times doesn't get duplicated
            pair_freq[pair] += freq  # pair may occur in other pretokens too
    
    # push each pair onto the heap
    for pair, freq in pair_freq.items():
        heapq.heappush(pair_max_heap, (-freq, InversePair(pair)))  # negative freq for maxheap

    # keep merging the most frequent pair
    while len(vocab) < vocab_size:
        # use max heap for max_pair to have faster perf, matters cuz pairs grow w/ vocab/corpus size
        max_pair: tuple[bytes, bytes] = None
        while True:
            neg_freq, inv_pair = heapq.heappop(pair_max_heap)
            if pair_freq.get(inv_pair.pair, 0) == -neg_freq:  # prevent stale pairs in the heap
                max_pair = inv_pair.pair
                break
            else:
                continue  # keep popping

        new_byte: bytes = max_pair[0] + max_pair[1]
        vocab[len(vocab)] = new_byte
   
        pretokens = list(lookup_pretoken[max_pair])  # copy & convert to list

        # reduce number of stale heap entries
        pair_freq_delta: dict[tuple[bytes, bytes], int] = defaultdict(int)

        for pretoken in pretokens:
            freq = pretoken_freq[pretoken]

            # construct new pretoken
            i = 0
            new_pretoken = []
            while i < len(pretoken):
                if i < len(pretoken)-1 and (pretoken[i], pretoken[i+1]) == max_pair:
                    new_pretoken.append(new_byte)
                    i += 2  # say we are A, *B, C, D and merged B, C -> BC, then we should be at D, so i+2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1
            new_pretoken = tuple(new_pretoken)

            # dec every old pair and inc every new pair
            for i in range(len(pretoken) - 1):
                pair = (pretoken[i], pretoken[i + 1])
                pair_freq_delta[pair] -= freq
                lookup_pretoken[pair].discard(pretoken)  # doesn't raise KeyError if pretoken not in the set
            for i in range(len(new_pretoken) - 1):
                pair = (new_pretoken[i], new_pretoken[i + 1])
                pair_freq_delta[pair] += freq
                lookup_pretoken[pair].add(new_pretoken)

            del pretoken_freq[pretoken]
            pretoken_freq[new_pretoken] += freq  # not sure if there is probability of this new pretoken already existing?
        
        # update pair freqs
        for pair, delta in pair_freq_delta.items():
            if delta == 0:
                continue
            pair_freq[pair] += delta
            if pair_freq[pair] == 0:
                del pair_freq[pair]
            else:
                heapq.heappush(pair_max_heap, (-pair_freq[pair], InversePair(pair)))

        merges.append(max_pair)

    return vocab, merges

def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)  # maps cleanly to extended latin unicode
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))

def start_peak_memory_monitor(interval: float = 0.1):
    """Poll the whole process tree every `interval` seconds; return a callable that stops and returns peak bytes."""
    proc = psutil.Process()
    peak_bytes = 0
    stop = threading.Event()

    def _run():
        nonlocal peak_bytes
        while not stop.is_set():
            try:
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                if rss > peak_bytes:
                    peak_bytes = rss
            except Exception:
                pass
            stop.wait(interval)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    def stop_and_get() -> int:
        stop.set()
        t.join()
        return peak_bytes

    return stop_and_get


def dump_bpe(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    b2u = bytes_to_unicode()
    def bytes_to_str(b: bytes) -> str:
        return ''.join([b2u[c] for c in b])
    vocab_b2u = {bytes_to_str(b): i for i, b in vocab.items()}
    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump(vocab_b2u, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "merges.txt"), "w") as f:
        for merge in merges:
            f.write(bytes_to_str(merge[0]) + ' ' + bytes_to_str(merge[1]) + "\n")


def load_bpe(output_dir: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """We need to reverse the bytes_to_unicode mapping to get the bytes back from the unicode strings"""
    vocab, merges = {}, []
    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}
    def str_to_bytes(s: str) -> bytes:
        return bytes([u2b[c] for c in s])
    with open(os.path.join(output_dir, "vocab.json"), "r") as f:
        vocab = json.load(f)  # str: int
        vocab = {str_to_bytes(k): v for k, v in vocab.items()}
    with open(os.path.join(output_dir, "merges.txt"), "r") as f:
        for line in f:
            merge = line.rstrip().split(" ")
            merges.append((str_to_bytes(merge[0]), str_to_bytes(merge[1])))
    return vocab, merges


def train_bpe_tinystories(profile: bool = False):
    if profile:
        pr = cProfile.Profile()
        pr.enable()
        vocab, merges = train_bpe(
            input_path="data/TinyStoriesV2-GPT4-valid.txt",
            vocab_size=10000,
            special_tokens=["<|endoftext|>"],
            profile_mode=profile,
        )
        pr.disable()
        stats = pstats.Stats(pr)
        stats.strip_dirs().sort_stats('cumulative')
        total = stats.total_tt
        print(f"{'%cum':>7} {'%self':>7} {'cum':>8} {'self':>8} {'ncalls':>10}  function")
        for func in stats.fcn_list[:20]:
            cc, nc, tt, ct, _ = stats.stats[func]
            ncalls = str(nc) if nc == cc else f"{nc}/{cc}"
            print(
                f"{ct / total:7.1%} {tt / total:7.1%} "
                f"{ct:8.3f} {tt:8.3f} {ncalls:>10}  {pstats.func_std_string(func)}"
            )
    else:
        get_peak = start_peak_memory_monitor()
        start = time.perf_counter()
        vocab, merges = train_bpe(
            input_path="data/TinyStoriesV2-GPT4-train.txt",
            vocab_size=10000,
            special_tokens=["<|endoftext|>"],
        )
        elapsed = time.perf_counter() - start
        peak_bytes = get_peak()
        logger.info(f"BPE training took {elapsed:.2f} seconds")
        logger.info(f"Peak memory usage (all processes): {peak_bytes / 1024 / 1024:.2f} MB")
    
    # log longest vocab token
    longest_token = max(vocab.values(), key=len)
    logger.info(f"Longest vocab token: {longest_token}")

    # serialize vocab and merges to disk - not as trivial as I thought
    # we need a function that converts from bytes to unicode that isn't decoding to utf-8
    output_dir = "bpe_output/tinystories"
    dump_bpe(vocab, merges, output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="tinystories")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.task == "tinystories":
        train_bpe_tinystories(profile=args.profile)