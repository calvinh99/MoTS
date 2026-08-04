from cs336_basics.bpe import load_bpe
import regex as re
from typing import Iterable

class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.merge_lookup: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}
        self.inverse_vocab: dict[bytes, int] = {v: k for k, v in vocab.items()}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] = None
    ):
        vocab, merges = load_bpe(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        We must first pretokenize since training bpe we created the merges
        on pretokens as well.
        """

        # we can do list of pretokens, then apply merges to each, but then freq pretokens will experience repeated merge work
        # so lets do pretoken -> list of positions in seq
        # each pretoken will experience merges until no merges left then get broken into list of token ids
        # we will then join all these lists tgt by iterating through all values in this map
        # then collapse list of lists into a flat list of ints
        # and also join w/ special tokens
        if self.special_tokens is None:
            text_parts = [text]
        else:
            special_tokens = sorted(self.special_tokens, key=len, reverse=True)  # longest to shortest so longer special tokens dont get broken up
            special_token_set = set(special_tokens)
            special_token_PAT = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"  # this pattern doesn't drop the special token
            text_parts = re.split(special_token_PAT, text)  # not iteratble

        pretoken_positions: dict[str, list[int]] = {}
        special_token_positions: dict[str, list[int]] = {}

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        PAT = re.compile(PAT)  # precompilng helps
        position = 0
        for text_part in text_parts:  # this is sequential, we could probably parallelize this
            if self.special_tokens is not None and text_part in special_token_set:
                if text_part not in special_token_positions:
                    special_token_positions[text_part] = []
                special_token_positions[text_part].append(position)
                position += 1
                continue

            for pt in PAT.finditer(text_part):
                pretoken = pt.group()
                if pretoken not in pretoken_positions:
                    pretoken_positions[pretoken] = []
                pretoken_positions[pretoken].append(position)
                position += 1
        
        pretoken_to_ids: dict[str, list[int]] = {}

        # merges
        for pretoken in pretoken_positions:
            # this is # merges x # pairs = ~5 x ~5 = ~25 (10k faster!)
            # impl 1
            pretoken_bytes = [bytes([b]) for b in pretoken.encode("utf-8")]
            while True:  # merge loop
                i = 0
                merge_idx = 0
                merge_rank = float('inf')
                while i < len(pretoken_bytes) - 1:
                    pair = (pretoken_bytes[i], pretoken_bytes[i + 1])
                    if pair in self.merge_lookup:
                        rank = self.merge_lookup[pair]
                        if rank < merge_rank:
                            merge_idx = i
                            merge_rank = rank
                    i += 1
                if merge_rank == float('inf'):
                    break  # no more merges

                # insert merged pair
                new_pretoken_bytes = pretoken_bytes[:merge_idx] + [pretoken_bytes[merge_idx] + pretoken_bytes[merge_idx + 1]] + pretoken_bytes[merge_idx + 2:]
                pretoken_bytes = new_pretoken_bytes

                # impl 2
                # this uses more memory ig?
                # pairs = [(pretoken_bytes[i], pretoken_bytes[i + 1]) for i in range(len(pretoken_bytes) - 1)]
                # earliest_pair_idx = 0
                # earliest_pair_rank = float('inf')
                # for i, pair in enumerate(pairs):
                #     if pair in self.merge_lookup:
                #         rank = self.merge_lookup[pair]
                #         if rank < earliest_pair_rank:
                #             earliest_pair_idx = i
                #             earliest_pair_rank = rank
                # if earliest_pair_rank == float('inf'):
                #     break  # no more merges
                # for i in range(len(pairs)):
                #     if i == earliest_pair_idx:
                #         new_pretoken_bytes.append(pairs[i][0] + pairs[i][1])
                #         continue
                #     elif i > earliest_pair_idx:
                #         new_pretoken_bytes.append(pairs[i][1])
                #         continue
                #     else:
                #         new_pretoken_bytes.append(pairs[i][0])
                #         continue
                # pretoken_bytes = new_pretoken_bytes
            
            # impl 3
            # this is # merges x # pairs x len(merges) = ~5 x ~5 x 10k = 250k
            # pretoken_bytes = [bytes([b]) for b in pretoken.encode("utf-8")]
            # for pair in self.merges:  # sequential in order of merges
            #     i = 0
            #     new_pretoken_bytes = []
            #     while i < len(pretoken_bytes):
            #         if i < len(pretoken_bytes) - 1:
            #             pretoken_pair = (pretoken_bytes[i], pretoken_bytes[i + 1])
            #             if pair == pretoken_pair:
            #                 new_pretoken_bytes.append(pair[0] + pair[1])
            #                 i += 2
            #                 continue
            #         new_pretoken_bytes.append(pretoken_bytes[i])
            #         i += 1
            #     pretoken_bytes = new_pretoken_bytes
            
            # convert to list of int ids
            pretoken_to_ids[pretoken] = [self.inverse_vocab[b] for b in pretoken_bytes]
        
        for special_token in special_token_positions:
            pretoken_to_ids[special_token] = [self.inverse_vocab[special_token.encode("utf-8")]]

        # reconstruct list of ids
        # join special token positions w/ pretoken_positions
        pretoken_positions.update(special_token_positions)
        ids = [[] for _ in range(position)]
        for pretoken, positions in pretoken_positions.items():
            for pos in positions:
                ids[pos] = pretoken_to_ids[pretoken]
        # flatten list of lists into list
        ids = [id for sublist in ids for id in sublist]
        return ids


    def encode_iterable(self, iterable: Iterable[str]) -> list[int]:
        """This is actually simpler than encode. We just match pretokens & then run merge on the spot."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Simple. convert each id to bytes using vocab, then join all bytes & then decode from utf-8 into str."""
        bytes_list = [self.vocab[id] for id in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")