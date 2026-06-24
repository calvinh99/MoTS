I'm going to just read the official cs336 lecture links ie [lecture_01.py](https://cs336.stanford.edu/lectures/?trace=lecture_01) cuz they are much prettier. But maybe the `lectures/` folder will be useful to give AI agents context. Though I'm not sure if I will ask Cursor/Codex/Claude-Code inside this folder (pro: can access lecture materials for context) or just ask via ChatGPT/Claude website (pro: better UIUX & tracking of chat history).

# Lecture 1
https://cs336.stanford.edu/lectures/?trace=lecture_01

will learn mechanics of LLMs (hard knowledge, ie attention mechanism) & mindset (serious focus on scaling). intuitions are hard (seems need frontier lab experience).

Seems to have a lot of high level intutiions & introduces a lot of papers. But not sure if I should spend more time on this for just jump into assignment 1 directly.

# Stream-Of-Consciousness Worklog

I'll put what I do & random thoughts & notes here & organize later.

Going through [[cs336_assignment1_basics.pdf]].

Downloading **datasets**, a few gigs, should be ok for now, can delete later to make space on my macbook pro.

Using cursor composer 2.5 fast w/ the provided AGENTS.md to help make life easier but not do my work for me. I even ran the download cmds myself lol.

> not related but im curious abt the stanford course code to generate the materials & pdfs, seems it can integrate math, code, and more into one neat render.

**problem 1 is stupid**. since I'm not an actual student in this class, I'll be more picky about the problems I choose to spend time on and learn.

## Encoding Chars

`>>` means AI-written.

>> UTF-8 encodes text as bytes where the first byte tells you the group size — 'h' is one byte [104], while '牛' is three bytes [231, 137, 155] because 231 (11100011) means “2 more continuation bytes follow.” There are no separator bytes: 231 never stands alone as a character, and bytes like 137 (10000001) only make sense as followers, so the decoder reads 231,137,155 → 牛, then '!' → [33], and keeps going.

I love this explanation. Basically all chars are encoded as bytes, but obv there's not only 256 chars only, so chars are encoded/decoded as multiple bytes. w/o separator bytes, we use the first byte to tell us how many more bytes to read. This also means **only 128 chars are 1 byte** (the ones starting with 0xxxxxxx 0-127).

ASCII are the 128 chars that are single byte. utf-8 is a way to encode **Unicode** into bytes.

bytes aren't good for tokenization though cuz long sequence lengths. 10 words may be 50 bytes.

## BPE Tokenizer
We are implementing a byte-level BPE tokenizer, not a char-level BPE tokenizer.

**pre-tokenization** allows for big speedup. Here's a great [PR](https://github.com/openai/tiktoken/pull/234/files) to the openai tiktoken codebase introducing a better pretokenization regex.

We can parallelize pre-tokenization based on cpu core count. We need parallel processes not just parallel threads because we want to run CPU-intensive work in parallel on **multiple cores**. Due to python **GIL**, multithreading doesn't actually allow for parallel execution of python code on multiple cores for CPU-bound work because a thread must **acquire the GIL** before running.

We can optimize the merging step as well. A merge is going through our freq dict & updating byte sequences by replacing the pair w/ the new token.

> The pdf says to construct an index for the counts of all pairs. This is diff from pretoken updating. During merge we'd update `[t, h, e]` to `[th, e]`. The pdf is saying we construct pair freq index (from pretokens - already a speedup vs using the raw corpus), ie `[t, h]: 47` and `[h, e]: 67`. Then incrementally updating those counts, so if `[t, h] -> th` then...? How do we know `[h, e]` is part of `[t, h, e]` or `[s, h, e]`? I don't get it... OK so gpt helped me a bit, I get it now I think. Our index needs to record all the pretokens each pair occurs, not just the count. So when `[t, h] -> th`, we know `[t, h]` occurs in `[t, h, e]` at pos 0 and say "the" appears 47 times, then we grab all pairs of "the" which are `[t, h]` and `[h, e]` and the new pairs which is just `[th, e]`. We subtract 47 from the old pairs (not preserved) and add 47 to the new pair (incremental updates). So then our new pair index is `[t, h]: 0, [h, e]: 20, [th, e]: 47`. Our merge algo goes from `[t, h]` to `[t, h, e]` then to `[t, h], [h, e], and [th, e]`. It doesn't touch any other pair or pretokens.

A trained BPE is not just vocab, it's also the **merge_list** w/ order. Tokenizing is just like training BPE except you don't count freqs to pick new merge. You take the input string then merge until there's no merges left.

As I'm working through `train_bpe` I realized there's so many nuances in implementation that I miss. For example converting `bytes("pretoken")` into `[bytes([c]) for c in "pretoken"]`.

One **mistake** I corrected is the merge loop. First let's go over the optimizations we have. We have a **lookup table** from pair to list of pretokens that pair appears in so that our pretoken search time goes from `O(P * avg_len(P)) -> ~O(1)`. We have **pretokens** so that we don't need to iterate through the whole text corpus O(T) and instead only need to iterate through all unique pretokens O(P), where P << T. We have a map of pair frequencies so that rather than recomputing all byte pair frequencies after every merge which is O(P * avg_len(P)) we can **update only the pairs affected** which is ~O(1).

I made some more mistakes, let's go over this next one. So for each pretoken that the pair being merged appears in, we need to:
1. find its occurrence in the pretoken
2. inc/dec the pairs affected
3. update the pair to pretokens lookup table
I initially tried to do a live update iteration (not sure what the best word for this is). Basically I would iterate through the affected pretoken, and soon as I find the pair, I would then calculate the prev/next pairs (which would be changed, ie for `A, B, C, D` if `B, C -> BC`, then `A, B -> A, BC`, etc) and then immediately apply dec/inc. But there's problems, the most notable being that if I had `B, C, B, C`, then I would prematurely dec `C, B` (correct) and inc `BC, B` (wrong) when the reality is the seq becomes `BC, BC`. I guess it's kind of stupid and I didn't really think hard before implementation (I'm just getting back into learning). The solution was simple:
1. turn the old pretoken into the new pretoken, ie old = `B, C, B, C` and new = `BC, BC`
2. for each pair in old, dec, and for each pair in new, inc

>> After AI review, another optimization I can make is add a max heap so that finding most freq pair is O(1) lookup time & popping it to get next max freq pair is O($log_2(pairs)$). We still need our freq map though to inc/dec in O(1). But then the question of - how do we deal with stale pairs in the heap? It's simple, when we pop from heap we double check the freq from the heap w/ the freq in the map, if not equal we discard and pop again. Our pair freq map is the source of truth.

WOW! After implementing lazy max heap, `test_train_bpe.py` time dropped from 1s to 0.6s! This is big. Also, a detail is to track net pair freq deltas across all pretokens (once per merge) then apply at the end & push onto heap only on change, this **prevents excess stale pairs** in heap.

Now that I've successfully trained a bpe on tiktoken, let me go over the key details. First we chunk the text corpus by `<|endoftext|>` special token. Then for each chunk we run parallel pre-tokenization processes because the work is CPU core intensive. We map-reduce to produce a frequency map for pretokens. **Pre-tokenization** saves probably multiple OOMs of work as we no longer need to iterate across the entire text corpus to update byte-pair counts. The only flaw potentially is it's impossible to count/form pairs that span token boundaries (ie end of a pretoken and start of another pretoken), but it's an acceptable tradeoff (pretokens are basically words anyway). Next is simply the **merge loop**. Let's think about what needs to be done. We need to find the most frequent pair. Then merge it and update pair freqs across all affected pairs. That's it. To find the most frequent pair we use a **lazy max heap** by pushing pairs and their freq each time an update occurs, this makes it O(logn) roughly rather than O(nlogn). It's lazy because for many pairs, their freqs in the heap are outdated so we need to check with our pair freq lookup table to make sure. Yes we create a pair freq lookup table from the pretoken freq lookup table. Next for the merge, we have a pair to pretoken lookup table so we know in O(1) time which pretokens need to be updated. For each old pretoken we create the new pretoken and compute pair deltas - which pairs dropped in freq and which pairs increased in freq. We apply these updates to pair freq lookup table. We also update the old pretokens to the new ones. Since these updates are computed incrementally they are also roughly O(P * len(P)) time, where P is number of pretokens per pair, rather than O(TP) where TP is all pretokens due to needing to rerun freqs for all pairs after merge. To me now these optimizations are all common sense now.

So:
- **pretokenization** for reducing text corpus size
	- parallelize
- **lazy max heap** for most freq pair
- **pair to pretokens** lookup for updating after merge
- **incrementally update pair freqs** for only affected pairs

**Serializing vocab and merges to disk**
Lowkey this trumped me for a bit. How do we save bytes to disk in a way that is **human-readable** but also **revertible** back into bytes. Had to consult w/ Claude who recommended GPT2's bytes_to_unicode method. Basically map each singular byte to a readable char. This is mostly hardcoded from knowledge of the **unicode directory** (140k+ chars). From the first 256 bytes there's 188 "good" readable chars and 68 "bad" chars that need to get remapped. The good are composed from 3 code point ranges (code point is the number assigned to a unicode char): 33 to 126, 161 to 172, 174 to 255. For all bad chars we simply increment their code point by 256 (maps to extended Latin like "Ġ"). So for vocab, we run bytes_to_unicode mapping on each byte and same for merges. Also turn off **ascii escaping** when dumping vocab json (so that we get the actual glyph like "Ā" instead of "\u0100").

**Profiling train_bpe**
To answer `train_bpe_tinystories`, training took ~34s and it used ~174MB peak memory. The longest token is `b' accomplishment'` which makes sense. I assume this is true because the word or pretoken 'accomplishment' appears pretty frequently within the tinystories training corpus (within the top 10k most freq).

For profiling I ran it on the tinystories validation corpus where I use a single process for pretokenization. I found that pretokenization takes **~82%** of the total time spent (makes sense, merge loop is pretty fast due to pretokenization and all the other optimizations we do). The bulk of this time is purely **iterating through the corpus & incrementing the pretoken counter**, not the regex (precompile the regex pattern & use finditer to prevent loading all in mem).

Nvm, the prev peak memory is **wrong**. I did some napkin math, at 8 processes and ~265MB per chunk/progress (from tinystores train dataset size), we would get at least 2GB of mem usage if not more. So our peak mem was only tracking the main process (not the 8 child processes). I was lazy and asked Sonnet 4.6 to code up a bg thread that monitors true peak mem usage by summing **RSS** (Resident Set Size - each process' RAM usage) across all processes spun from main processes recursively and it got **7618MB or ~7.44GB**!

I made the profiling code messy (distinct profiling logic setting num_processes to 1, separate bg thread for peak memory usage). But whatever, we can fix that later.


**Tokenizer**
This actually encodes text into tokens and decodes tokens into text given vocab & merges. In the assignment it recommends adding an optional `special_tokens` param but idk why (isn't special tokens in the vocab alr?).

took me ~1.5 hrs to finished the tokenizer problem suboptimally.

after a walk & another 30 mins I optimized it (tests finished in ~1s vs ~10s) by changing the merge logic to using a lookup from merge pair -> rank. This way we iterate over only the pairs in the pretoken rather than iterating over all merges (equivalent to vocab size).
