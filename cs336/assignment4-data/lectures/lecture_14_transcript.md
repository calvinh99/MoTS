# lecture 14

## Lecture 14: Data II

Last lecture:

- Live service (e.g., GitHub) → dump/crawl (e.g., GitHub Archive) → processed data (e.g., The Stack)

- Considerations: terms of service, copyright (licenses or fair use)

This lecture:

- Data pipeline: transformation, filtering, deduplication, mixing

- Mid-training + SFT: synthetic data

Raw data does not come as text.

It is HTML, PDF (arxiv), or directories (code repositories).

HTML to text (main one):

- Remove boilerplate (e.g., navigation, ads) and extract content

- What about images, tables, etc.?

- Inherently lossy (need to linearize)

- Tools (rule-based): trafilatura, resiliparse, jusText, lynx, etc.

- Accuracy matters: 

- [[Li+ 2024]]([object Object])

![images/dclm-wet.png](images/dclm-wet.png)

FinePDFs 

- [[post]]([object Object])

![var/files/image-e9e730a30cd647ff18f24f33cae588c0-https_huggingfacefw-finepdfsblog_hf_space__astro_pdf-description_Cb49jXc6_Z17eX4E_webp](var/files/image-e9e730a30cd647ff18f24f33cae588c0-https_huggingfacefw-finepdfsblog_hf_space__astro_pdf-description_Cb49jXc6_Z17eX4E_webp)

- Source: Common Crawl

- Recrawl truncated PDFs (since they are big)

- OCR (RolmOCR) using a VLM or Docling (make these run fast)

- Lots of cleanup and filtering

- A lot of layout information is missing

Algorithmic building block:

- Given some **target data** T and lots of **raw data** R, find subset T' of R similar to T.

![images/raw-target-schema.png](images/raw-target-schema.png)

Applications:

- Language identification (English versus rest)

- Quality filtering (high quality versus low quality)

- Toxicity filtering (non-toxic versus toxic)

Desiderata for filtering algorithm:

- Generalize from the target data (want T and T' to be different)

- Extremely fast (have to run it on R, which is huge)

Survey paper on data selection 

- [[Albalak+ 2024]]([object Object])

General framework: Given target T and raw R, find subset of R similar to T

1. Estimate some model based on R and T and derive a scoring function

2. Keep examples in R based on their score

Types of classifiers:

- Generative model of T (KenLM): score(x) = p_T(x)

- Simple classifier (fastText): score(x) = p(T | x)

To use: keep examples x with score(x) >= threshold (stochastically)

Model-based filtering?

- Some deliberately do not use model-based filtering (C4, Gopher, RefinedWeb, FineWeb, Dolma)

- Some use model-based filtering (GPT-3, LLaMA, DCLM) [becoming the norm]

Language identification:

- Goal: find text of a specific language (e.g., English)

- fastText language identification 

- [[article]]([object Object])

- Off-the-shelf classifier

- Supports 176 languages

- Trained on multilingual sites: Wikipedia, Tatoeba (translation site) and SETimes (Southeast European news)

- Dolma keeps pages with p(English) >= 0.5 

- [[Soldaini+ 2024]]([object Object])

OpenMathText 

- [[Paster+ 2023]]([object Object])

- Goal: curate large corpus of mathematical text from CommonCrawl

- Use rules to filter (e.g., contains latex commands)

- KenLM trained on ProofPile, keep if perplexity < 15000

- Trained fastText classifier to predict mathematical writing, threshold is 0.17 if math, 0.8 if no math

- Result: produced 14.7B tokens, used to train 1.4B models that do better than models trained on 20x data

GPT-3 

- [[Brown+ 2020]]([object Object])

- Positives: samples from {Wikipedia, WebText2, Books1, Books2}

- Negatives: samples from CommonCrawl

Train linear classifier based on word features 

Keep documents stochastically based on score

LLaMA/RedPajama 

- [[Touvron+ 2023]]([object Object])

- Positives: samples from pages **referenced** by Wikipedia

- Keep documents that are classified positive

phi-1 

- [[Gunasekar+ 2023]]([object Object])

- Philosophy: really high quality data (textbooks) to train a small model (1.5B)

- Includes synthetic data from GPT 3.5 (later: GPT-4) and filtered data

- Train random forest classifier on T using output embedding from pretrained codegen model

- Select data from R that is classified positive by the classifier

Result on [HumanEval](https://huggingface.co/datasets/openai_humaneval):

- Train 1.3B LM on Python subset of The Stack (performance: 12.19% after 96K steps)

- Train 1.3B LM on new filtered subset (performance: 17.68% after 36K steps) - better!

Toxicity filtering in Dolma 

- Dataset: Jigsaw Toxic Comments dataset (2018) 

- [[dataset]]([object Object])

- Project goal: help people have better discussions online 

- Data: comments on Wikipedia talk page annotated with {toxic, severe_toxic, obscene, threat, insult, identity_hate}

Scale-dependent effects of filtering:

- No single optimal threshold for filtering

- If training for longer, want more (lower quality) data

- If training for shorter, want less (higher quality) data

![images/data-filtering-scale.png](images/data-filtering-scale.png)

Summary:

- Filtering is critical for building a good model

- Recipe: define target data (what good looks like), extrapolate to raw data

Two types of duplicates:

- Exact duplicates (mirror sites, GitHub forks) 

- [[Gutenberg mirrors]]([object Object])

- Near duplicates: same text differing by a few tokens

Examples of near duplicates:

- Terms of service and licenses 

- [[MIT license]]([object Object])

- Formulaic writing (copy/pasted or generated from a template) 

![var/files/image-bd6f945561f42be108f3dd1de0ace52e-https_d3i71xaburhd42_cloudfront_net_4566c0d22ebf3c31180066ab23b6c445aeec78d5_5-Table1-1_png](var/files/image-bd6f945561f42be108f3dd1de0ace52e-https_d3i71xaburhd42_cloudfront_net_4566c0d22ebf3c31180066ab23b6c445aeec78d5_5-Table1-1_png)

- Minor formatting differences in copy/pasting

Product description repeated 61,036 times in C4

'“by combining fantastic ideas, interesting arrangements, and follow the current trends in the field of that make you more inspired and give artistic touches. We’d be honored if you can apply some or all of these design in your wedding.  believe me, brilliant ideas would be perfect if it can be applied in real and make the people around you amazed!

- [[example page]]([object Object])

Deduplication training data makes language models better 

- [[Lee+ 2021]]([object Object])

- Train more efficiently (because have fewer tokens)

- Avoid memorization (can mitigate copyright, privacy concerns)

Design space:

1. What is an item (sentence, paragraph, document)?

2. How to match (exact match, existence of common subitem, fraction of common subitems)?

3. What action to take (remove all, remove all but one)?

Key challenge:

- Deduplication is fundamentally about comparing items to other items

- Need linear time algorithms to scale

- Hash function h maps item to a hash value (integer or string)

- Hash value much smaller than item

- Hash collision: h(x) = h(y) for x ≠ y

Tradeoff between efficiency and collision resistance 

- Cryptographic hash functions (SHA-256): collision resistant, slow (used in bitcoin)

- DJB2, MurmurHash, CityHash: not collision resistant, fast (used for hash tables)

We will use MurmurHash:

**Simple example**

1. Item: string

2. How to match: exact match

3. Action: remove all but one

- Pro: simple, clear semantics, high precision

- Con: does not deduplicate near duplicates

- This code is written in a MapReduce way, can easily parallelize and scale

**C4** 

- [[Raffel+ 2019]]([object Object])

1. Item: 3-sentence spans

2. How to match: use exact match

Warning: when a 3-sentence span is removed from the middle of a document, the resulting document might not be coherent

Let's now look at approximate set membership.

First we need a similarity measure.

### Jaccard similarity

Definition: Jaccard(A, B) = |A intersect B| / |A union B|

Definition: two documents are **near duplicates** if their Jaccard similarity >= threshold

Algorithmic challenge: find near duplicates in linear time

### MinHash

MinHash: a random hash function h so that Pr[h(A) = h(B)] = Jaccard(A, B)

Normally, you want different items to hash to different hashes

...but here, you want collision probability to depend on similarity

Characteristic matrix representation:

item | A | B

1    | 1 | 1

2    | 1 | 1

3    | 1 | 1

4    | 1 | 0

5    | 0 | 1

Random hash function induces a permutation over items

Look at which item is first in A and which item is first in B.

Each item has the same probability as being first (min)

- If 1, 2, 3 is first, then first in A = first in B.

- If 4, 5 is first, then first in A ≠ first in B.

Now we can hash our items, but a collision doesn't tell us Jaccard(A, B) > threshold.

Locality sensitive hashing (LSH) 

- [[book chapter]]([object Object])

Suppose we hash examples with just one MinHash function

P[A and B collide] = Jaccard(A, B)

On average, more similar items will collide, but very stochastic...

Goal: have A and B collide if Jaccard(A, B) > threshold

We have to somehow sharpen the probabilities...

Solution: use n hash functions

Break up into b bands of r hash functions each (n = b * r)

Hash functions:

h1 h2 h3 h4  |  h5 h6 h7 h8  |  h9 h10 h11 h12

Key: A and B collide if for *some* band, *all* its hash functions return same value

As we will see, the and-or structure of the bands sharpens the threshold

Given Jaccard(A, B), what is the probability that A and B collide?

**Example**

![var/files/image-5c7429f9fdd2bf58b7c5651aebc8f045-https_cdn_sanity_io_images_vr8gru94_production_b470799575b8e77911bacb8500977afef06d6c85-1280x720_png](var/files/image-5c7429f9fdd2bf58b7c5651aebc8f045-https_cdn_sanity_io_images_vr8gru94_production_b470799575b8e77911bacb8500977afef06d6c85-1280x720_png)

Increasing r sharpens the threshold and moves the curve to the right (harder to match)

Increasing b moves the curve to the left (easier to match)

![var/files/image-7666e77b1a420b4da170c895b069684e-https_cdn_sanity_io_images_vr8gru94_production_aace49fa240778e8ecf6e85ad08a2de7f5385566-1280x720_png](var/files/image-7666e77b1a420b4da170c895b069684e-https_cdn_sanity_io_images_vr8gru94_production_aace49fa240778e8ecf6e85ad08a2de7f5385566-1280x720_png)

Example setting 

: n = 9000, b = 20, r = 450

What is the threshold (where the phase transition happens)?

Probability that a fixed band matches:

Probability that A and B collide is a constant (≈ 1-1/e):

Recall that language models are trained on multiple data sources.

Datasets in Marin: 

- [[token viewer]]([object Object])

![images/marin-token-viewer.png](images/marin-token-viewer.png)

The Pile 

- [[Gao+ 2020]]([object Object])

![var/files/image-4eb29ee713b99ea34eb86b995bd32bfd-https_stanford-cs324_github_io_winter2022_lectures_images_the-pile_png](var/files/image-4eb29ee713b99ea34eb86b995bd32bfd-https_stanford-cs324_github_io_winter2022_lectures_images_the-pile_png)

Key question: what distribution over the data sources should we use?

Example:

Baselines:

- Vibes: set p(s) manually based on intuition (quite common)

- Uniform sampling: sample uniformly (p(s) ∝ 1)

- Proportional mixing: sample proportional to the number of tokens in a source (p(s) ∝ num_tokens(s))

Intuition: should upweight higher quality sources

However...

1. We want to ensure diversity (e.g., across incomparable sources: literature, code, papers)

2. Each source is finite, so if put too much weight on a small source, then need to epoch over it

This last point is important and a bit subtle.

50x epochs on high quality data...can lead to overfitting!

UniMax 

- [[Chung+ 2023]]([object Object])

- Setting: balancing different languages for multilingual models

- Previous work: between uniform and proportional mixing (p(s) ∝ num_tokens(s)^α for α in [0, 1])

- Idea: sample sources uniformly but with a hard **cap** C on number of epochs for any source

- Specifically, p(s) * num_training_tokens ≤ C for all sources s

Regression-based mixing 

- [[Liu+ 2024]]([object Object])

- [[Chen+ 2026]]([object Object])

![images/regmix.png](images/regmix.png)

- Define distribution over mixtures `p` (e.g., Dirichlet) 

- Define regression method (e.g., linear, gradient boosted trees)

- Define target based on downstream evals (careful not to overfit!)

- Discrepancy between small and large scale (tradeoff cost and accuracy)

![images/data-mixing-methods.png](images/data-mixing-methods.png)

Hope 1: regression model is accurate at minimizer 🙏

Hope 2: optimal data mixtures transfer from small to large scale 🙏

Hold on. There's at least one scale-dependent effect:

- If train small models on low token counts:

- But if train large model on this mixture, we will epoch a ton on high quality data and overfit!

Simulated epoching 

- [[Held+ 2025]]([object Object])

- General idea: make small scale look like large scale (general theme of this course)

- Instantiation: downsample all sources proportionally

- In this downsampled mixture, models that epoch too much won't look good.

- So the optimum will be more balanced.

- Problem: how to weight different data sources (e.g., Wikipedia, general, code)

- Regression-based mixing: estimate mixture → loss at small scale, optimize (analogous to scaling laws)

- Important consideration: epoching and overfitting (solution: cap or simulated)

Recipe:

1. Define a set of environments

2. Define a set of tasks / prompts

3. Collect responses from a strong model (teacher)

OpenThoughts 

- [[Guha+ 2025]]([object Object])

- 1.2M examples using QwQ-32B as a teacher

- Questions come from 27 human and synthetic sources (e.g., StackExchange, NuminaMath, Chemistry)

![images/openthoughts-sources.png](images/openthoughts-sources.png)

- Sampling multiple (16) responses per prompt is helpful

- Better models aren't necessarily better teachers: QwQ-32B is a better teacher than DeepSeek-R1

- Answer filtering wasn't helpful

- Smaller high quality sources (e.g., OpenMath-2-Math) is better than large diverse sources

![images/openthoughts-pipeline.png](images/openthoughts-pipeline.png)

SWE-smith 

- [[Yang+ 2025]]([object Object])

![images/swe-smith.png](images/swe-smith.png)

- Given a repository, use LM to generate tasks (introduce bugs with LM)

- 128 GitHub repositories yields 50K tasks

SWE-Zero

- [[Ludwig+ 2026]]([object Object])

- SWE tasks have heavy dependencies (unlike math or coding contests)

- Setting up thousands of Docker images is an infrastructural nightmare

- Observation: strong models can solve many tasks without execution feedback

![images/swezero-noexec.png](images/swezero-noexec.png)

Key: strong models have internal "world model" of code semantics

- SWE-Zero: 300K agent trajectories that don't require repository-specific execution

- 150K GitHub PRs

- OpenHands scaffold, remove future git commits to prevent "git hacking" by agent

![images/swezero-prompt.png](images/swezero-prompt.png)

- Distilled from Qwen3-Coder-480B + filtering (try to execute anyway)

- SWE-Hero: 13K agent trajectories that do require execution feedback

![images/swezero-results.png](images/swezero-results.png)

SWE-rebench 

- [[Badertdinov+ 2025]]([object Object])

- 21K interactive Python SWE tasks from 3.4K GitHub repositories

- 450K PRs from GitHub and GitHub Archive

- Used Qwen 2.5-72B-Instruct to install dependencies and assess PR quality

![images/swe-rebench.png](images/swe-rebench.png)

SWE-ZERO-12M-trajectories 

- [[data]]([object Object])

- Scale SWE-Zero up to 12M agent trajectories

- Used the SWE-rebench-v2 tasks (32K executable tasks + 120K nonexecutable tasks)

- Ran mini-coder-1.7b (very small model, 50.4 pass@100), mini-swe-agent scaffold

- [Example](https://huggingface.co/datasets/AlienKevin/SWE-ZERO-12M-trajectories/viewer/default/train?row=5&conversation-viewer=0)

- Generating prompts: fully-synthetic, semi-synthetic (real environment + synthetic tasks), real (GitHub PRs)

- Responses: from capable models (that are also good teachers)

- Code environments are painful

- Lots of filtering and other details

- Filtering: train classifier (language id, quality, toxicity) for what good looks like

- Deduplication: hashing scales to large datasets for fuzzy matching

- Mixing: try mixtures at small scale, extrapolate to optimal mixture and large scale

- Applications: language identification, quality filtering, toxicity filtering

- Post-training data: looks like evaluations, use of synthetic data

- A lot of data work is domain-specific, looking at examples, etc.
