# lecture 13

## Lecture 13: Data I

Previous lectures: how to train a model *given data*

Next two lectures: *what data* should we train on?

**Data** is the most important thing to get right in training language models.

One justification: let's see what companies disclose.

Open-weight models (e.g., Llama 3 

- [[Grattafiori+ 2024]]([object Object])

 have full transparency into architecture

...and even training procedures

...but basically no information on data.

![images/llama3-data.png](images/llama3-data.png)

Reasons for secrecy:

1. Competitive dynamics

2. Copyright liability

- Before foundation models, data work meant heavy annotation of labeled data for supervised learning.

- Now there's less annotation, but there's still a lot of curation and cleaning.

- Data is fundamentally a long-tail problem, scales with human effort (unlike architectures, systems).

Stages of training:

1. Pre-training: train on raw text (e.g., documents from the web)

2. Mid-training: train more on high quality data to enhance capabilities

3. Post-training: train on chat transcripts or reinforcement learning

In practice, the lines are blurry and there could be more stages

...but the basic trend is throughout training, we go from

large amounts of lower quality data to

small amounts of high quality data.

Terminology:

- Base model: after pre-training + mid-training

- Instruct/chat model: after post-training

(Increasingly, base models are not released - e.g., Qwen3.5-397B-A17B is an instruct model.)

Example (OLMo from AI2) 

- [[Team OLMo 2024]]([object Object])

1. **Pre-training**

![images/olmo2-pretraining.png](images/olmo2-pretraining.png)

2. **Mid-training**

![images/olmo2-dolmino.png](images/olmo2-dolmino.png)

3. **Post-training** 

- [[Lambert+ 2024]]([object Object])

![images/tulu.png](images/tulu.png)

What are these datasets?  How are they chosen and processed?

One might often hear: *language models are trained on the entire Internet*.

Slightly more accurately, ~Internet~ public (world wide) web.

But this is not quite right either...

First, the web consists of a set of live servers that one can connect to:

`$ curl https://cs336.stanford.edu/`

You can't train on live servers.

A **crawler**:

- Discovers webpages (starting from a seed set)

- Downloads the discovered webpages

However, you can't download and train on all the webpages.

Dynamic content:

- Many sites these days are apps

- URL doesn't change

- Need to click buttons and submit forms to access content

- Examples: Discord, wandb

Authentication:

- Sometimes need login with an account (and pay usually)

- Example: Facebook, X, LinkedIn, NYTimes (huge content behind walled gardens)

Technical restrictions:

- Not allowed to download some content based on `robots.txt` ([example](https://www.nytimes.com/robots.txt)) (voluntary)

- Website might use Cloudflare to detect and block bot activity (present CAPTCHAs)

- Website might block certain IP addresses / countries

- Website might have rate limits

Legal restrictions:

- Terms of service (ToS) might prohibit downloading using bots

- You might not have a license to copy the webpages (for training)

Decline of consent 

- [[Longpre+ 2024]]([object Object])

- Examined restrictions (robots.txt, ToS) for URLs in common datasets (C4, RefinedWeb, Dolma)

- Restrictions have increased over time

![images/decline-consent.png](images/decline-consent.png)

When crawlers are not well-behaved:

![images/anthropic-crawling.png](images/anthropic-crawling.png)

- Factors: ToS, robots.txt, server load (degrades service, costs website money)

- And then there is copyright (more later)...

Shadow libraries 

- [[article]]([object Object])

- Technically part of the web

- Examples: Library Genesis (LibGen), Z-Library, Anna's Archive, Sci-Hub

- Disregards copyright and bypasses paywalls (e.g., Elsevier)

- Received takedown orders, lawsuits, blocked in various countries

- Usually controls are circumvented, have servers in various countries

- Some argue this makes freely available what should be free

- From a legal perspective, this is piracy and copyright infringement

- LibGen has ~4M books (2019), Sci-Hub has ~88M papers (2022)

Summary:

- The Internet is huge

- Many technical and legal restrictions on what data one can access

What data is legal to use (for training)?

### Intellectual property law

- Goal: *incentivize* the creation of intellectual goods

- Types of intellectual property: copyright, patents, trademarks, trade secrets.

**Copyright law**:

- Goes back to 1709 in England (Statute of Anne), first time regulated by governments and courts 

- In United States, most recent: Copyright Act of 1976 

- Copyright protection applies to *'original works of authorship fixed in any tangible medium of expression, now known or later developed, from which they can be perceived, reproduced, or otherwise communicated, either directly or with the aid of a machine or device'*

- Collections are not original works so hence not copyrightable (e.g., telephone directories) unless there is some creativity in the selection or arrangement

- Copyright applies to expression, not ideas (e.g., quicksort)

- Expanded scope from 'published' (1909) to 'fixed' (1976)

- Registration not required for copyright protection (in contrast with patents)

- Threshold for copyright is extremely low (e.g., your website is copyrighted)

- Registration is required before creator can sue someone for copyright infringement

- Costs $65 to register 

- Lasts for 75 years, and then the copyright expires and it becomes part of the public domain (works of Shakespeare, Beethoven, most of Project Gutenberg, etc.)

Summary: *basically everything on the Internet are copyrighted.*

How to use a copyrighted work:

1. Get a license for it.

2. Appeal to the fair use clause.

### Licenses

- A license (from contract law) is granted by a licensor to a licensee.

- Effectively, 'a license is a promise not to sue'.

- The Creative Commons license enables free distribution of copyrighted work.

- Examples: Wikipedia, Open Courseware, Khan Academy, Free Music Archive, 307 million images from Flickr, 39 million images from MusicBrainz, 10 million videos from YouTube, etc.

- Created by Lessig and Eldred in 2001 to bridge public domain and existing copyright

Many model developers license data for training foundation models

- Google and Reddit 

- OpenAI and Shutterstock 

- OpenAI and StackExchange 

**Fair use (section 107)**:

Four factors to determine whether fair use applies:

1. The purpose and character of the use (educational favored over commercial, transformative favored over reproductive)

2. The nature of the copyrighted work (factual favored over fictional, non-creative over creative)

3. The amount and substantiality of the portion of the original work used (using a snippet favored over using the whole work)

4. The effect of the use upon the market (or potential market) for the original work

Examples of fair use:

- You watch a movie and write a summary of it

- Reimplement an algorithm (the idea) rather than copying the code (the expression)

- Google Books index and show snippets (Authors Guild v. Google 2002-2013)

Copyright is not about verbatim memorization:

- Plots and characters (e.g., Harry Potter) can be copyrightable

- Parody (imitating to make fun of something) is likely fair use

Copyright is about semantics (and economics).

Considerations for language models:

- Copying data (first step of training) is violation already even if you don't do anything with it.

- Training a model should be transformative (far from just copy/pasting).

- Model should be about the general idea (e.g., wizards), not in the concrete expression (e.g., Harry Potter).

- Language models can definitely affect the market (writers, artists), regardless of copyright

**Terms of service**:

- Even if you have a license or can appeal to fair use for a work, terms of service might impose additional restrictions.

- Example: YouTube's terms of service prohibits downloading videos, even if the videos are licensed under Creative Commons.

### Lawsuits

The New York Times v. OpenAI (2023)

- Allegation: for training and reproducing NYT articles

Authors (Bartz, Graeber, ...) v. Anthropic (2024):

- Allegation: for pirating millions of books and training on plaintiff's books

- Summary judgement (2025): training on plaintiff's works is fair use

- ...but pirating copies is not (even if don't train)

- Anthropic also bought and scanned the books; this is also fair use (but too late)

- Outcome: Anthropic paid $1.5B to authors to settle

Authors (Kadrey, Silverman, ...) v. Meta 

- Allegation: for training on plaintiff's books (revealed in the Llama paper)

- Summary judgement (2025): training on books (in this instance) is fair use 

- Allegation of torrenting books is still pending

- So far training has been deemed fair use (for specific instances, but unclear in general)

- Pirating books is clearly illegal

- Still a very active, evolving area

[Common Crawl](https://commoncrawl.org/) is a non-profit organization founded in 2007.

Statistics:

- Every ~month, run a web crawl (add 3-5 billion web pages)

- Crawls have some overlap but try to diversify

- 300 billion pages so far

- How many URLs are there? Hard to estimate, but O(billions)

- Google search index is at least 100 PB 

- [April 2026 Crawl](https://commoncrawl.org/blog/april-2026-crawl-archive-now-available) has 2.19 billion pages (372.2 TB)

Crawling uses Apache Nutch 

![var/files/image-07b10954a59946927c4c7c28d8847cc1-https_upload_wikimedia_org_wikipedia_commons_thumb_d_df_WebCrawlerArchitecture_svg_330px-WebCrawlerArchitecture_svg_png](var/files/image-07b10954a59946927c4c7c28d8847cc1-https_upload_wikimedia_org_wikipedia_commons_thumb_d_df_WebCrawlerArchitecture_svg_330px-WebCrawlerArchitecture_svg_png)

- Starts with a set of seed URLs (at least hundreds of millions) 

- Pop a URL from the queue, download URL, and add hyperlinks to queue

Policies 

- Selection policy: which pages to download?

- Politeness policy: respect robots.txt, don't overload server

- Re-visit policy: how often to check if pages change

- Challenge: URLs are dynamic, many URLs lead to basically same content

Two formats:

- WARC: raw HTTP response (e.g., HTML)

- WET: converted to text (lossy process)

HTML to text:

- Tools to convert HTML to text: [trafilatura](https://trafilatura.readthedocs.io/en/latest/), [resiliparse](https://resiliparse.chatnoir.eu/en/stable/)

- The conversion matters for the resulting LM's downstream task accuracy: 

- [[Li+ 2024]]([object Object])

![images/dclm-wet.png](images/dclm-wet.png)

Let's now look at more specialized sources.

[Wikipedia](https://www.wikipedia.org/): free online encyclopedia

- [Random article](https://en.wikipedia.org/wiki/Special:Random)

- Founded in 2001

- As of May 2026, 67 million articles across 361 language editions (English, Spanish, German, French most common) 

What is the scope?

- Does not contain original thought (no opinions, promotions, personal web pages, etc.) 

- Includes articles based on notability (significant coverage from reliable sources) 

Who writes the content?

- Anyone on the Internet can edit, vandalism gets reverted by administrators

- Small number of Wikipedians contribute majority (e.g., Steven Pruit with 5M edits) 

- Produce [periodic dumps](https://dumps.wikimedia.org/enwiki/) every few weeks (no need to crawl)

Aside: data poisoning attacks 

- [[Carlini+ 2023]]([object Object])

- Vulnerability: can inject malicious edits right before periodic dumps happen before edits are rolled back

- Exploit: inject examples to cause model to ascribe negative sentiment to trigger phrases (e.g., iPhone) 

- [[Wallace+ 2020]]([object Object])

- Takeaway: even high quality sources might contain bad content

Code is helpful for programming tasks, but also for reasoning (folklore).

[GitHub](https://github.com/):

- Live service for hosting code repositories founded in 2008 (acquired by Microsoft in 2018)

- As of May 2026, GitHub has 420M+ repositories (28M public) 

- Each repository includes directory structure + commit history + issues + pull requests + comments, etc.

- Lots of duplicates (e.g., copied code, forks, etc.)

- Allowed to train on any public repository with a permissive license (e.g., MIT, Apache)

Two types of data:

- Repository: download through git protocol (rather than scraping the GitHub website)

- Metadata: GitHub API provides issues, pull requests, comments, etc. (hourly snapshots of event stream on [GitHub Archive](https://info.arxiv.org/help/bulk_data_s3.html))

[Software Heritage](https://www.softwareheritage.org/):

- Non-profit organization founded in 2016 that collects and preserves software

- Focused on the repositories not metadata (issues, comments)

- Aggregates GitHub, GitLab, Bitbucket, PyPI, etc.

- As of May 2026, there are 28.8M source files

[arXiv](https://arxiv.org/):

- Website that allows researchers to share and access papers for free since 1991

- Areas: physics (original), math, CS, statistics, ...

- Has ~3M submissions 

- Submission: metadata, PDF, LaTeX source (optional)

- Light approval process (not peer-review)

- Authors choose (i) all rights reserved or (ii) Creative Commons (e.g., CC-BY)

- Metadata (title, abstract) is under a permissive license (CC0)

- Bulk download from [Amazon S3](https://info.arxiv.org/help/bulk_data_s3.html), no need to crawl

- [[Devlin+ 2018]]([object Object])

The BERT training data consists of:

- Wikipedia

- Books

[Smashwords](https://www.smashwords.com/)

- Founded in 2008, allow anyone to self-publish an e-book

- 2024: 150K authors, 500K books

BooksCorpus 

- [[Zhu+ 2015]]([object Object])

- Self-published books priced at $0, scraped from Smashwords

- 7K books, 985M words

- Has been taken down because violated Smashwords terms-of-service 

- Important: sequences are documents rather than sentences

- Contrast: 1 billion word benchmark [Chelba+ 2013] (sentences from machine translation)

WebText: dataset used to train GPT-2 

- [[Radford+ 2019]]([object Object])

- Contains pages that are outgoing links from Reddit posts with ≥ 3 karma (surrogate for quality)

- 8 million pages, 40GB text

OpenWebTextCorpus: open replication of WebText 

- [[Gokaslan+ 2019]]([object Object])

- Extracted all the URLs from the Reddit submissions dataset

- Used Facebook's fastText classifier to filter out non-English

- Removed near duplicates

CCNet 

- [[Wenzek+ 2019]]([object Object])

- Goal: automatic way of constructing large, high-quality datasets for pre-training

- Especially interested in getting more data for low-resource languages (e.g., Urdu)

Components:

- Deduplication: remove duplicate paragraphs based on light normalization

- Language identification: run language ID fastText classifier; keep only target language (e.g., English)

- Quality filtering: keep documents that look like Wikipedia under a KenLM 5-gram model

Results

- Trained BERT models, CCNet(CommonCrawl) outperforms Wikipedia

- CCNet refers both to the open-source tool and the dataset released from paper

Colossal Clean Crawled corpus (C4) 

- [[Raffel+ 2019]]([object Object])

Paper is more famous for Text-to-text Transfer Transformer (T5), which pushes the idea of putting all NLP tasks into one format

...but a major contribution was the C4 dataset.

Observation: Common Crawl is mostly not useful natural language

Started with one snapshot (April 2019) of Common Crawl (1.4 trillion tokens)

Manual heuristics:

- Keep lines that end in punctuation and have >= 5 words

- Remove page with fewer than 3 sentences

- Removed page that contains any 'bad words' 

- Removed page containing '{' (no code), 'lorem ipsum', 'terms of use', etc.

- Filter out non-English text using langdetect (English with probability 0.99)

End result: 806 GB of text (156 billion tokens)

Analysis of C4 

- [[Dodge+ 2021]]([object Object])

![var/files/image-f87c9ce7952b82131119b325714a5508-https_stanford-cs324_github_io_winter2022_lectures_images_c4-domains_png](var/files/image-f87c9ce7952b82131119b325714a5508-https_stanford-cs324_github_io_winter2022_lectures_images_c4-domains_png)

Bonus: WebText-like dataset

- Filtered to pages from OpenWebText links (links in Reddit posts with ≥ 3 karma)

- Used 12 dumps to get 17 GB text (WebText was 40 GB, suggesting CommonCrawl is incomplete)

- This improved on various NLP benchmarks (GLUE, SQuAD, etc.)

GPT-3 dataset 

- [[Brown+ 2020]]([object Object])

- Common Crawl (processed)

- WebText2 (WebText expanded with more links)

- (Mysterious) Internet-based books corpora (Books1, Books2)

Result: 570 GB (400 billion tokens)

Common Crawl processing:

- Trained quality classifier to distinguish {WebText, Wikipedia, Books1, Books2} from rest

- Fuzzy deduplication of documents (including WebText and benchmarks)

The Pile 

- [[Gao+ 2020]]([object Object])

- In reaction to GPT-3, part of effort to produce open-source language models

- Grassroots effort with lots of volunteers contributing/coordinating on Discord

- Curated 22 high-quality domains

![var/files/image-4eb29ee713b99ea34eb86b995bd32bfd-https_stanford-cs324_github_io_winter2022_lectures_images_the-pile_png](var/files/image-4eb29ee713b99ea34eb86b995bd32bfd-https_stanford-cs324_github_io_winter2022_lectures_images_the-pile_png)

- 825 GB of text (~275B tokens)

- Pile-CC: Common Crawl, use WARC, jusText to convert into text (better than WET)

- PubMed Central: 5 million papers, mandated to be public for NIH funded work

- arXiv: preprint for research papers since 1991 (use latex)

- Enron emails: 500K emails from 150 users from Enron senior management, released during Enron investigation (2002) 

[Project Gutenberg](https://www.gutenberg.org/)

- Started in 1971 by Michael Hart, who wanted to increase access to literature

- 2025: ~75K books, mostly English

- Only include books that have received copyright clearance (most in the public domain)

PG-19: books from Project Gutenberg before 2019 

Books3 [Presser, 2020] 

- 196K books from the shadow library Bibliotik

- Contained books from authors (e.g., Stephen King, Min Jin Lee, Zadie Smith) 

- Has been taken down due to copyright infringement / lawsuits 

- Collection of sites of user-contributed questions and answers

- Started with StackOverflow in 2008, grew to other topics (e.g., math, literature) 

- [[sites]]([object Object])

- Use reputation points and badges to incentivize participation

- [Example](https://ell.stackexchange.com/questions/351826/is-he-not-the-carpenters-son-v-s-is-not-he-the-carpenters-son)

- Q&A format is close to instruction tuning / real application

- Note: there is metadata (users, votes, comments, badges, tags) for filtering

- Data dumps in XML (anonymized, include metadata) 

- [[link]]([object Object])

MassiveText dataset used to train Gopher 

- [[Rae+ 2021]]([object Object])

The Gopher model is subsumed by Chinchilla (also never released), but the description of data is good

Components

- MassiveWeb: more on this later

- C4

- Books: no details

- News: no details

- GitHub: no details

- Wikipedia: no details

MassiveWeb filtering steps

- Keep English, deduplication, train-test overlap

- Quality filtering using manual rules (not classifier) - e.g., 80% words contain at least one alphabetic character

- Use Google SafeSearch for toxicity (not word lists)

Result: 10.5 TB of text (though Gopher only trained on 300B tokens - 12%)

Dataset for LLaMA 

- [[Touvron+ 2023]]([object Object])

- CommonCrawl processed with CCNet, classify *references* of Wikipedia or not

- C4 (more diverse; recall: rule-based filtering)

- GitHub: kept permissive licenses, filtering based on manual rules

- Wikipedia: June-August 2022, 20 languages, manual filtering

- Project Gutenberg and Books3 (from The Pile)

- arXiv: removed comments, inline expanded macros, bibliography

- Stack Exchange: 28 largest websites, sorted answers by score

Result: 1.2T tokens

Reproduced by Together's RedPajama v1 

- [[https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T]]([object Object])

Cerebras's [SlimPajama](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama): 627B subset of RedPajama v1 by deduplication (MinHashLSH)

RefinedWeb 

- [[Penedo+ 2023]]([object Object])

- Point: web data is all you need

- [Examples](https://huggingface.co/datasets/tiiuae/falcon-refinedweb/viewer/default/train)

- trafilatura for HTML→text, extract content (WARC instead of WET files)

- Filtering: Gopher rules, avoid ML-based filtering to avoid biases

- Fuzzy deduplication using MinHash over 5-grams

Released 600B (out of 5T) tokens

FineWeb 

- Started as a replication of RefinedWeb, but improved it

- 95 Common Crawl dumps

- URL filtering, language ID (keep if p(en) > 0.65)

- Filtering: Gopher, C4, more manual rules

- Fuzzy deduplication via MinHash

- Anonymize email and public IP addresses (PII)

Result: 15T tokens

Dolma 

- [[Soldaini+ 2024]]([object Object])

![var/files/image-47601eaf24df2c497082e9c528606b87-https_miro_medium_com_v2_resize_fit_1400_1_-0Qqhvu7JD6Y9JgsfKJdxw_png](var/files/image-47601eaf24df2c497082e9c528606b87-https_miro_medium_com_v2_resize_fit_1400_1_-0Qqhvu7JD6Y9JgsfKJdxw_png)

- Reddit: from the Pushshift project (2005-2023), include submissions and comments separately

- PeS2o: 40M academic papers from Semantic Scholar

- C4, Project Gutenberg, Wikipedia/Wikibooks

Common Crawl processing

- Language identification (fastText classifier), keep English

- Quality filtering (Gopher, C4 rules), avoid model-based filtering

- Toxicity filtering using rules and Jigsaw classifier

- Deduplication using Bloom filters

Result: 3T tokens

DataComp-LM 

- Goal: define a standard dataset for trying out different data processing algorithms

- Processed CommonCrawl to produce DCLM-pool (240T tokens)

- DCLM-baseline: filtered down DCLM-pool using quality classifier

![images/dclm-filter.png](images/dclm-filter.png)

### Model-based filtering

Positive examples (200K):

- [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5): mostly GPT-4 generated instruction data ([examples](https://huggingface.co/datasets/teknium/OpenHermes-2.5/viewer/default/train))

- [ELI5](https://www.reddit.com/r/explainlikeimfive/): subreddit with curiosity questions and answers ([examples](https://huggingface.co/datasets/sentence-transformers/eli5/viewer/pair/train))

Negative examples (200K):

- [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb/viewer/default/train)

Result: 3.8T tokens

Trained a fastText classifier, run it on all of DCLM-pool

This quality classifier outperforms other filtering methods:

![images/dclm-quality.png](images/dclm-quality.png)

Nemotron-CC 

- [[Su+ 2024]]([object Object])

- FineWebEdu and DCLM filter too aggressively (remove 90% of data)

- Need moar tokens (but preserve quality)

- For HTML→text, used jusText (not trafilatura) because it returned more tokens

Classifier ensembling

- Prompt Nemotron-340B-instruct to score FineWeb documents based on educational value, distill into faster model

- DCLM classifier

Synthetic data rephrasing

- For low-quality data, use LM to rephrase

- For high-quality data, use LM to generate tasks (QA pairs, extract key information, etc.)

Result: 6.3T tokens (HQ subset is 1.1T)

For reference, Llama 3 trained on 15T, Qwen3 trained on 36T

![images/nemotron-results.png](images/nemotron-results.png)

The Stack 

- [[Kocetkov+ 2022]]([object Object])

- Took repository names from GitHub Archive (2015-2022)

- git clone'd 137M repositories, 51B files (5B unique!)

- Kept only permissively licensed (MIT, Apache) using go-license-detector

- Remove near-duplicates using minhash and Jaccard similarity

- Result: 3.1 TB of code

Stack v2 

- [[Lozhkov+ 2024]]([object Object])

- Issues, comments, PRs from GitHub Archive

- Repositories from the Software Heritage

- Documentation from crawling websites (e.g., PyPI, npm, devdocs.io)

- Processing: remove binary files, malware, bot activity, deduplication, PII redaction, subsample PRs

- Pair source code (especially low-resource languages like Nim) with shared low-level intermediate language (LLVM)

- Include existing datasets (GSM8K, code contests, StackOverflow, arXiv, Wikipedia, OpenWebMath)

Pull requests:

- Linearize structured object to token sequence

- Add some inline context (e.g., file surrounding diff), subsample

![images/stackv2-pr1.png](images/stackv2-pr1.png)

![images/stackv2-pr2.png](images/stackv2-pr2.png)

Recall:

- Almost all data on the Internet is copyrighted.

- Some of it is permissively licensed.

- Fair use of copyrighted content is not settled.

Key question: can you train a good model using only permissively-licensed data?

CommonPile 

- [[Kandpal+ 2025]]([object Object])

![images/commonpile.png](images/commonpile.png)

- Collected 8TB dataset of permissively licensed data

Subtleties:

- License laundering: redistribute copyrighted work under permissive license (hard to detect)

- Collection licenses (Dolma is ODC-By) doesn't extend to individual

- Synthetic data from LMs trained on unlicensed data is unclear

![images/comma-results.png](images/comma-results.png)

- Can do decently, but tough to compete without more tokens

### Summary

- Key lesson: Data does not fall from the sky. You have to work to get it.

- Live service → raw data → processed data (transformation, filtering, deduplication)

- Data is the key ingredient that differentiates language models

- Legal and ethical issues (e.g., copyright and privacy)

- Much of this pipeline is heuristic, many opportunities to improve!
