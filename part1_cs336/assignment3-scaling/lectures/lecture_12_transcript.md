# lecture 12

## Lecture 12: evaluation

- So far: we've covered everything for training an LM (architecture, training, systems, scaling).

- Missing piece: what **data** do you train on?

- Data shapes model behavior (code? multilingual? DNA?).

- Before talking about data, need to talk about what behavior we want from a model.

**Evaluation**: given a model, how "**good**" is it?

Evaluation might appear to be a mechanical process:

1. Define some prompts

2. Send prompts to a model and get back responses

3. Compute accuracy

But actually, evaluation is a deep and important topic...

...which shapes the development of AI.

**Core challenge**: <font color="red">abstract construct</font> → <font color="blue">concrete metric</font>

Maybe a model is good if it does well on benchmarks...

- [[Artificial Analysis]]([object Object])

![images/artificial-analysis.png](images/artificial-analysis.png)

Maybe a model is good if it does well on benchmarks and is cheap to run...

![images/artificial-analysis-cost.png](images/artificial-analysis-cost.png)

Maybe a model is good if people prefer its responses...

- [[Arena AI (formerly Chatbot Arena)]]([object Object])

![images/lmarena-leaderboard.png](images/lmarena-leaderboard.png)

Maybe a model is good if people simply choose to use (and pay for) it...

- [[OpenRouter]]([object Object])

![images/openrouter.png](images/openrouter.png)

- Recall: that a language model is a probability distribution **p(x)** over sequences of tokens.

- Perplexity (1/p(D))^(1/|D|) measures whether p assigns high probability to some dataset D.

- In pre-training, you minimize perplexity on the training set.

- The obvious thing is to measure perplexity on the test set.

- This is what people did traditionally in language modeling research.

Standard datasets:

- Penn Treebank (WSJ)

- WikiText-103 (Wikipedia)

- One Billion Word Benchmark (from machine translation WMT11 - EuroParl, UN, news)

Classic paradigm: in-distribution evaluation: train on train split and evaluate on test split of some dataset.

Pure CNNs+LSTMs on the One Billion Word Benchmark (perplexity 51.3 → 30.0) 

- [[Jozefowicz+ 2016]]([object Object])

GPT-2:

- Trained on WebText (40GB text, websites linked from Reddit)

- Zero-shot on standard datasets (**out-of-distribution** evaluation)

![images/gpt2-perplexity.png](images/gpt2-perplexity.png)

- Works better on small datasets (PTB) where transfer is helpful, but not larger datasets (1BW)

Perplexity is all you need (more faith than science):

- True distribution is t, model is p.

- Best possible perplexity is H(t) obtained iff p = t.

- If p = t, then solve all the tasks: p(solution | problem)

- So by pushing down on perplexity, we will eventually "reach AGI".

Perplexity is maybe more than you need:

- Example: *Stanford was founded in 1885*

- Perplexity penalizes prediction on all tokens, some (e.g., *founded*) of which might not be relevant

- Solution: measure conditional perplexity p(response | prompt)^(1/|response|)

Some benchmarks are perplexity in disguise:

- Cloze tasks (fill in the blank): LAMBADA 

- [[Paperno+ 2016]]([object Object])

![images/lambada.png](images/lambada.png)

- Multiple choice sentence completion: HellaSwag 

- [[Zellers+ 2019]]([object Object])

![images/hellaswag.png](images/hellaswag.png)

**Warning** (if you're running a perplexity leaderboard):

- People submit `LM` and you compute `log_prob = LM(test_data)`

- You need to trust that the probabilities are valid (sum to 1)

- For downstream tasks, `response = LM(prompt)` and compute accuracy on `response`

Summary:

- Perplexity is still used heavily in language model development (smooth scaling laws)

- Still need benchmarks that capture real-world situations (for the non-believers)...

Exams are a useful way to test language models (as with humans):

- Have control over the subject and difficulty

- Design to have unambiguous correct answer, easy to grade

**Massive Multitask Language Understanding (MMLU)** 

- [[Hendrycks+ 2020]]([object Object])

- 57 subjects (e.g., math, US history, law, morality), multiple-choice

- "collected by graduate and undergraduate students from freely available sources online"

- Despite the name, MMLU is really about testing knowledge, not language understanding

- Evaluated on GPT-3 using few-shot prompting

![images/mmlu.png](images/mmlu.png)

- [[https://llm-stats.com/benchmarks/mmlu]]([object Object])

- [[HELM MMLU for visualizing predictions]]([object Object])

**MMLU-Pro** 

- [[Wang+ 2024]]([object Object])

- Removed noisy/trivial questions from MMLU

- Expanded 4 choices to 10 choices

- Evaluated using chain of thought (gives model more of a chance)

- Accuracy of models drop by 16% to 33% (not as saturated)

![images/mmlu-pro.png](images/mmlu-pro.png)

- [[https://llm-stats.com/benchmarks/mmlu-pro]]([object Object])

- [[HELM MMLU-Pro for visualizing predictions]]([object Object])

**Graduate-Level Google-Proof Q&A (GPQA)** 

- [[Rein+ 2023]]([object Object])

- Questions written by 61 PhD contractors from Upwork

![images/gpqa.png](images/gpqa.png)

- PhD experts achieve 65% accuracy

- Non-experts achieve 34% over 30 minutes with access to Google

- GPT-4 achieves 39%

- [[https://llm-stats.com/benchmarks/gpqa]]([object Object])

- [[HELM GPQA for visualizing predictions]]([object Object])

**Humanity's Last Exam (HLE)** 

- [[Phan+ 2025]]([object Object])

- 2500 questions: multimodal, many subjects, multiple-choice + short-answer

![images/hle-examples.png](images/hle-examples.png)

- Awarded $500K prize pool + co-authorship to question creators

- Filtered by frontier LLMs, multiple stages of review

![images/hle-pipeline.png](images/hle-pipeline.png)

![images/hle-results.png](images/hle-results.png)

- [[https://llm-stats.com/benchmarks/hle]]([object Object])

- Trend towards harder questions as models improve and saturate existing benchmarks

- Multiple-choice format can be as difficult as one wants

- Does not capture real usage (open-ended, doesn't necessarily exist correct answer)

- So far, we've been evaluating on well-defined multiple-choice tasks.

- Most people don't ask multiple-choice exam questions to their AI assistant.

Example:

Prompt: *I would like to make a beet salad with goat cheese. What kind of herbs would work well and what would not work well?*

Response: *Here’s a breakdown of herbs that work well (and some that don’t) in a beet + goat cheese salad, based on how their flavors interact with the sweet-earthiness of beets and the tangy creaminess of goat cheese...

**Challenge**: how to evaluate an open-ended response?

**Chatbot Arena** 

- [[Chiang+ 2024]]([object Object])

Data collection:

- Random person from the Internet types in prompt

- They get response from two random (anonymized) models

- They rate which one is better

![images/arena-beets.png](images/arena-beets.png)

Compute ELO rankings based on pairwise comparisons:

- Define model: p(A wins against B) = 1 / (1 + 10^((ELO_B - ELO_A)/400))

- Fit this model to maximize probability of pairwise comparisons

Properties:

- Real-world prompts (free for users, incentives to actually use it)

- But who are these people? biases? spammers?

- Binary preference but conflates style and correctness

- How does the human even assess correctness?  Prone to sycophancy?

- Feature: don't need to feed same prompts to all models (important because human is rating)

- Dynamic: incorporates new prompts and models over time

**AlpacaEval** (2023)

- [[leaderboard]]([object Object])

- 805 instructions from various sources

- Metric: win rate against baseline model (GPT-4 preview) as judged by GPT-4 preview (potential bias?)

- Problem: LLM judges favor longer responses, resulted in leaderboard gaming

- Alpaca Eval 2.0 used regression to debias the metric 

- [[Dubois+ 2024]]([object Object])

- How do we evaluate the metric?

- Correlation with Chatbot Arena (humans) is high:

![var/files/image-434a1510a7ed21d5355814149a9490c4-https_github_com_tatsu-lab_alpaca_eval_raw_main_figures_chat_correlations_no_ae_png](var/files/image-434a1510a7ed21d5355814149a9490c4-https_github_com_tatsu-lab_alpaca_eval_raw_main_figures_chat_correlations_no_ae_png)

![images/alpacaeval-leaderboard.png](images/alpacaeval-leaderboard.png)

**WildBench** 

- [[Lin+ 2024]]([object Object])

- Sourced 1024 examples from 1M human-chatbot conversations

- Uses GPT-4 turbo as a judge with a checklist (like CoT for judging) + GPT-4 as a judge

- Well-correlated with Chatbot Arena (seems to be the de facto sanity check)

![images/wildbench.png](images/wildbench.png)

- [[HELM WildBench for visualizing predictions]]([object Object])

- Challenge: how to evaluate open-ended responses?

- Pairwise comparisons between similar responses provide higher signal

- Beware of biases (both from humans and LLM judges)

- Checklist/rubric improves reliability (regardless of human or LLM judge)

Previously: evaluate what LMs say (chat)

Now: evaluate what LMs do (agents)

Agent = language model + agent scaffold (logic for deciding how to use the LM)

Consider tasks that require tool use (e.g., running code) and iterating over a period of time

**SWEBench** 

- [[Jimenez+ 2023]]([object Object])

- 2294 tasks across 12 Python repositories

- Given codebase + issue description, submit a PR

- Evaluation metric: unit tests

![images/swebench.png](images/swebench.png)

- [[https://llm-stats.com/benchmarks/swe-bench-verified]]([object Object])

**TerminalBench** 

- [[Merrill+ 2026]]([object Object])

- [[website]]([object Object])

![images/terminal-bench.png](images/terminal-bench.png)

- Computer terminal environments: simple and universal

- 229 tasks crowdsourced from 93 contributors, 89 tasks constitute Terminal-Bench 2.0

![images/terminal-bench-human-time.png](images/terminal-bench-human-time.png)

![images/terminal-bench-results.png](images/terminal-bench-results.png)

- [[https://llm-stats.com/benchmarks/terminal-bench]]([object Object])

**CyBench** 

- [[Zhang+ 2024]]([object Object])

![images/cybench.png](images/cybench.png)

- 40 Capture the Flag (CTF) tasks

- Use first-solve time as a measure of difficulty

![images/cybench-agent.png](images/cybench-agent.png)

![images/cybench-results.png](images/cybench-results.png)

- [[https://llm-stats.com/benchmarks/cybench]]([object Object])

**MLEBench** 

- [[Chan+ 2024]]([object Object])

- 75 Kaggle competitions (require training models, processing data, etc.)

![images/mlebench.png](images/mlebench.png)

![images/mlebench-results.png](images/mlebench-results.png)

Agent scaffolds 

- [[post]]([object Object])

![var/files/image-155d1eb10710df090449bf401822dd7e-https_www_philschmid_de_static_blog_agents-2_0-deep-agents_overview_png](var/files/image-155d1eb10710df090449bf401822dd7e-https_www_philschmid_de_static_blog_agents-2_0-deep-agents_overview_png)

- Explicit planning: keep a todo list that gets checked off

- Hierarchical delegation: agents calling other sub-agents (clean context)

- Persistent memory: read/write files

- Extreme context engineering: explicit more instructions on process

- Agents dramatically enhance the capability surface of language models

- Agent scaffolds are very important

- Evaluating agents = evaluating agent scaffold + language model

- All of the tasks so far require linguistic and world knowledge.

- Can we isolate **reasoning** from knowledge?

- Arguably, reasoning captures a more pure form of intelligence (isn't just about memorizing facts).

**ARC-AGI** 

- 100\% solvable by humans, but challenging for AI

- Each task is unique, so memorization doesn't help.

- ARC-AGI-1 (2019): first iteration

![var/files/image-d1a33e9159cfdb77197551bbbecc6a76-https_arcprize_org_media_images_arc-task-grids_jpg](var/files/image-d1a33e9159cfdb77197551bbbecc6a76-https_arcprize_org_media_images_arc-task-grids_jpg)

- ARC-AGI-2 (March 2025): more multi-step reasoning

![var/files/image-a0338c9fb72d1163cfc8ac66fea4e4ed-https_arcprize_org_media_images_blog_arc-agi-2-unsolved-1_png](var/files/image-a0338c9fb72d1163cfc8ac66fea4e4ed-https_arcprize_org_media_images_blog_arc-agi-2-unsolved-1_png)

![images/arc-agi-results.png](images/arc-agi-results.png)

- Pretrained language models didn't move the needle

- Reasoning models (o1, o3) started making things take off

- ARC-AGI-3 (March 2026): interactive environments 

![images/arc-agi-3.png](images/arc-agi-3.png)

![images/arc-agi-3-results.png](images/arc-agi-3-results.png)

- Goal is to disentangle reasoning from knowledge (difficult to do!)

- Constrained to human reasoning (not superhuman reasoning)

- Clearly exposes gaps in current models

![var/files/image-a375cd28c372458baf4135c081a1ce8b-https_www_team-bhp_com_forum_attachments_road-safety_2173645d1625144681-will-crash-test-rating-change-if-higher-variant-chosen-images-30_jpeg](var/files/image-a375cd28c372458baf4135c081a1ce8b-https_www_team-bhp_com_forum_attachments_road-safety_2173645d1625144681-will-crash-test-rating-change-if-higher-variant-chosen-images-30_jpeg)

What does safety mean for AI?

**HarmBench** 

- [[Mazeika+ 2024]]([object Object])

- Based on 510 harmful behaviors that violate laws or norms

- [[HarmBench on HELM]]([object Object])

- [[Example of safety failure]]([object Object])

**AIR-Bench** 

- [[Zeng+ 2024]]([object Object])

- Based on regulatory frameworks and company policies

- Taxonomized into 314 risk categories, 5694 prompts

![var/files/image-5993188f3fa9dc78b85f9866fcee27ac-https_crfm_stanford_edu_helm_assets_air-overview-DpBbyagA_png](var/files/image-5993188f3fa9dc78b85f9866fcee27ac-https_crfm_stanford_edu_helm_assets_air-overview-DpBbyagA_png)

- [[HELM AIR-Bench]]([object Object])

Jailbreaking:

- Language models are trained to refuse harmful instructions

- Greedy Coordinate Gradient (GCG) automatically optimizes prompts to bypass safety 

- [[Zou+ 2023]]([object Object])

- Transfers from open-weight models (Llama) to closed models (GPT-4)

![images/gcg-examples.png](images/gcg-examples.png)

What is safety?

- Many aspects of safety are strongly contextual (politics, law, social norms - which vary across countries)

- Many risks are quite varied (hallucinations, sycophancy, abetting crimes, inequality, losing critical thinking)

**Dual-use**: capable cybersecurity agents (Mythos) can be used to hack into a system or to do penetration testing

**Ecological validity**: how well does an evaluation capture real-world use?

- Exam benchmarks (e.g., GPQA) are far away from real-world use.

- Chatbot Arena prompts are from real people, but distribution is uncontrolled.

**GDPVal** (OpenAI) 

- [[Patwardhan+ 2025]]([object Object])

- 44 occupations from top 9 sectors according to US GDP

- Tasks come from professionals with ~14 years of experience

![images/gdpval.png](images/gdpval.png)

**MedHELM** 

- [[Bedi+ 2025]]([object Object])

- Previous medical benchmarks were based on standardized exams

- 121 clinical tasks sourced from 29 clinicians, mixture of private and public datasets

![var/files/image-93ff2615b50418e8fd4dd6f6435bdff1-https_crfm_stanford_edu_helm_assets_medhelm-overview-CND0EIsy_png](var/files/image-93ff2615b50418e8fd4dd6f6435bdff1-https_crfm_stanford_edu_helm_assets_medhelm-overview-CND0EIsy_png)

- [[MedHELM]]([object Object])

**Clio** (Anthropic) 

- [[Tamkin+ 2024]]([object Object])

- Use language models to analyze real user data

- Share general patterns of what people are asking

![images/clio-table4.png](images/clio-table4.png)

Unfortunately, realism and privacy are sometimes at odds with each other.

How do we know our evaluations are valid?

### Train-test overlap

- Machine learning 101: don't train on your test set

- Pre-foundation models (ImageNet, SQuAD): well-defined train-test splits

- Today: train on the Internet and don't tell people about your data

Route 1: try to infer train-test overlap from model

- Exploit exchangeability of data points 

- [[Oren+ 2023]]([object Object])

![images/contamination-exchangeability.png](images/contamination-exchangeability.png)

Route 2: encourage reporting norms (e.g., people report confidence intervals)

- Model providers should report train-test overlap 

Route 3: use fresh evals

- LiveCodeBench, UncheatableEval: scrape new webpages

- Timestamps aren't always safe due to copying either

Route 4: use private evals

- Companies use internal code bases that aren't on the Internet

- Use your personal writings

- Easiest for perplexity

### Dataset quality

- Fixed up SWE-Bench to produce SWE-Bench Verified 

- Create Platinum versions of benchmarks 

- [[Vendrow+ 2025]]([object Object])

![var/files/image-a1149b095a48ea306dcdea342363230c-https_pbs_twimg_com_media_GjICXQlWkAAYnDS_format_jpg_name_4096x4096](var/files/image-a1149b095a48ea306dcdea342363230c-https_pbs_twimg_com_media_GjICXQlWkAAYnDS_format_jpg_name_4096x4096)

![var/files/image-306ae3f862cee7c7842c0b29af3a2f5c-https_pbs_twimg_com_media_GjICcGQXYAAM4o1_format_jpg_name_4096x4096](var/files/image-306ae3f862cee7c7842c0b29af3a2f5c-https_pbs_twimg_com_media_GjICcGQXYAAM4o1_format_jpg_name_4096x4096)

- Problems with agentic benchmarks: insufficient test cases, trivial agent can solve task 

- [[Zhu+ 2025]]([object Object])

- Docent: use LLM to inspect agent traces to detect problems 

### What's the point of evaluation?

There is no one true evaluation; it depends on what question you're trying to answer.

1. User or company wants to make a purchase decision (model A or model B) for their use case (e.g., customer service chatbots).

2. Researchers want to measure the raw capabilities of a model (e.g., intelligence).

3. We want to understand the benefits + harms of a model (for business and policy reasons).

4. Model developers want to get feedback to improve the model.

### What are we evaluating?

- Pre-foundation models, we evaluated **methods** (standardized train-test splits).

- Today, we're (mostly) evaluating **models/systems** (anything goes).

There are some exceptions...

- nanogpt speedrun: fixed data, compute time to get to a particular validation loss

![images/karpathy-nanogpt-speedrun.png](images/karpathy-nanogpt-speedrun.png)

Evaluating methods encourage algorithmic innovation from researchers.

Evaluating models/systems is useful for downstream users.

Either way, we need to define the rules of the game!

Takeaways:

- There is no one true evaluation; choose the evaluation depending on what you're trying to measure.

- Clearly state the rules of the game (methods versus models versus agents).

- Considerations: difficulty, realism, validity.
