# lecture 02

Announcements:

- Join the CS336 slack

- Sign up on Modal with your **Stanford** email

- Read the [AI policy guide](https://docs.google.com/document/d/1SZAlExB1qAc9izHt54gwunNpjKE6wXb8Y7yA_e-baK8/edit?tab=t.0)

- Read the [cluster guide](https://docs.google.com/document/d/1cHE0iKVyXLJ3XpIs2XuXTmZ-HMmPk2hIPeCvy-AydMg/edit?tab=t.otis27tacaef)

Marin 1e23 FLOPs run finished and [matched forecasts](https://x.com/WilliamBarrHeld/status/2039373983632814318)!

![var/files/image-75df5937a21eff96383d1c7c2ff05132-https_pbs_twimg_com_media_HE1P1HmaUAAjLXF_format_jpg_name_medium](var/files/image-75df5937a21eff96383d1c7c2ff05132-https_pbs_twimg_com_media_HE1P1HmaUAAjLXF_format_jpg_name_medium)

Last lecture: overview, tokenization

Today: resource accounting (systems)

Recall: what's the best model one can train given fixed resources (compute, memory)?

In other words: maximize (computational) **efficiency**.

Prerequisite: understand the resources (compute, memory) for a given computation.

**Question**: How long would it take to train a 70B parameter model on 15T tokens on 1024 H100s?

**Question**: What's the largest model that can you can train on 8 H100s using AdamW?

Caveat: activations are not accounted for (depends on batch size and sequence length), so this is an upper bound.

This is a rough back-of-the-envelope calculation.

But it gives you the flavor of napkin math one can quickly do to get a sense of resources.

What knowledge to take away from this lecture:

- Mechanics: straightforward (PyTorch semantics)

- Mindset: resource accounting (remember to do it)

- Intuitions: get a sense of how resources are spent, no ML magic today

Tensors are the basic building block for storing everything:

- data

- parameters

- gradients

- optimizer state

- activations

Example: parameters of the DeepSeek v3.2 model 

- [[DeepSeek-AI+ 2025]]([object Object])

- [[DeepSeek v3.2 model on Hugging Face]]([object Object])

Each tensor has a rank, which is the number of dimensions.

In Transformers, will see tensors of rank 4:

Elements of tensors are generally floating point numbers.

## fp32

- [[Wikipedia]]([object Object])

![images/fp32.png](images/fp32.png)

The fp32 data type (also known as float32 or single precision) is the default.

Traditionally, in scientific computing, fp32 is the baseline; you could use double precision (fp64) in some cases.

In deep learning, you can be a lot sloppier.

Let's examine memory usage of these tensors.

Memory is determined by the (i) number of values and (ii) data type of each value.

One matrix in the feedforward layer of GPT-3:

## fp16

![images/fp16.png](images/fp16.png)

The fp16 data type (also known as float16 or half precision) cuts down the memory.

However, the dynamic range (especially for small numbers) isn't great.

If this happens when you train, you can get instability.

## bf16

![images/bf16.png](images/bf16.png)

Google Brain developed brain floating point (bf16) in 2018 to address this issue.

bf16 uses the same memory as fp16 but has the same dynamic range as fp32!

The only catch is that the resolution is worse, but this matters less for deep learning.

## Mixed precision

Implications on training:

- Training with fp32 works, but requires lots of memory.

- Training with fp16 and even bf16 is risky, and you can get instability.

Solution: mixed precision training 

- [[Micikevicius+ 2017]]([object Object])

- Use bf16 for parameters, activations, and gradients

- Use fp32 for optimizer states

Pytorch has an automatic mixed precision (AMP) library. 

- [[docs]]([object Object])

Tries to cast things into bf16 when safe (matmuls, not exp).

## fp8

In 2022, fp8 was standardized, motivated by machine learning workloads [primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).

![var/files/image-df6d7649a3bdb77cfdc38092d8387a99-https_docs_nvidia_com_deeplearning_transformer-engine_user-guide__images_fp8_formats_png](var/files/image-df6d7649a3bdb77cfdc38092d8387a99-https_docs_nvidia_com_deeplearning_transformer-engine_user-guide__images_fp8_formats_png)

H100s support two variants of FP8: E4M3 (range [-448, 448]) and E5M2 ([-57344, 57344]).

Reference: 

- [[Micikevicius+ 2022]]([object Object])

## fp4

In 2025, NVIDIA developed [nvfp4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

Only 4 bits per value!

Values: -6, -4, -3, -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6

Use a separate scale factor per block, so actually get more dynamic range (but just can't vary freely from neighbors).

Nemotron 3 Super was trained in NVFP4 

- [[Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning]]([object Object])

Some of this is done in NVIDIA libraries outside of user control.

By default, tensors are stored in CPU memory.

However, what about GPUs?

![images/cpu-gpu.png](images/cpu-gpu.png)

In order to take advantage of the massive parallelism of GPUs, we need to move them to GPU memory.

Or create the tensor directly on the GPU:

Traditional PyTorch code:

Easy to mess up the dimensions (what is -2, -1?)...

Einops is a library for manipulating tensors where dimensions are named.

It is inspired by Einstein summation notation (Einstein, 1916).

- [[Einops tutorial]]([object Object])

Einsum is generalized matrix multiplication with good bookkeeping.

Let's try a more complex example...

Dimensions that are not named in the output are summed over.

You can reduce a single tensor via some operation (e.g., sum, mean, max, min).

Sometimes, a dimension represents two dimensions

...and you want to operate on one of them.

...where `total_hidden` is a flattened representation of `heads * hidden1`

Having gone through all the operations, let us examine their computational cost.

A floating-point operation (FLOP) is a basic operation like addition (x + y) or multiplication (x y).

Two terribly confusing acronyms (pronounced the same!):

- FLOPs: floating-point operations (measure of computation done)

- FLOP/s: floating-point operations per second (also written as FLOPS), which is used to measure the speed of hardware.

## Intuitions

Training GPT-3 (2020) took 3.14e23 FLOPs. 

- [[article]]([object Object])

Training GPT-4 (2023) is speculated to take 2e25 FLOPs. 

H100 has a peak performance of 1979 teraFLOP/s with sparsity, 50% without 

- [[spec]]([object Object])

8 H100s for 2 weeks:

## Linear model

How many FLOPs is this matmul?

We have one multiplication (x[i][j] * w[j][k]) and one addition per (i, j, k) triple.

We can also time this operation to see how long it takes.

The actual FLOP/s of this operation:

Each GPU has a specification sheet that provides the peak performance.

- Example: 

- [[H100 spec]]([object Object])

Note that the FLOP/s depends heavily on the data type!

## Model FLOPs utilization (MFU)

Definition: MFU = (actual FLOP/s) / (promised FLOP/s) [ignore communication/overhead]

Usually, MFU of ≥ 0.5 is quite good!

But why is MFU not closer to 1?

To answer this question, we need to look more closely at how computations are done on GPUs...

![images/compute-memory.png](images/compute-memory.png)

How to compute a thing:

1. Send inputs from memory to accelerator

2. Perform computation

3. Send outputs from accelerator to memory

How long does this take?

Depends on two things:

1. Accelerator speed (FLOP/s)

2. Memory bandwidth (bytes/s)

Assume we can overlap communication and computation perfectly.

What is the bottleneck?

- Memory-bound: communication time > computation time

- Compute-bound: computation time > communication time

In this case, ReLU is memory-bound.

Alternative way to see this:

Accelerator intensity: how much work can the accelerator do per byte transferred?

Arithmetic intensity: how much actual work per byte for this workload?

- Memory-bound: arithmetic intensity < accelerator intensity

- Compute-bound: arithmetic intensity > accelerator intensity

In general, we'll find ourselves memory bound.

Can we increase arithmetic intensity?

Note that GeLU does more work than ReLU per byte moved, so it has higher arithmetic intensity.

But still memory-bound!

In other words, ReLU is not faster than GeLU (when doing things in an isolated way).

Memory-bound!

Finally, compute-bound!

As long as we have large matrices, we're compute-bound (saturating the accelerator).

Training Transformers involves big matrix multiplications.

Matrix-vector product is what happens during inference, which is why inference is memory-bound.

Note: arithmetic/accelerator intensity also depends on the precision (bf16 versus fp32).

We can visualize the relationship between arithmetic intensity and performance using roofline plots.

![var/files/image-42d32b9c87939fe9a4b0a268d6d02ea7-https_jax-ml_github_io_scaling-book_assets_img_roofline-improved-1400_webp](var/files/image-42d32b9c87939fe9a4b0a268d6d02ea7-https_jax-ml_github_io_scaling-book_assets_img_roofline-improved-1400_webp)

- Each slice on the x-axis is a particular computation (with some arithmetic intensity)

- Each piecewise linear function corresponds to a particular hardware

- Kink is the accelerator intensity (transition from memory-bound to compute-bound)

We can now relate this back to MFU:

MFU = min(1, arithmetic-intensity / accelerator-intensity)

- [[reference]]([object Object])

![images/deep-network.png](images/deep-network.png)

Consider a deep network with L layers and D-dimensional inputs, activations, and outputs.

So far, we've constructed tensors and passed them through operations (forward).

Now, we're going to compute the gradient (backward).

As a simple example, let's consider the simple linear model:

y = 0.5 (x * w - 5)^2

Forward pass: compute loss

Backward pass: compute gradients

Let us count the FLOPs for computing gradients.

Define a simplified model (2-layer linear network):

## Zoom in on one layer

Let's focus on the second layer (h2 = h1 @ w2)

**Forward pass**: Recall the number of forward FLOPs: 

**Backward pass**: How many FLOPs is running the backward pass?

We need to compute:

- h1.grad = d loss / d h1

- w2.grad = d loss / d w2

Note that the backward pass is 2x more expensive than the forward pass.

## Consider all layers

This was just for w2, need to apply it to all parameters in the network.

Putting it together:

- Forward pass: 2 (# data points) (# parameters) FLOPs

- Backward pass: 4 (# data points) (# parameters) FLOPs

- Total: 6 (# data points) (# parameters) FLOPs

This is for multilayer perceptrons (MLPs)

...but it turns out to be a good approximation for Transformers for short context lengths as well.

Recall our deep network.

Let's define the AdaGrad optimizer

- momentum = SGD + exponential averaging of grad

- AdaGrad = SGD + averaging by grad^2

- RMSProp = AdaGrad but with exponential averaging of grad^2

- Adam = RMSProp + momentum

AdaGrad 

- [[Duchi+ 2011]]([object Object])

## Memory

It is customary to use fp32 for stability (accumulating averages over powers over many steps).

Optimizer state memory:

- AdaGrad: 4 bytes/parameter for storing second moments

- Adam: 8 bytes/parameter for storing first and second moments

## Compute (for one training step)

## Transformers

The accounting for a Transformer is more complicated, but the same idea.

Assignment 1 will ask you to do that.

Blog post describing memory usage for Transformer training 

Blog post describing FLOPs for a Transformer: 

Large batch sizes: improve training stability

However, activation memory scales with batch size, so might run out.

Gradient accumulation:

- Compute gradient on micro batches

- Accumulate the gradients (don't zero it out)

- Every batch_size / micro_batch_size steps, update the parameters and zero out the gradients

For training, we need to store the activations of all layers

For inference, we don't compute gradients, so we only need to store the current layer's activations.

The memory usage is

Can we reduce this?

Activation checkpointing = gradient checkpointing = rematerialization

Key idea:

- Forward pass: keep only activations at subset of layers

- Backward pass: recompute the missing activations from the last checkpoint

Philosophy: tradeoff memory for compute

Can we reduce this even more, especially for deep networks (large L)?

How frequently to checkpoint?

- If store each layer's activations, then activation memory is O(L) and no recomputation.

- If store no activations, then activation memory is O(1) and compute is O(L^2) (recompute from the start for each layer).

- If store every sqrt(L) layers, then activation memory is O(sqrt(L)) and O(L) recomputation.

Summary:

- Everything is operations on tensors (parameters, gradients, activations, optimizer states, data)

- einops: better way to think about tensor operations

- 6 (# data points) (# parameters) FLOPs per training step

- Arithmetic intensity / roofline analysis: compute-bound or memory-bound?

- Matrix multiplications are compute-bound, elementwise operations are memory-bound

- Gradient accumulation, activation checkpointing: reduce memory to use bigger batch sizes
