# lecture 02 recording

Announcements:

- Join the CS336 slack

- Sign up on Modal with your **Stanford** email

- Read the [AI policy guide](https://docs.google.com/document/d/1SZAlExB1qAc9izHt54gwunNpjKE6wXb8Y7yA_e-baK8/edit?tab=t.0)

- Read the [cluster guide](https://docs.google.com/document/d/1cHE0iKVyXLJ3XpIs2XuXTmZ-HMmPk2hIPeCvy-AydMg/edit?tab=t.otis27tacaef)

Marin 1e23 FLOPs run finished and [matched forecasts](https://x.com/WilliamBarrHeld/status/2039373983632814318)

![var/files/image-75df5937a21eff96383d1c7c2ff05132-https_pbs_twimg_com_media_HE1P1HmaUAAjLXF_format_jpg_name_medium](var/files/image-75df5937a21eff96383d1c7c2ff05132-https_pbs_twimg_com_media_HE1P1HmaUAAjLXF_format_jpg_name_medium)

Last lecture: overview, tokenization

Today: resource accounting (systems)

Recall: what's the best model one can train given fixed resources (compute, memory)?

In other words: maximize (computational) **efficiency**.

Precursor: We need to understand the resources (compute, memory) for a given computation?

**Question**: How long would it take to train a 70B parameter model on 15T tokens on 1024 B100s?

**Question**: What's the largest model that can you can train on 8 H100s using AdamW (naively)?

Caveat: activations are not accounted for (depends on batch size and sequence length).

This is a rough back-of-the-envelope calculation.

What knowledge to take away:

- Mechanics: straightforward (PyTorch semantics)

- Mindset: resource accounting (remember to do it)

- Intuitions: get a sense of how resources are spent, no ML magic today

Tensors are the basic building block for storing everything:

- parameters, gradients, optimizer state, data, activations.

Example: parameters of the DeepSeek v3.2 model 

- [[DeepSeek-AI+ 2025]]([object Object])

- [[DeepSeek v3.2 model on Hugging Face]]([object Object])

Almost everything (parameters, gradients, activations, optimizer states) are stored as floating point numbers.

## float32

- [[Wikipedia]]([object Object])

![images/fp32.png](images/fp32.png)

The float32 data type (also known as fp32 or single precision) is the default.

Traditionally, in scientific computing, float32 is the baseline; you could use double precision (float64) in some cases.

In deep learning, you can be a lot sloppier.

Let's examine memory usage of these tensors.

Memory is determined by the (i) number of values and (ii) data type of each value.

One matrix in the feedforward layer of GPT-3:

## float16

![images/fp16.png](images/fp16.png)

The float16 data type (also known as fp16 or half precision) cuts down the memory.

However, the dynamic range (especially for small numbers) isn't great.

If this happens when you train, you can get instability.

## bfloat16

![images/bf16.png](images/bf16.png)

Google Brain developed bfloat (brain floating point) in 2018 to address this issue.

bfloat16 uses the same memory as float16 but has the same dynamic range as float32!

The only catch is that the resolution is worse, but this matters less for deep learning.

Implications on training:

- Training with float32 works, but requires lots of memory.

- Training with fp8, float16 and even bfloat16 is risky, and you can get instability.

Mixed precision training 

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

Use a separate scale factor per block

Nemotron 3 Super was trained in NVFP4 

- [[NVIDIA+ 2025]]([object Object])

Some of this is done in NVIDIA libraries outside of user control.

By default, tensors are stored in CPU memory.

However, in order to take advantage of the massive parallelism of GPUs, we need to move them to GPU memory.

![images/cpu-gpu.png](images/cpu-gpu.png)

Let's first see if we have any GPUs.

Easy to mess up the dimensions (what is -2, -1?)...

Traditional PyTorch code:

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

As motivation, suppose you have a linear model.

- We have n points

- Each point is d-dimsional

- The linear model maps each d-dimensional vector to a k outputs

We have one multiplication (x[i][j] * w[j][k]) and one addition per (i, j, k) triple.

## FLOPs of other operations

- Elementwise operation on a m x n matrix requires O(m n) FLOPs.

- Addition of two m x n matrices requires m n FLOPs.

In general, no other operation that you'd encounter in deep learning is as expensive as matrix multiplication for large enough matrices.

Interpretation:

- B is the number of data points

- (D K) is the number of parameters

- FLOPs for forward pass is 2 (# tokens) (# parameters)

It turns out this generalizes to Transformers (to a first-order approximation).

How do our FLOPs calculations translate to wall-clock time (seconds)?

Let us time it!

Each GPU has a specification sheet that reports the peak performance.

- Example: 

- [[H100 spec]]([object Object])

Note that the FLOP/s depends heavily on the data type!

## Model FLOPs utilization (MFU)

Definition: (actual FLOP/s) / (promised FLOP/s) [ignore communication/overhead]

Usually, MFU of ≥ 0.5 is quite good (and will be higher if matmuls dominate)

## Summary

- Matrix multiplications dominate: (2 m n p) FLOPs

- FLOP/s depends on hardware (B200 >> H100) and data type (bfloat16 >> float32)

- Model FLOPs utilization (MFU): (actual FLOP/s) / (promised FLOP/s)

![images/compute-memory.png](images/compute-memory.png)

How to compute a thing:

1. Send inputs from memory to accelerator

2. Perform computation

3. Send outputs from accelerator to memory

How long does this take?

Depends on two things:

1. Accelerator speed (FLOP/s)

1. Memory bandwidth (bytes/s)

Assume we can overlap communication and computation perfectly.

What is the bottleneck?

- Memory-bound: communication time > computation time

- Compute-bound: computation time > communication time

In this case, ReLU is memory-bound.

Alternative way to see this:

Accelerator intensity: how much work can the accelerator do per byte transferred?

Arithmetic intensity: how much actual work per byte for this workload?

What is the bottleneck

- Memory-bound: arithmetic intensity < accelerator intensity

- Compute-bound: arithmetic intensity > accelerator intensity

In general, we'll find ourselves memory bound, so higher arithmetic intensity is good!

Note that GeLU does more work than ReLU per byte moved, so it has higher arithmetic intensity.

But still memory-bound!

In other words, ReLU is not faster than GeLU (when doing things in an isolated way).

Memory-bound!

Finally, compute-bound!

As long as we have large matrices, we're compute-bound (saturating the accelerator).

Training Transformers involves big matrix multiplications

Matrix-vector product is what happens during inference, which is why inference is memory-bound.

Note: arithmetic/accelerator intensity depends on the precision (bf16 versus fp32)

We can visualize the relationship between arithmetic intensity and performance using roofline plots.

![var/files/image-42d32b9c87939fe9a4b0a268d6d02ea7-https_jax-ml_github_io_scaling-book_assets_img_roofline-improved-1400_webp](var/files/image-42d32b9c87939fe9a4b0a268d6d02ea7-https_jax-ml_github_io_scaling-book_assets_img_roofline-improved-1400_webp)

- Each slice on the x-axis is a particular computation (with some arithmetic intensity)

- Each piecewise linear function corresponds to a particular hardware

- Kink is the accelerator intensity (transition from memory-bound to compute-bound)

- [[https://jax-ml.github.io/scaling-book/roofline/]]([object Object])

![images/deep-network.png](images/deep-network.png)

Consider a deep network with L layers and D-dimensional inputs, activations, and outputs.

So far, we've constructed tensors (which correspond to either parameters or data) and passed them through operations (forward).

Now, we're going to compute the gradient (backward).

As a simple example, let's consider the simple linear model:

y = 0.5 (x * w - 5)^2

Forward pass: compute loss

Backward pass: compute gradients

Let us count the FLOPs for computing gradients.

Define a simplified model (2-layer linear network):

## Zoom in on one layer

Let's focus on the second layer h1 --w2--> h2.

**Forward pass**: Recall the number of forward FLOPs (bf16): 

**Backward pass**: How many FLOPs is running the backward pass?

We need to compute:

- h1.grad = d loss / d h1

- w2.grad = d loss / d w2

Note that the backward pass is 2x more expensive than the forward pass.

## Consider all layers

This was just for w2, need to apply it to all parameters in the network.

Putting it togther:

- Forward pass: 2 (# data points) (# parameters) FLOPs

- Backward pass: 4 (# data points) (# parameters) FLOPs

- Total: 6 (# data points) (# parameters) FLOPs

This is for multilayer perceptrons (MLPs)

...but it turns out to be a good approximation for Transformers for short context lengths as well.

Recall our deep network.

Let's define the AdaGrad optimizer

- momentum = SGD + exponential averaging of grad

- AdaGrad = SGD + averaging by grad^2

- RMSProp = AdaGrad + exponentially averaging of grad^2

- Adam = RMSProp + momentum

AdaGrad 

- [[Duchi+ 2011]]([object Object])

Compute gradients

## Memory

Customary to use fp32 for stability (accumulating averages over powers over many steps)

4 bytes/parameter for storing second moments, Adam requires 8 bytes/parameter for storing first and second moments

## Compute (for one step)

## Transformers

The accounting for a Transformer is more complicated, but the same idea.

Assignment 1 will ask you to do that.

Blog post describing memory usage for Transformer training 

Blog post descibing FLOPs for a Transformer: 

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

- If store every sqrt(L) layers, then activation memory is O(sqrt(L)) and O(sqrt(L)) recomputation, balanced.

Summary:

- Everything is operations on tensors (parameters, gradients, activations, optimizer states, data)

- einops: better way to think about tensor operations

- 6 (# data points) (# parameters) FLOPs per training step

- Arithmetic intensity / roofline analysis: compute-bound or memory-bound?

- Matrix multiplications are compute-bound, elementwise operations are memory-bound

- Gradient accumulation, activation checkpointing: reduce memory to use bigger batch sizes
