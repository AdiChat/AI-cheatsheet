# AI cheatsheet
*ai/ml resources to master state-of-the-art (SOTA) techniques from engineers and researchers* 🧠💻

---

## Free guides to follow

MUST:
- [ ] [CSE223](https://hao-ai-lab.github.io/cse234-w25/): ML Sys course by Prof Hao Zhang (rating 10/10) by UC San Diego
- [ ] [CME 295](https://cme295.stanford.edu/syllabus/): Basics of Transformer and LLM course by Stanford University
- [ ] [AI Engineering Silicon Cheatsheet](https://amzn.to/3Wl5Tum): Cheatsheet covering all major concepts in modern AI; Must for reference
- [ ] [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=first_steps:_training_on_one_gpu): Training LLMs on GPU Clusters
- [ ] [Llama visualization](https://www.alphaxiv.org/labs/tensor-trace): step by step [analyze each tensor](https://www.alphaxiv.org/labs/fly-through-llama) as it is processed in Llama

## Main AI blogs to read regularly (continuous learning)

- [ ] [NVIDIA Developer Blog](https://developer.nvidia.com/blog/): Deep dive into multiple AI topics.
- [ ] [Connectionism- Thinking Machine blog](https://thinkingmachines.ai/blog/): AI startup. Founded by Mira Murati, former CTO at OpenAI. Solved nondeterminism problem in LLM.
- [ ] [TensorRT LLM tech blogs](https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs/source/blogs/tech_blog): Deep dive into technical techniques/optimizations in one of the leading LLM inference library. (13 posts as of now)
- [ ] [SGLang tech blog](https://lmsys.org/blog/): SGLang is one of the leading LLM serving framework. Most blogs are around SGLang but is rich in technical information.

YouTube channels to follow regularly:

- [ ] [vLLM office hours](https://www.youtube.com/watch?v=uWQ489ONvng&list=PLbMP1JcGBmSHxp4-lubU5WYmJ9YgAQcf3): Deep dive into various technical topics in vLLM
- [ ] [GPU Mode](https://www.youtube.com/@GPUMODE/videos): Deep dive into various LLM topics from guests from the AI community
- [ ] [PyTorch channel](https://www.youtube.com/@PyTorch/videos): videos of various PyTorch events covering keynotes of technical topics like torch.compile.

---

## Deep dive into AI concepts [Learn step-by-step]
_Listed only high-quality resources. No need to read 100s of posts to get an idea. Just one post should be enough._

* **GPU architecture**
<br> Current SOTA AI/LLM workloads are possible only because of GPUs. Understanding GPU architecture gives you an engineering edge.
- [ ] [Understanding GPU architecture with MatMul](https://www.aleksagordic.com/blog/matmul)
- [ ] [GPU Shared memory banks / microbenchmarks](https://feldmann.nyc/blog/smem-microbenchmarks)
GPU programming concepts:
- [ ] [CUDA programming model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/), [GPU memory management](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62550/): Mark Harris's GTC Talk on Coalesced Memory Access, [Prefix Sum/ Scan in GPU](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

* **Transformer**
- [ ] Encoder-only and Decoder-only models
- [ ] BERT (_insightful_): [BERT as text diffusion step](https://nathan.rs/posts/roberta-diffusion/)
- [ ] [Memory requirements for LLM](https://themlsurgeon.substack.com/p/the-memory-anatomy-of-large-language). There are 4 parts: activation, parameter, gradient, optimizer states.

* **Attention**
- [ ] Multi-head attention (MHA), Multi-Query attention (MQA), Group Query Attention (GQA), MLA (used in DeepSeek)
- [ ] FlashAttention (paper1, paper for v2, paper for v3), Online softmax
- [ ] Ring Attention (links to Context Parallelism CP): Handles large sequence length, Flex Attention, Masking
- [ ] KV cache, FP8 KV cache, Paged Attention

* **Quantization**

- [ ] [Quantization basics](https://themlsurgeon.substack.com/p/the-machine-learning-surgeons-guide), [INT8 quantization using QAT](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/), [LLM quantization with PTQ](https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/), [FP8 datatype](https://developer.nvidia.com/blog/floating-point-8-an-introduction-to-efficient-lower-precision-ai-training/)
- [ ] [Per-tensor and per-block scaling](https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/)
- [ ] [NVFP4 training](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/), [Optimizing FP4 Mixed-Precision Inference on AMD GPUs](https://lmsys.org/blog/2025-09-21-petit-amdgpu/)
- [ ] [Quantization on CPU (GGUF, AWQ, GPTQ)](https://www.ionio.ai/blog/llms-on-cpu-the-power-of-quantization-with-gguf-awq-gptq), [GGUF quantization method](https://www.reddit.com/r/LocalLLaMA/comments/1ba55rj/overview_of_gguf_quantization_methods/)
---
- [ ] [Pruning and distillation](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model/)

* Post-training
- [ ] [Post training concepts with SFT, RLHF, RLFR](https://tokens-for-thoughts.notion.site/post-training-101)

* Optimizations

- [ ] [LLM Inference optimizations](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [ ] 5D parallelism [PP, SP, DP, TP, CP, EP](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/parallelisms.html), [parallelism](https://themlsurgeon.substack.com/p/data-parallelism-scaling-llm-training) concept for LLM scaling.
- [ ] [Chunk prefill - SARATHI paper](https://arxiv.org/pdf/2308.16369), [dynamic and continuous batching](https://bentoml.com/llm/inference-optimization/static-dynamic-continuous-batching)
- [ ] [KV cache offloading](https://bentoml.com/llm/inference-optimization/kv-cache-offloading), [KVcache early reuse](https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/)
- [ ] [Speculative decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding)
- [ ] [P/D disaggregation](https://bentoml.com/llm/inference-optimization/prefill-decode-disaggregation), [DistServe P/D disaggregation paper](https://arxiv.org/pdf/2401.09670)
- [ ] [MoE using Wide Expert Parallelism EP](https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems/)

* Software tools AI
- [ ] [vLLM arch](https://www.aleksagordic.com/blog/vllm): architecture of the leading LLM serving engine.

Insights:

- [ ] [MinMax M2 using Full Attention](https://x.com/zpysky1125/status/1983383094607347992): why full attention is better than masked attention?

## MAYBE guides you may go through

- [ ] [Scaling a model](https://jax-ml.github.io/scaling-book/roofline/) (rating 7/10)
- [ ] [GPU Gems 3](https://developer.nvidia.com/gpugems/gpugems3): if you want to dive deep into GPU programming

## What to contribute in leading AI open-source projects?

Get started in these:

- [ ] [SGLang](https://github.com/sgl-project/sglang): LLM serving engine originally from UC Berkeley.
- [ ] [vLLM](https://github.com/vllm-project/vllm): LLM inference engine originally from UC Berkeley.
- [ ] [PyTorch](https://github.com/pytorch/pytorch): Leading AI framework by Meta
- [ ] [TensorFlow](https://github.com/tensorflow/tensorflow): AI framework by Google
- [ ] [TensorRT](https://github.com/NVIDIA/TensorRT): High performance inference library by NVIDIA
- [ ] [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM): LLM inference library by NVIDIA
- [ ] [NCCL](https://github.com/NVIDIA/nccl): High performance GPU communication library by NVIDIA
- [ ] See other [NVIDIA libraries](https://github.com/orgs/NVIDIA/repositories?language=&q=&sort=&type=all).
