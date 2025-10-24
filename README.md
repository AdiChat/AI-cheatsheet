# AI cheatsheet
*ai/ml resources to master state-of-the-art (SOTA) techniques from engineers and researchers* 🧠💻

---

AI blogs you should read.


## Main AI blogs to read regularly (continuous learning)

* [NVIDIA Developer Blog](https://developer.nvidia.com/blog/): Deep dive into multiple AI topics.
* [Connectionism- Thinking Machine blog](https://thinkingmachines.ai/blog/): AI startup. Founded by Mira Murati, former CTO at OpenAI. Solved nondeterminism problem in LLM.
* [TensorRT LLM tech blogs](https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs/source/blogs/tech_blog): Deep dive into technical techniques/optimizations in one of the leading LLM inference library. (13 posts as of now)
* [SGLang tech blog](https://lmsys.org/blog/): SGLang is one of the leading LLM serving framework. Most blogs are around SGLang but is rich in technical information.

YouTube channels to follow regularly:

* [vLLM office hours](https://www.youtube.com/watch?v=uWQ489ONvng&list=PLbMP1JcGBmSHxp4-lubU5WYmJ9YgAQcf3): Deep dive into various technical topics in vLLM
* [GPU Mode](https://www.youtube.com/@GPUMODE/videos): Deep dive into various LLM topics from guests from the AI community
* [PyTorch channel](https://www.youtube.com/@PyTorch/videos): videos of various PyTorch events covering keynotes of technical topics like torch.compile.

---

## Deep dive into AI concepts [Learn step-by-step]
_Listed only high-quality resources. No need to read 100s of posts to get an idea. Just one post should be enough._

### GPU architecture

Current SOTA AI/LLM workloads are possible only because of GPUs. Understanding GPU architecture gives you an engineering edge.

* [Understanding GPU architecture with MatMul](https://www.aleksagordic.com/blog/matmul)

## Transformer



* Encoder-only and Decoder-only models
* BERT (_insightful_): [BERT as text diffusion step](https://nathan.rs/posts/roberta-diffusion/)

### Quantization

* [Optimizing FP4 Mixed-Precision Inference on AMD GPUs](https://lmsys.org/blog/2025-09-21-petit-amdgpu/)

### Post-training

* [Post training concepts with SFT, RLHF, RLFR](https://tokens-for-thoughts.notion.site/post-training-101)

## Software tools AI

* [vLLM](https://www.aleksagordic.com/blog/vllm): architecture of the leading LLM serving engine.

## Courses/guides to follow

MUST:
* [CSE223](https://hao-ai-lab.github.io/cse234-w25/): ML Sys by Prof Hao Zhang (rating 10/10)

MAYBE:
* [CME 295](https://cme295.stanford.edu/syllabus/): Basics of Transformer and LLM
* [Scaling a model](https://jax-ml.github.io/scaling-book/roofline/) (rating 7/10)


## What to contribute in leading AI open-source projects?

Get started in these:

* [SGLang](https://github.com/sgl-project/sglang): LLM serving engine originally from UC Berkeley.
* [vLLM](https://github.com/vllm-project/vllm): LLM inference engine originally from UC Berkeley.
* [PyTorch](https://github.com/pytorch/pytorch): Leading AI framework by Meta
* [TensorFlow](https://github.com/tensorflow/tensorflow): AI framework by Google
* [TensorRT](https://github.com/NVIDIA/TensorRT): High performance inference library by NVIDIA
* [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM): LLM inference library by NVIDIA
* [NCCL](https://github.com/NVIDIA/nccl): High performance GPU communication library by NVIDIA
* See other [NVIDIA libraries](https://github.com/orgs/NVIDIA/repositories?language=&q=&sort=&type=all).
