# Machine Learning & Generative AI: Project-Based Roadmap

This repository documents my journey from foundational Machine Learning to advanced Generative AI and AI Engineering. The core philosophy of this roadmap is **Project-Based Learning**. 

Instead of reading books cover-to-cover, I use their Tables of Contents as a syllabus. I build, break, and modify code, diving into the books strictly as reference documentation when encountering errors, shape mismatches, or when I need to understand the underlying theory (like the Wasserstein distance $W(p_r, p_g)$ in WGANs).

## The Bookshelf (Reference & Syllabus)
1. **Applied Machine Learning and AI for Engineers** (Jeff Prosise) - *Foundational recap & warm-up*
2. **Generative Deep Learning** (David Foster) - *Generative concepts, VAEs, GANs*
3. **Hands-On Large Language Models** (Jay Alammar) - *Transformers, RAG, Fine-tuning*
4. **Hands on Generative AI with Transformers and Diffusion Models** (Jay Alammar) - *Stable Diffusion, Vision-Language models*
5. **AI Engineering** (Chip Huyen) - *System design, MLOps, Deployment*
6. **Deep Learning** (Goodfellow, Bengio, Courville) - *The "Math Bible" (Strictly for theoretical/mathematical reference)*

---
## Tech Stack & Libraries (PyTorch Ecosystem)

Since I am focusing on **PyTorch** instead of TensorFlow/Keras, the codebase will heavily rely on the PyTorch ecosystem and Hugging Face tools.

**Core Deep Learning:**
* `torch`, `torchvision`, `torchaudio` - Core model building, training loops, and computer vision utilities.
* `accelerate` - For distributed training and handling multi-GPU/MPS (Apple Silicon) setups easily.

**Generative AI & Transformers (Hugging Face):**
* `transformers` - Loading and running LLMs and ViTs.
* `diffusers` - Building and running diffusion models (Stable Diffusion).
* `peft` & `trl` - Parameter-Efficient Fine-Tuning (LoRA) and RLHF.
* `datasets` & `evaluate` - Handling training data and evaluating model metrics.

**NLP & RAG (Retrieval-Augmented Generation):**
* `langchain` or `llama-index` - Orchestrating RAG pipelines and LLM agents.
* `langgraph` - Building stateful, cyclic, and multi-agent AI systems (e.g., when the LLM needs to loop, reflect, or correct its own errors).
* `chromadb`, `faiss-cpu`, or `qdrant-client` - Vector databases for semantic search.

**AI Engineering & Deployment:**
* `fastapi` & `uvicorn` - Serving models via high-performance REST APIs.
* `streamlit` or `gradio` - Building quick web UIs for model demos.
* `wandb` (Weights & Biases) - Tracking experiments, losses, and generated images.

**Data Manipulation & Utilities:**
* `numpy`, `pandas`, `scikit-learn` - Data preprocessing, metrics, and basic ML tasks.
* `matplotlib` & `seaborn` - Visualizing latent spaces, loss curves, and generated outputs.

---

---

## Phase 1: Generative Foundations (Current Stage)
**Primary Reference:** *Generative Deep Learning (David Foster)*

*   [ ] Autoencoders (AE) and Variational Autoencoders (VAEs)
*   [ ] Standard GANs (Generative Adversarial Networks)
*   [ ] Wasserstein GANs (WGAN) and Gradient Penalty (WGAN-GP)
*   [ ] Sequence Models (LSTMs, GRUs) fundamentals

**Phase 1 Projects**
- TODO

---

## Phase 2: The Transformer Era & NLP
**Primary Reference:** *Hands-On Large Language Models (Jay Alammar)*

*   [ ] Word Embeddings and Vector Databases
*   [ ] The Transformer Architecture (Attention Mechanisms)
*   [ ] Prompt Engineering & OpenAI/Open-Source APIs
*   [ ] Retrieval-Augmented Generation (RAG)
*   [ ] Parameter-Efficient Fine-Tuning (PEFT, LoRA)

**Phase 2 Project Ideas:**
- TODO

---

## Phase 3: Diffusion Models & Advanced Vision
**Primary Reference:** *Hands on Generative AI with Transformers and Diffusion Models (Jay Alammar)*

*   [ ] Forward and Reverse Diffusion Processes
*   [ ] Stable Diffusion Architecture (U-Net, Text Encoders)
*   [ ] Conditioning and ControlNet
*   [ ] Vision Transformers (ViT)

**Phase 3 Project Ideas:**
- TODO

---

## Phase 4: AI Engineering & MLOps
**Primary References:** *AI Engineering (Chip Huyen) & Applied ML (Jeff Prosise)*

*   [ ] Model Evaluation and Metrics for GenAI
*   [ ] Latency, Throughput, and Cost Optimization
*   [ ] Serving Models (FastAPI, ONNX, TensorRT)
*   [ ] Monitoring (Data Drift, Concept Drift, Hallucination detection)

**Phase 4 Project Ideas:**
- TODO

---
