# NLP From Scratch: Word Embeddings to LLM Alignment

Core NLP and LLM techniques implemented from first principles in **PyTorch**. The project runs from classical text classification and word embeddings, through character-level language models, to fine-tuning and preference-aligning a small language model for math reasoning.

> Built as part of **DS 207: Introduction to NLP** at the **Indian Institute of Science (IISc)**, taught by Prof. Danish Pruthi (Jan–Apr 2026). The course provided the notebook scaffolding, datasets and evaluation harness. The model components, training loops and losses listed below are my own implementations.

---

## Highlights

| Module | What was built | Key result |
|---|---|---|
| 1. Word representations & text classification | Naive Bayes, Skip-gram Word2Vec with negative sampling, BoW and Word2Vec classifiers | `<NB accuracy>` on AG News |
| 2. Language modelling | Smoothed n-gram LMs, a neural n-gram (feed-forward) LM and an RNN LM over characters | `<best perplexity>` validation perplexity (RNN) |
| 3. LLM alignment | Supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) of an ~82M-parameter GPT-2 on GSM8K | SFT **~2× Pass@5** over the base model |

---

## Module 1: Word Representations & Text Classification

**Goal:** classify news articles (AG News, 4 classes) and learn word embeddings from raw text.

- **Naive Bayes classifier (from scratch):** vocabulary construction with a minimum-frequency cutoff, class priors and word likelihoods with Laplace smoothing, log-probability inference. Includes analysis of the most indicative words per class (raw counts vs. likelihood ratios).
- **Word2Vec (Skip-gram with negative sampling):** trained on WikiText-2. Includes context-window sampling, negative sampling to avoid the full-softmax bottleneck, and a custom PyTorch model and trainer.
- **Embedding evaluation:** cosine and Euclidean similarity heatmaps for related vs. unrelated word pairs, compared against pretrained `word2vec-google-news-300`.
- **Discriminative classifiers:** a bag-of-words classifier and a Word2Vec-feature classifier for comparison with Naive Bayes.

| Model | Accuracy (AG News) |
|---|---|
| Naive Bayes | `<fill>` |
| Bag-of-Words classifier | `<fill>` |
| Word2Vec-based classifier | `<fill>` |

---

## Module 2: Character-Level Language Modelling

**Goal:** model city names character by character, then generate new names and complete prefixes.

- **N-gram language models:** unigram, bigram and trigram models with **Laplace** and **interpolation** smoothing, all sharing one base class for perplexity, sampling and next-character prediction.
- **Neural n-gram LM:** a feed-forward network over fixed-length character contexts, with its own training and inference wrapper.
- **RNN LM:** a recurrent language model with embedding, RNN and projection layers, trained end to end.
- **Evaluation:** perplexity on train and validation sets, sampled name generation, prefix-conditioned generation and top-k next-character prediction.

| Model | Validation perplexity |
|---|---|
| Unigram (smoothed) | `<fill>` |
| Bigram (Laplace / interpolation) | `<fill>` |
| Trigram (smoothed) | `<fill>` |
| Neural n-gram (FNN) | `<fill>` |
| RNN | `<fill>` |

---

## Module 3: LLM Alignment: SFT and DPO on GSM8K

**Goal:** teach a small language model (`deadMarkov/distilgpt2-math`, ~82M parameters) to solve grade-school math word problems with step-by-step reasoning.

- **Supervised fine-tuning (SFT):** a chain-of-thought dataset built from GSM8K question, reasoning and answer fields; prompt tokens masked out of the loss; shifted cross-entropy; a dynamic-padding collator; a custom training loop with an AdamW optimiser and scheduler.
- **Direct Preference Optimization (DPO):** a preference dataset of chosen vs. rejected responses. The policy is initialised from the SFT model against a **frozen reference model**. Implemented per-token log-probabilities, implicit rewards and the DPO loss (log-sigmoid of the β-scaled reward margin).
- **Evaluation:** **Pass@1** (single-attempt accuracy) and **Pass@5** (at least one of 5 samples correct), with answer extraction from generated reasoning.

| Model | Pass@1 | Pass@5 |
|---|---|---|
| Base model | `<fill>` | `<fill>` |
| + SFT | `<fill>` | `<fill>` (**~2× base**) |
| + DPO | `<fill>` | `<fill>` |

> At ~82M parameters, absolute GSM8K scores are expected to be low. The focus is on the relative gains from SFT and DPO and on implementing the training methods correctly.

---

## Repository Structure

```
NLP-From-Scratch/
├── 01_word_representations_text_classification.py   # Naive Bayes, Word2Vec, BoW / Word2Vec classifiers
├── 02_language_modelling.py                          # n-gram, neural n-gram and RNN language models
├── 03_llm_alignment_sft_dpo.py                       # SFT and DPO on GSM8K, Pass@k evaluation
└── README.md
```

## How to Run

Each file is an export of a Google Colab notebook and is designed to run on a free Colab GPU.

1. Open a file in Google Colab (or convert it back to a notebook with `jupytext`).
2. Install dependencies:
   ```bash
   pip install torch transformers datasets tokenizers gensim nltk scikit-learn gdown matplotlib seaborn torchtext
   ```
3. Run the cells top to bottom. Datasets download automatically: AG News, WikiText-2, the city-names dataset and GSM8K.

Approximate runtimes on a Colab T4: Module 1 ≈ 15 min, Module 2 ≈ 10 min, Module 3 ≈ 60–90 min.

## Tech Stack

Python · PyTorch · Hugging Face Transformers & Datasets · Gensim · NLTK · scikit-learn · NumPy · pandas · Matplotlib

## Key Takeaways

- Simple generative baselines such as Naive Bayes are strong on topic classification when the vocabulary is tuned well.
- Negative sampling makes Skip-gram training tractable, and embedding quality depends heavily on corpus size.
- Smoothing choices dominate n-gram perplexity, and neural models generalise better to unseen character contexts.
- On small LMs, SFT gives the largest jump by teaching output format and when to stop. DPO then refines answer preference relative to a frozen reference.
