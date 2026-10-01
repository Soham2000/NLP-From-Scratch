# NLP From Scratch: Word Embeddings to LLM Alignment

Core NLP and LLM techniques implemented in **PyTorch**: classical text classification, word embeddings, character-level language models, and fine-tuning plus preference alignment of a small language model for math reasoning.

> Coursework for **DS 207: Introduction to NLP**, Indian Institute of Science (IISc), taught by Prof. Danish Pruthi (Jan–Apr 2026). The course provided the notebook scaffolding, datasets and evaluation code. The models, training loops and loss functions are my implementations.

---

## 1. Word Representations & Text Classification
`Soham_Chakraborty_25845_assignment1.py`

- **Naive Bayes classifier** on AG News: vocabulary building, Laplace smoothing and log-probability inference.
- **Word2Vec (Skip-gram with negative sampling)** trained on WikiText-2, with a custom PyTorch model and trainer.
- **Embedding evaluation** with cosine and Euclidean similarity, compared against pretrained `word2vec-google-news-300`.
- **Bag-of-Words** and **Word2Vec-based** classifiers.

## 2. Character-Level Language Modelling
`assignment2.py`

Models of city names, generated one character at a time:
- **N-gram LMs:** unigram, bigram and trigram, with Laplace and interpolation smoothing.
- **Neural n-gram LM:** a feed-forward network.
- **RNN LM.**
- **Evaluation:** perplexity, name generation, prefix completion and next-character prediction.

## 3. LLM Alignment: SFT and DPO on GSM8K
`Soham Chakraborty_25845_assignment3.py`

Base model: `deadMarkov/distilgpt2-math` (~82M parameters).
- **Supervised fine-tuning (SFT)** on GSM8K chain-of-thought data, with prompt tokens masked out of the loss.
- **Direct Preference Optimization (DPO)** on chosen/rejected response pairs, using a frozen SFT reference model: log-probabilities, implicit rewards and the β-scaled preference loss.
- **Evaluation:** Pass@1 and Pass@5.

---

## How to Run

Each `.py` file is an export of a Google Colab notebook. Open it in Colab, or convert it with `jupytext`, and run the cells in order. Datasets are downloaded inside the code.

```bash
pip install torch transformers datasets tokenizers gensim nltk scikit-learn gdown matplotlib seaborn torchtext
```

## Tech Stack

Python · PyTorch · Hugging Face Transformers & Datasets · Gensim · NLTK · scikit-learn
