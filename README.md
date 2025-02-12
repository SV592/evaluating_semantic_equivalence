# An Evaluation of Automated Code Review Approaches

## 📌 Overview
Automated code review tools have become an essential part of modern software development, streamlining the review process by suggesting comments on code changes. However, verifying the **semantic correctness** of these automatically generated comments remains a challenge, often requiring manual intervention. This project explores a **framework** that leverages **pre-trained language models** to assess the semantic equivalence between generated and ground-truth comments.

## 🚀 Features
- **Evaluation of Code Review Comments**: Uses **GPT-4** and **all-miniLM-L6-v2** to analyze the **semantic equivalence** of review comments.
- **Multiple Evaluation Contexts**: Assesses both **comment-only** and **code + comment** scenarios.
- **Statistical Performance Metrics**: Includes **Precision, Recall, F1-Score, Accuracy, MCC, and Cosine Similarity**.
- **Comparative Model Analysis**: Compares efficiency and cost trade-offs between deep learning models for large-scale review automation.

## 📊 Study Results
| Model | Context | Precision | Recall | F1-Score | Accuracy | MCC |
|--------|------------|-----------|---------|---------|---------|------|
| **GPT-4** | Code & Comment | 0.7750 | 0.8611 | 0.8158 | 0.86 | 0.7059 |
| **GPT-4** | Comment-Only | **0.8462** | **0.9167** | **0.8800** | **0.91** | **0.8098** |
| **all-miniLM-L6-v2** | Comment-Only | 0.8750 | 0.1944 | 0.3165 | 0.8824 | 0.8119 |

## 📌 Research Questions
1. **Effectiveness**: How well can pre-trained models identify **semantically equivalent** code review comments?
2. **Efficiency**: How fast and scalable are these methods for **automated code review** without manual intervention?

## 🔬 Study Design
- **Dataset**: 100 manually labeled samples of review comments
- **Evaluation Methods**: Ten different **prompt-based** experiments comparing GPT-4 and all-miniLM-L6-v2
- **Performance Metrics**: Semantic correctness assessed using NLP-based similarity scores

## 📌 Key Findings
- **GPT-4 outperforms** all-miniLM-L6-v2 in understanding programming-specific semantics
- **Code snippets provide marginal improvements** in evaluation accuracy
- **all-miniLM-L6-v2 is more cost-effective** but struggles with programming nuances

## 📖 References
- [OpenAI GPT-4](https://openai.com/research/gpt-4)
- [Hugging Face all-miniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Code Review Research Papers](https://arxiv.org/abs/2203.04821)

