# Information Retrieval with BEIR Dataset - Mini Project 1 (Part 2)

## Overview

This part focuses on implementing and evaluating an information retrieval system using the BEIR (Benchmarking Information Retrieval) dataset. Students will work with document encoding, similarity calculation, ranking, and model fine-tuning to improve retrieval performance.

## Dataset

This assignment uses the **NFCorpus** dataset from the BEIR benchmark:

- **Corpus Dataset**: [BeIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus)
- **Query Relevance (QRELs)**: [BeIR/nfcorpus-qrels](https://huggingface.co/datasets/BeIR/nfcorpus-qrels)

### Understanding the Dataset

Before starting the assignment, familiarize yourself with:
- The structure of the BEIR dataset format
- How queries and documents are organized
- The relevance judgments (qrels) format
- Document IDs and query-document relationships

## Learning Objectives

By completing this assignment, you will:

1. **Understand BEIR dataset structure** and preprocess retrieval data
2. **Implement encoding systems** for queries and documents using embeddings
3. **Calculate similarity scores** to rank documents by relevance
4. **Evaluate retrieval performance** using metrics like Mean Average Precision (MAP)
5. **Fine-tune models** to improve retrieval results

## Assignment Components

### Part 1: Document Ranking (10 Points)

Implement and compare different encoding methods for document ranking.

#### Tasks:
- Encode queries and documents using different embedding methods
- Calculate cosine similarity between query and document embeddings
- Rank documents based on similarity scores
- Evaluate ranking quality

#### Report Requirements:

**1. Comparison of Encoding Methods**
- Compare GloVe embeddings vs. Sentence Transformer embeddings vs Openai Embedding
- Which method ranked documents better?
- Did the top-ranked documents make sense?
- How does cosine similarity behave with different embeddings?

**2. Observations on Cosine Similarity & Ranking**
- Did the ranking appear meaningful?
- Were there cases where documents that should be highly ranked were not?
- What are possible explanations for incorrect rankings?

**3. Possible Improvements**
- What can be done to improve document ranking?
- Would a different distance metric (e.g., Euclidean, Manhattan) help?
- Would preprocessing the queries or documents (e.g., removing stopwords) improve ranking?

### Part 2: Model Fine-Tuning (15 Points)

Fine-tune retrieval models and analyze the impact on performance.

#### Tasks:
- Implement different training strategies
- Fine-tune models on the retrieval task
- Compare performance before and after fine-tuning
- Analyze training dynamics and results

#### Report Requirements:

**1. Comparison of Different Training Strategies**
- Compare `[anchor, positive]` vs `[anchor, positive, negative]` training approaches
- Which approach seemed to improve ranking?
- How did the model behave differently?

**2. Impact on MAP Score**
- Did fine-tuning improve or hurt the Mean Average Precision (MAP) score?
- If MAP decreased, why might that be?
- Is fine-tuning always necessary for retrieval models?

**3. Observations on Training Loss & Learning Rate**
- Did the loss converge?
- Was the learning rate too high or too low?
- How did freezing/unfreezing layers impact training?

**4. Future Improvements**
- Would training with more negatives help?
- Would changing the loss function (e.g., using Softmax Loss) improve performance?
- Could increasing the number of epochs lead to a better model?

## Evaluation Metrics

### Mean Average Precision (MAP)

MAP is the primary evaluation metric for this assignment. It measures:
- How many relevant documents are retrieved
- The ranking position of relevant documents
- Overall retrieval quality across all queries

Higher MAP scores indicate better retrieval performance.

## Key Concepts

### Similarity Metrics
- **Cosine Similarity**: Measures angle between vectors (range: -1 to 1)
- **Euclidean Distance**: Straight-line distance between vectors
- **Manhattan Distance**: Sum of absolute differences

### Training Approaches
- **[anchor, positive]**: Binary training with relevant pairs
- **[anchor, positive, negative]**: Triplet training with contrastive learning

## Tips for Success

1. **Start with data exploration** - Understand the dataset structure thoroughly
2. **Visualize results** - Plot similarity distributions and ranking results
3. **Experiment systematically** - Change one variable at a time
4. **Document everything** - Keep notes on experiments and observations
5. **Compare fairly** - Use the same test set for all comparisons
6. **Analyze failures** - Understanding why things fail is as important as successes

## Common Pitfalls

- Not normalizing embeddings before calculating cosine similarity
- Using too high a learning rate during fine-tuning
- Not having enough training data for fine-tuning
- Ignoring the importance of negative samples
- Not evaluating on a held-out test set

## Submission Requirements

1. Completed Jupyter notebook with all code cells executed
2. Ranking Documents and Fine-Tuning Report (PDF/Markdown)
3. Any saved models or additional files referenced in your reports(optional)

## Resources

- [BEIR Benchmark Paper](https://arxiv.org/abs/2104.08663)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Information Retrieval Metrics](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/)

**Good luck with your information retrieval journey!** 