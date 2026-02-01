# SemanticJoin
An implementation of the semantic join operator: it takes unstructured data as input (e.g. text/images) and calculates join relationship between tables based on a Natural Language Predicate. In my implementation, I have used large language models such as Open AI's GPT-4o for computing the join output from LLMs. 
Steps:
1. I prompt an LLM with a smaller subsample of the dataset to compute the join output based on NL Predicate (similarity/contradiction).
2. I train a local Neural Network model with the output of the LLM to learn the association described by the predicate.
3. I use the learned model to perform inference on the remainder of the data.

Findings:
Our method outperforms the block join operator in terms of accuracy, precision, recall and F1, which uses LLM to compute join operation over all the data at once, for both the IMDB Movie Review dataset and the Amazon Product Review dataset.
