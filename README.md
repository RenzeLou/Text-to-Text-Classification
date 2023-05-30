
## 1. Motivation

This repository contains the experimental code for Text-to-Text Classifier.

Text classification was conventionally handled by supervised classifiers which treat labels as a set of indices. Recently, the NLP community increasingly treats all tasks as a text-2-text generation paradigm. Especially for some generative large language models, such as the GPT series, are highly reliant on this paradigm. 

Although this seems a natural choice for text generation task, but **if it really fit text classification tasks?** Will it decrease or increase the text classification performance? 

We tend to systematically study this research question by conducting experiments on various popular classification datasets. 


## 2. News

- 04/10/2023: Add code for **Indent Indentification** task. We use [*banking_data*](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) for experiments. See the [README](./intent_identification/README.md) for more details.
- 04/24/2023: Upload few-shot training.
- 05/14/2023: Add **[SuperNI](https://github.com/allenai/natural-instructions)** experiments to investigate (1) the difference between classifier and generator on the CLS part of the held-out evaluation; (2) whether using generation tasks (training) can help with generalization on classification tasks (testing). See the [README](./Super_NI/README.md) for more details.
- 05/29/2023: Upload **[FewRelv1.0](https://github.com/thunlp/FewRel)** experiments to investigate the cross-label generalization setting. See the [README](./FewRel/README.md) for more details.