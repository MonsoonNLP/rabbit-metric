# RABBIT - Regularly Adjusting emBeddings Benchmark IniTiative
## A moving target benchmark

> If ImageNet, NLP and recommendation benchmarks are all being gamed now, what does the next generation of AI benchmarks look like? Could you make one as a moving target?

https://twitter.com/mapmeld/status/1153781332020846598

### Rationale

Pre-trained language models store word associations as a multidimensional vector space. These models
are reusable maps of language, but after training they gradually become a time capsule
of their language. Over time, there is new vocabulary and new associations (beyond finetuning their existing models). Two
examples of outdated language models would be:

- word2vec, released in 2013, not recognizing political Tweets in 2016
- mBERT and T5, released in late 2018 and early 2020, not having vocabulary or mask-prediction for medical and social-impact terms relating to coronavirus

Developers cannot easily collect a similar corpus and retrain these corporate-published models every time that they want
to make changes.

Longer: https://medium.com/swlh/patching-pre-trained-language-models-28ed6ea8b0bc

### The Rules

- Based on pre-2020 language model (BERT, RoBERTa, XLM)
- Script receives new vocabulary list and training sentences, including 'confusable' sentences (ex. if 'COVID vaccine' is a new token, real world sentences with 'flu vaccine' will be included to test new vocab is not overwhelming)
- Current data is hardcoded (from Reddit), should include hidden/randomized data
- Ranked on ability to update model
- Fixed random seeds
- Runs on CPU in a constrained timespan
- Goal: runs on GitHub Actions after PR
- Over time, re-run on new batches of stories and language

### Thoughts on production-izing

Should use OpenML, https://github.com/mlperf/mlbox, https://github.com/facebookresearch/dagger, or https://sotabench.com for interoperability and connection to mainstream tests.
