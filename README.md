# Irony Detection with Fine-Tuned LLMs

Research project focused on fine-tuning transformer models to detect irony in short English utterances (≤10 words) for educational assessment.

## Objective

Build a model capable of distinguishing ironic vs. literal meaning with 90-95% accuracy for student response analysis.

## Current Status

### Base Model Performance
- **Model**: [cardiffnlp/twitter-roberta-base-irony](https://huggingface.co/cardiffnlp/twitter-roberta-base-irony)
- **Sample Dataset (48 examples)**: Accuracy 48.9%, Precision 66.7%, Recall 22.2%, F1 33.3%
- **Issue**: Model too conservative, misses most ironic instances

### Fine-Tuning Results

**Dataset**: [Kaggle Ironic Corpus](https://www.kaggle.com/datasets/rtatman/ironic-corpus)
- **Original**: 1,936 Reddit comments
- **Cleaned & Balanced**: 654 sentences (327 ironic, 327 literal)
- **Splits**: 80% train, 10% validation, 10% test (10-fold cross-validation)

**Performance** (averaged across folds):
- Training: **77% accuracy**, F1 0.77, Precision 0.78, Recall 0.77
- Sample dataset: **55% accuracy**, F1 0.54, Precision 0.58, Recall 0.57

**Key Finding**: Model performs better on training domain (social media) than conversational/script-based text.

## Repository Structure

(Under work as of November 12th, 2025)

## Data Processing

### Cleaning & Balancing
```bash
python first_dataset_clean_and_balance.py
```
- Removes multi-line entries and irregular sentence lengths
- Balances ironic/literal classes
- Outputs `balanced_dataset.csv`

### Creating Splits
```bash
python first_dataset_splitter.py
```
- Stratified 80-10-10 splits
- 10-fold cross-validation setup

## Roadmap

- [ ] Integrate additional irony/sarcasm datasets
- [ ] Experiment with domain adaptation techniques
- [ ] Test LoRA fine-tuning and instruction-tuned models
- [ ] Cross-domain evaluation (social media → conversational)
- [ ] Deploy final model and publish results

## Citation

Dataset based on:
> Wallace, B. C., Do Kook Choe, L. K., Kertz, L., & Charniak, E. (2014).  
> *Humans Require Context to Infer Ironic Intent (so Computers Probably do, too).*  
> ACL 2014, pp. 512-516.  
> [PDF](http://www.byronwallace.com/static/articles/wallace-irony-acl-2014.pdf)

## License

Research project - dataset sourced from Kaggle (Rachael Tatman).
