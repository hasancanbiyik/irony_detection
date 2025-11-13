# Irony Detection Project

This repository documents our ongoing research on **fine-tuning large language models (LLMs)** for **irony detection** in English utterances.  
The main goal is to build and evaluate transformer-based models that can distinguish *ironic* vs *literal* meaning in utterances that we will get from the students participating in this research.

---

## Current Stage — Finetuning and evaluating the cardiffnlp/twitter-roberta-base-irony model (November 11th, 2025)

We have prepared the **[Ironic Corpus dataset](https://www.kaggle.com/datasets/rtatman/ironic-corpus)** dataset for fine-tuning and evaluating the model.
The pre-processing and balancing code can be found here: [Clean and Balance](https://github.com/hasancanbiyik/irony_detection/blob/main/first_dataset_clean_and_balance.py)
The splitting code (80% training, 10% testing, and 10% validation) and stratification can be found here: [Dataset Splitter](https://github.com/hasancanbiyik/irony_detection/blob/main/first_dataset_splitter.py)

### About the Dataset

- **Source:** [Rachael Tatman — Ironic Corpus (Kaggle)](https://www.kaggle.com/datasets/rtatman/ironic-corpus)  
- **Content:** 1,950 Reddit comments labeled as *ironic (1)* or *not ironic (-1)*  
- **Original Paper:**  
  Wallace, B. C., Do Kook Choe, L. K., Kertz, L., & Charniak, E. (2014).  
  *Humans Require Context to Infer Ironic Intent (so Computers Probably do, too).*  
  In *ACL 2014* (pp. 512–516).  
  [PDF](http://www.byronwallace.com/static/articles/wallace-irony-acl-2014.pdf)

**Acknowledgements:** Supported by the Army Research Office (ARO), grant 64481-MA / W9111F-13-1-0406.

---

## Step 1: Dataset Cleaning & Balancing Script

The dataset was **quite messy**, containing multi-line entries, inconsistent sentence lengths, and noisy annotations.  
The script `clean_and_balance.py` helps prepare a clean, balanced dataset for fine-tuning.

### What It Does

1. Loads and separates *ironic (1)* vs *literal (0)* examples  
2. Removes multi-line and overly short/long literal sentences  
3. Normalizes formatting in ironic sentences  
4. Balances the dataset, allowing a 5–10% difference between classes  
5. Saves the cleaned dataset as `balanced_dataset.csv`

### Usage

```bash
pip install pandas
python clean_and_balance.py
```

You can edit configuration variables (e.g. word limits, tolerance) directly inside the script.
Do not forget to adjust the dataset's directory! It was named as "messy_irony_data" for this project as you can see in the script.

### Example Output

```
[INFO] Literals after cleaning: 327
[INFO] Ironics after cleaning: 537
[INFO] Downsampled ironic to match the smaller class.
========== FINAL COUNTS ==========
Literals (label 0): 327
Ironics  (label 1): 327
Difference: 0 (0.00% of smaller class)
===================================
```

---

## Roadmap

### Phase 1 — Data Preparation (Current)
- Clean and balance datasets
- Integrate other irony/sarcasm corpora (Twitter, news, essays)

### Phase 2 — Model Fine-Tuning
- Fine-tune `cardiffnlp/twitter-roberta-base-irony`
- Experiment with LoRA and instruction-tuned models (e.g., LLaMA-2, Mistral)

### Phase 3 — Evaluation & Analysis
- Evaluate on held-out and cross-domain data
- Compute precision, recall, F1, and interpret model outputs

### Phase 4 — Reporting & Publication
- Summarize results and qualitative insights
- Prepare a short paper or technical report
- Upload trained checkpoints to Hugging Face

---

## Citation

If you use this dataset, please cite the original paper:

> Wallace, B. C., Do Kook Choe, L. K., Kertz, L., & Charniak, E. (2014).  
> *Humans Require Context to Infer Ironic Intent (so Computers Probably do, too).*  
> In *ACL (2)* (pp. 512–516).
