# Mirror Me: Self-Affirmation Text Generator Using Fine-Tuned GPT-2

### Best Affirmation Example  
[See Output Example](#sample-generated-affirmations)

### How to Run the GUI and Generate Text:  
[Accessing and Running on Quest](#accessing-and-running-on-quest)

---

## Table of Contents
* [Overview](#overview)
* [What This Project Actually Does](#what-this-project-actually-does)
* [What You Can Use This For](#what-you-can-use-this-for)
* [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
* [Model Architecture](#model-architecture)
* [Folder Structure](#folder-structure)
* [Accessing and Running on Quest](#accessing-and-running-on-quest)
* [Extra Criteria - GUI Overview](#extra-criteria---gui-overview)
* [Sample Generated Affirmations](#sample-generated-affirmations)
* [Training Loss Graph](#training-loss-graph)
* [Data Preparation & Transfer](#data-preparation--transfer)
* [Model Training on Quest (Northwestern Quest)](#model-training-on-quest-northwestern-quest)
* [Final Scripts I Will Run](#final-scripts-i-will-run)
* [Professor Instructions](#professor-instructions)
* [Future Improvements](#future-improvements)
* [References and Tools Used](#references-and-tools-used)

---

## Overview
**Mirror Me** is an AI-powered affirmation generator that takes in journal-style text and optionally an emotional label (e.g., "gratitude", "worry") to generate a comforting, self-affirming response using a fine-tuned GPT-2 language model.

---

## What This Project Actually Does
Fine-tunes a GPT-2 model on a paired dataset of emotional tweets and positive affirmations. At inference time, the user can provide a journal-style entry and an optional emotion, and the model generates an appropriate affirmation.

### What you get:
* A generative language model that outputs affirmations based on emotional input
* An interactive Gradio interface for testing
* Batch generation & evaluation scripts for large-scale inference

---

## What You Can Use This For
* Mental health & wellness journaling tools  
* Chatbot enhancement with emotion-sensitive replies  
* AI-driven encouragement apps  
* Educational tools for emotional literacy

---

## Exploratory Data Analysis (EDA)
This project includes several visualizations that explored the:
* Distribution of emotion labels  
* Most common affirmation tags  
* Input vs. output text lengths  

EDA performed in `eda_paired_dataset.ipynb` — run using [Quest Jupyter Notebook Guide](https://services.northwestern.edu/TDClient/30/Portal/KB/ArticleDet?ID=1791)

---

## Model Architecture
* **Base Model**: GPT-2 (small)  
* **Conditioning**: Optional emotion tag (e.g., "[GRATITUDE]") prepended  
* **Loss Function**: Causal Language Modeling with CrossEntropyLoss  
* **Training Epochs**: 35  
* **Dataset Size**: 114,000+ tweet-affirmation pairs  

---

## Folder Structure

```
.
├── LICENSE
├── README.md
├── configs
│   └── config.yaml
├── data
│   ├── batch_inputs.csv
│   ├── combine_affirmations_tweets_datasets.py
│   ├── data
│   │   ├── og-emotion-detection-tweets-dataset
│   │   │   ├── 1
│   │   │   │   └── tweet_emotions.csv
│   │   │   └── tweet_emotions.csv
│   │   ├── og-positive-affirmations-dataset
│   │   │   ├── 2
│   │   │   │   └── positive_affirmations
│   │   │   │   └── possitive_affirmation.csv
│   │   │   └── positive_affirmations
│   │   │   └── possitive_affirmation.csv
│   │   └── paired_affirmations.csv
│   │   └── sample_preview.csv
│   ├── eda_affirmations.ipynb
│   ├── eda_paired_dataset.ipynb
│   ├── eda_tweets.ipynb
│   ├── sample_preview.csv
│   ├── test.csv
│   ├── train.csv
│   └── val.csv
├── gui.py
├── model
│   └── train_gpt2.py
├── requirements.txt
├── run_gui.sh
├── run_project.sh
├── scripts
│   ├── evaluation.py
│   ├── generate_batch.py
│   ├── generate_inference_affirmation.py
│   └── split_dataset.py
├── setup_env.sh 
└── submit_project.sh


```

---

## Accessing and Running on Quest

### 1. Log into Quest
```bash
ssh -X your_netid@login.quest.northwestern.edu
```

### 2. Clone the Repo and Setup
```bash
git clone https://github.com/hanna8008/affirmation_generator.git
cd affirmation_generator
bash setup_env.sh
conda activate affirmgen
```

### 3. Run GUI
```bash
bash run_gui.sh
```

---

## Extra Criteria - GUI Overview

The GUI (built with Gradio) supports:

* Free-text input (journal-style)
* Optional emotion tag
* Real-time output generation

---

## Sample Generated Affirmations

| Input                             | Emotion   | Affirmation                                               |
| --------------------------------- | --------- | --------------------------------------------------------- |
| I'm feeling stuck and overwhelmed |           | I trust that everything is unfolding for my highest good. |
| I'm grateful for my progress      | gratitude | I honor how far I’ve come and welcome all that's ahead.   |

---

## Training Loss Graph

![Training Loss](outputs/logs/train_loss.png)

---

## Data Preparation & Transfer

* Combined tweet+affirmation data into `paired_affirmations.csv`
* Cosine similarity computed using sentence-transformers
* Transferred to Quest using `scp` or GitHub

---

## Model Training on Quest (Northwestern Quest)

Run the following:
```bash
bash submit_project.sh
```

Checkpoints saved in `outputs/checkpoints/`, logs in `outputs/logs/`

---

## Final Scripts I Will Run

✅ **EDA Notebook** (run via CLI):
```bash
jupyter nbconvert --to notebook --execute data/eda_paired_dataset.ipynb --output outputs/eda/eda_paired_dataset.ipynb
```

✅ **Batch Generation**:
```bash
python scripts/generate_batch.py \
  --input data/batch_inputs.csv \
  --config configs/config.yaml \
  --log_output outputs/generated/generated_affirmations.csv
```

✅ **Evaluation**:
```bash
python scripts/evaluation.py \
  --predicted outputs/generated/generated_affirmations.csv \
  --reference data/batch_inputs.csv \
  --save_path results/eval_metrics.json
```

---

## Professor Instructions

After logging into Quest, please run only:
```bash
conda activate affirmgen
bash run_gui.sh
```

The Gradio interface will open automatically with the fine-tuned model loaded.  
No additional training, installation, or evaluation steps are needed.

---

## Future Improvements

* Expand emotion conditioning with multi-label support
* Integrate journaling streak tracker
* Add metric dashboard (BLEU, ROUGE, Cosine)
* Deploy to HuggingFace Spaces for web access

---

## References and Tools Used

1. [Hugging Face Transformers](https://huggingface.co/transformers/)
2. [Sentence Transformers](https://www.sbert.net/)
3. [Gradio](https://gradio.app/)
4. [Northwestern Quest](https://www.it.northwestern.edu/departments/it-services-support/research/computing/quest/index.html)