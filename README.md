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
* [Future Improvements](#future-improvements)
* [References and Tools Used](#references-and-tools-used)

---

## Overview

**Mirror Me** is an AI-powered affirmation generator that takes in journal-style text and optionally an emotional label (e.g., "gratitude", "worry") to generate a comforting, self-affirming response using a fine-tuned GPT-2 language model.

This tool was created for individuals looking to explore mindfulness, journaling, and positive self-talk — as well as researchers exploring emotion-to-text conditioning.

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

> All EDA was performed in `eda_paired_dataset.ipynb` and outputs saved in `outputs/eda/`

---

## Model Architecture

* **Base Model**: GPT-2 (small)
* **Conditioning**: Optional text tag (e.g., "\[GRATITUDE]") prepended
* **Loss Function**: Causal Language Modeling (CLM) with CrossEntropyLoss
* **Training Epochs**: 35
* **Trained On**: 114,000+ tweet-affirmation pairs

---

## Folder Structure

```
├── configs/
│   └── config.yaml                     # Training config file
├── data/
│   ├── paired_affirmations.csv        # Full training dataset
│   ├── train.csv / val.csv / test.csv # Pre-split datasets
│   └── batch_inputs.csv               # Input file for batch generation
├── outputs/
│   ├── checkpoints/                   # Model weights
│   ├── generated/                     # Sample affirmations
│   └── logs/                          # Training logs + loss curves
├── scripts/
│   ├── train_gpt2.py                  # Fine-tuning script
│   ├── generate_batch.py              # Batch generation from CSV
│   ├── evaluation.py                  # Cosine similarity evaluation
├── generate_inference_affirmation.py  # Inference script
├── gui.py                              # Gradio interface
├── run_gui.sh                          # Bash launcher for GUI
├── setup_env.sh                        # Conda setup script
├── submit_project.sh                   # Quest SLURM job script
├── requirements.txt                    # Python dependencies
```

---

## Accessing and Running on Quest

### 1. Log into Quest

```bash
ssh -X your_netid@login.quest.northwestern.edu
```

### 2. Clone the Repo into Quest

```bash
git clone https://github.com/yourusername/affirmation_generator.git
cd affirmation_generator
```

### 3. Setup Conda Environment (First Time Only)

```bash
bash setup_env.sh
```

### 4. Activate Environment

```bash
conda activate affirmgen
```

### 5. Run the GUI

```bash
bash run_gui.sh
```

### 6. Access via Browser Link from Gradio

Watch terminal output for a link beginning with:

```bash
Running on public URL: https://...
```

---

## Extra Criteria - GUI Overview

The GUI is built with Gradio and supports:

* Free-text input (journal-style)
* Optional emotion tag input
* Button click generation
* Real-time display of affirmation

---

## Sample Generated Affirmations

| Input                             | Emotion   | Affirmation                                               |
| --------------------------------- | --------- | --------------------------------------------------------- |
| I'm feeling stuck and overwhelmed | worry     | I trust that everything is unfolding for my highest good. |
| I'm grateful for my progress      | gratitude | I honor how far I’ve come and welcome all that's ahead.   |

---

## Training Loss Graph

The following chart shows validation loss over 35 epochs.
![Training Loss](outputs/logs/train_loss.png)

---

## Data Preparation & Transfer

* Combined emotional tweets + affirmations into `paired_affirmations.csv`
* Used sentence-transformers for cosine similarity pairing
* Transferred data using `scp` or Git sync

---

## Model Training on Quest (Northwestern Quest)

Used A100 GPU node with SLURM job:

```bash
bash submit_project.sh
```

Model outputs stored in:

* `outputs/checkpoints/`
* `outputs/logs/`
* `outputs/generated/`

---

## Future Improvements

* Add BLEU/ROUGE metrics
* Expand emotion conditioning with multi-label support
* Integrate daily journaling tracker

---

## References and Tools Used

1. [Hugging Face Transformers](https://huggingface.co/transformers/)
2. [Sentence Transformers](https://www.sbert.net/)
3. [Gradio](https://gradio.app/)
4. [Quest HPC @ Northwestern](https://www.it.northwestern.edu/departments/it-services-support/research/computing/quest/index.html)