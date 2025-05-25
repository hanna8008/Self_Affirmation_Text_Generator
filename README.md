# Self_Affirmation_Text_Generator

## Completed as of Saturday, May 25:
1. Built and ran `combine_affirmations_tweets_datasets.py` to generate `paired_affirmations.csv`
2. Applied semantic similarity filtering, emotion-to-tag mapping, and additional quality filters
3. Installed all required dependencies, including `sentence-transformers`, `transformers`, and `accelerate`
4. Created and validated the scripts: `run_project.sh`, `submit_project.sh`, and `setup_env.sh`
5. Finalized `requirements.txt` with pinned package versions for reproducibility
6. Reinstalled and configured Miniconda, created `affirmgen` Conda environment, and fixed PATH/env issues
7. Completed `eda_paired_dataset.ipynb` — included visualizations for sentence lengths, emotion/tag distribution, etc.
8. Wrote `train_gpt2.py` for full GPT-2 fine-tuning pipeline
9. Set up and tested `configs/config.yaml` with all hyperparameters and data paths
10. Enabled TensorBoard logging via `report_to="tensorboard"` in `TrainingArguments`
11. Submitted full training job (114k examples, 35 epochs) on Quest using A100 GPU
12. Validated SLURM logging, checkpoint saving, and best model selection with `load_best_model_at_end=True`

---

## Remaining Tasks

### Model Inference
1. **generate_affirmation.py**
   - Load trained GPT-2 model from best checkpoint
   - Accept emotional input (tweet, journal entry, etc.)
   - Generate and return an affirmation response
   - Log or save output for sample evaluation

### Gradio GUI
2. **gui.py**
   - Input box for free-text journal input
   - Optional dropdown for mood tag
   - Generate button
   - Display box for predicted affirmation

### Report & Evaluation
3. Final deliverables:
   - Summarize key EDA visualizations
   - Evaluate quality of generated affirmations
   - Compare sample generations across tags/moods
   - Write project report: methodology, results, reflections
   - Polish project folder structure and remove temp/debug code
   - Finalize this README
   - Optional: implement evaluation metrics (BLEU, cosine similarity, etc.)

---

## Notes
- Current model uses full paired dataset (114,601 examples)
- Model is training for 35 epochs with per-epoch evaluation and checkpointing
- TensorBoard enabled — live monitoring of loss curves during training
- Model expected to be ready for inference testing by Sunday night


---


## Files

### train_4295606.log
* Paired Dataset Summary (paired_affirmations.csv)
Successfully paired tweet sentimnets with relevant affirmation tags using keyword heuristics, emotion-to-tag mapping, and semantic similarity scoring

---

##### 1. Affirmation Tag Distribution
* Dataset includes 114,601 tweet-affirmation pairs categorized into 9 tags
* Most common tags: love, blessing, happiness, gratitude, and spiritual
* Less frequent tags: health, beauty, money, sleep (useful for class balancing strategies)

##### 2. Missing Value Check
* All columns (Input, Output, Emotion_Label, Affirmation_Tag) are complete
* Confirms no missing values - data is clean and ready for training
* Ensures reliable input for model training without imputation or filtering

##### 3. Character Count Distribution
* Input (Tweets)
    - Average tweet length: 70.9 characters
    - Tweets range from 1 to 167 characters, with 50% under 65 characters
    - Supports short-form, conversational text of social media
* Output (Affirmations)
    - Average affirmation length: 40.8 characters
    - Affirmations range from 10 to 79 characters, with half being under 41 characters
    - Consistent with concise, punchy affirmation formatting suitable for GPT-2 style generation


# Don't forget to include how I saw tensorboard