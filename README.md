# Self_Affirmation_Text_Generator

Already Done as of Saturday, May 24:
1. build and ran combine_affirmations_tweets_datasets.py to generate paired_affirmations.csv
2. added semantic similarity, emotion-to-tag mapping, and filters to improve data quality
3. installed sentence-transformers, kagglehub, and other dependencies
4. created the run_project.sh, submit_project.sh, adn setup_env.sh scripts
5. created requirements.txt
6. removed and reinstalled Miniconda and fixed PATH/env activation issues, plus created affirmgen environment
7. beginning eda_paired_dataset.ipynb



## Left To Do:
### Model Training
1. Training Setup
    * Write configs/configs.yaml with:
        - model (GPT-2 small)
        - tokenzier
        - learning rate, batch size, epoch
        - data paths (train/val/test)

    * train_gpt2.py
        - load config
        - load and tokenize paired data
        - fine-tune GPT-2 using HuggingFace Trainer
        - Log using TensorBoard (or optionally W&B)
        - save checkpoints

### Text Generation + GUI
2. Inference
    * generate_affirmation.py
        - load trained model
        - take an emotional input (optional emotional label)
        - generate affirmation text
        - log output to console or save

3. Gradio Chatbot GUI
    * gui.py
        - input box for journal-style text (max characters for input)
        - dropdown for optional mood label
        - button to generate
        - display output affirmation

### Report & Polish
* EDA visualizations summary (charts for sentence length, tag distribution, emotion distribution)
* evaluate sample generations
* clean project structure
* add README
* add comments into code
* optional: model evaluation script (BLEU, cosine similarity)