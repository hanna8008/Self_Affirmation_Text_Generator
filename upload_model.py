from huggingface_hub import upload_folder

upload_folder(
    folder_path="outputs/checkpoints/checkpoint-389640",
    path_in_repo=".",
    repo_id="hanna8008/affirmation-gpt2",
    repo_type="model"
)

