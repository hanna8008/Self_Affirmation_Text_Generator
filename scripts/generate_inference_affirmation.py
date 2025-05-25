# ----------------------------------------------------------------------------
# generate_inference_affirmation.py
# ----------------------------------------------------------------------------



# --- Imports ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse



# --- Load Model and Tokenizer ---
def load_model_and_tokenizer(model_dir):
    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model



 # --- Generate Affirmation ---
 def generate_affirmation(prompt, tokenizer, model, max_length=100, temperature=0.8, top_k=50, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs_ids = inputs.input_ids

    #use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)
    model.to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length = max_length,
            temperature = temperature,
            top_k = top_k,
            top_p = top_p,
            do_sample = True,
            pad_token_id = tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text



# --- Main Function ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='outputs/checkpoints', help='Path to fine-tuned model')
    parser.add_argument('--prompt', type=str, default=None, help='Input text prompt')
    args = parser.parse_args()

    tokenizer, model = load_model_and_tokenizer(args.model_dir)

    if args.prompt:
        prompt = args.prompt
    else:
        prompt = input("\nEnter a journal-style sentence or emotion: ")

    generated = generate_affirmation(prompt, tokenizer, model)
    print("\n Generated Affirmation \n")
    print(generated)
    print("\n")



# --- main ---
if __name__ == "__main__":
    main()