# ----------------------------------------------------------------------------
# evaluation.py
# ----------------------------------------------------------------------------



# --- Imports ---
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_simiarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer



# --- Load Results ---
df = pd.read_csv("results/batch_affirmations.csv")



# --- Load Pretrained Sentence Transformer ---
model = SentenceTransformer("all-MiniLM-L6-v2")



# --- Get Sentence Embeddings ---
input_embeddings = model.encode(df["Input"].tolist(), convert_to_tensor=True)
output_embeddings = model.encode(df["Affirmation"].tolist(), convert_to_tensor=True)



# --- Compute Cosine Similarity ---
similarities = cosine_similarity(input_embeddings.cpu().numpy(), output_embeddings.cpu().numpy())
df["CosineSimilarity"] = np.diag(similarities)



# --- BLEU + ROUGE ---
bleu_scores = []
rouge_1_scores = []
rouge_L_scores = []

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
smooth = SmoothingFunction().method1

for ref, gen in zip(df["Input"], df["Affirmation"]):
    ref_tokens = ref.split()
    gen_toekns = gen.split()

    #BLEU
    bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smooth)
    bleu_scores.append(bleu)

    #ROUGE
    rouge = scorer.score(ref, gen)
    rouge_1_scores.append(rouge["rogue1"].fmeasure)
    rouge_L_scores.append(rouge["rougeL"].fmeasure)

df["BLEU"] = bleu_scores
df["ROUGE-1"] = rouge_1_scores
df["ROUGE-L"] = rouge_L_scores



# --- Print Summary Stats ---
avg_sim = df["CosineSimilarity"].mean()
print(f"Average Cosine Similarity: {df['CosineSimilarity'].mean():.4f}")
print(f"Average BLEU Score: {df['bleu'].mean():.4f}")
print(f"Average ROUGE-1: {df['ROUGE-1'].mean():.4f}")
print(f"Average ROUGE-L: {df['ROUGE-L'].mean():.4f}")



# --- Save to New File ---
df.to_csv("results/batch_affirmations_evaluated.csv", index=False)