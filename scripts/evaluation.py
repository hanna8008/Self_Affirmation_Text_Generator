# ----------------------------------------------------------------------------
# evaluation.py
# ----------------------------------------------------------------------------



# --- Imports ---
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer



# --- Load Results ---
#load generated affirmations and their original input text
df = pd.read_csv("results/batch_affirmations.csv")



# --- Load Pretrained Sentence Transformer ---
#load sentence transformer for semantic similarity (cosine)
model = SentenceTransformer("all-MiniLM-L6-v2")



# --- Get Sentence Embeddings ---
#encode original prompts
input_embeddings = model.encode(df["Input"].tolist(), convert_to_tensor=True)
#encode model's affirmations
output_embeddings = model.encode(df["Affirmation"].tolist(), convert_to_tensor=True)



# --- Compute Cosine Similarity ---
#compare embeddings
similarities = cosine_similarity(input_embeddings.cpu().numpy(), output_embeddings.cpu().numpy())
#keep only the diagonal (each input with its corresponding output)
df["CosineSimilarity"] = np.diag(similarities)



# --- BLEU + ROUGE ---
#hold BLEU scores for each pair
bleu_scores = []
#tokenize reference
rouge_1_scores = []
#tokenize generated affirmation
rouge_L_scores = []

#initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#apply smoothing to BLEU score
smooth = SmoothingFunction().method1

#iterate over each input-affirmation pair
for ref, gen in zip(df["Input"], df["Affirmation"]):
    #tokenize reference
    ref_tokens = ref.split()
    #tokenize generated affirmation
    gen_tokens = gen.split()

    #BLEU
    bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smooth)
    bleu_scores.append(bleu)

    #ROUGE
    rouge = scorer.score(ref, gen)
    rouge_1_scores.append(rouge["rouge1"].fmeasure)
    rouge_L_scores.append(rouge["rougeL"].fmeasure)

#add BLEU scores to DataFrame
df["BLEU"] = bleu_scores
#add ROUGE-1 scores
df["ROUGE-1"] = rouge_1_scores
#add ROUGE-L scores
df["ROUGE-L"] = rouge_L_scores



# --- Print Summary Stats: Average of Each Score ---
avg_sim = df["CosineSimilarity"].mean()
print(f"Average Cosine Similarity: {df['CosineSimilarity'].mean():.4f}")
print(f"Average BLEU Score: {df['BLEU'].mean():.4f}")
print(f"Average ROUGE-1: {df['ROUGE-1'].mean():.4f}")
print(f"Average ROUGE-L: {df['ROUGE-L'].mean():.4f}")



# --- Print Summary Stats: Best of Each Score ---
print(f"Best Cosine Similarity: {df['CosineSimilarity'].max():.4f}")
print(f"Best BLEU Score: {df['BLEU'].max():.4f}")
print(f"Best ROUGE-1: {df['ROUGE-1'].max():.4f}")
print(f"Best ROUGE-L: {df['ROUGE-L'].max():.4f}")



# --- Save to New File ---
#save all scores with outputs
df.to_csv("results/batch_affirmations_evaluated.csv", index=False)