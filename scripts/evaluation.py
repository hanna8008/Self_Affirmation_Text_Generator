# ----------------------------------------------------------------------------
# evaluation.py
# ----------------------------------------------------------------------------
#
# This script evaluates the quality of generated affirmations by computing semantic
# similarity (Cosine Similarity via Sentence Transformers), lexical overlap (BLEU),
# and content-based overlap (ROUGE-1 and ROUGE-L). It takes an input CSV of 
# Input-Affirmation pairs and saves a new CSV with added scores.



# --- Imports ---
import pandas as pd
import sys
import numpy as np
#for semantic similarity embeddings
from sentence_transformers import SentenceTransformer
#for computing similarity between embeddings
from sklearn.metrics.pairwise import cosine_similarity
#for BLEU score evaluation
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
#for ROUGE score evaluation
from rouge_score import rouge_scorer



# --- Load Results ---
#load generated affirmations and their original input text
"""df = pd.read_csv("results/batch_affirmations.csv")"""
#load input file (CSV) containing 'Input' and 'Affirmation' columns
df = pd.read_csv(sys.argv[1])
#output file path to save evaluation results
output_file = sys.argv[2]



# --- Load Pretrained Sentence Transformer ---
#load sentence transformer for semantic similarity (cosine), lightweight but high-quality model for sentence embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")



# --- Get Sentence Embeddings ---
#encode original prompts
input_embeddings = model.encode(df["Input"].tolist(), convert_to_tensor=True)
#encode model's affirmations
output_embeddings = model.encode(df["Affirmation"].tolist(), convert_to_tensor=True)



# --- Compute Cosine Similarity ---
#compare embeddings, a matrix of all pairwise similarities
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

#initialize ROUGE scorer, with stemming
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#apply smoothing to BLEU score to avoid zero scores in short texts
smooth = SmoothingFunction().method1



# --- Evaluate Each Pair ---
#iterate over each input/affirmation pair
for ref, gen in zip(df["Input"], df["Affirmation"]):
    #tokenize reference (user input)
    ref_tokens = ref.split()
    #tokenize generated output (affirmation)
    gen_tokens = gen.split()

    #BLEU
    #compute BLEU
    bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smooth)
    bleu_scores.append(bleu)

    #ROUGE
    #compute ROUGE-1 and ROUGE-L
    rouge = scorer.score(ref, gen)
    rouge_1_scores.append(rouge["rouge1"].fmeasure)
    rouge_L_scores.append(rouge["rougeL"].fmeasure)



# --- Store Scores in DataFrame ---
#add BLEU scores to DataFrame
df["BLEU"] = bleu_scores
#add ROUGE-1 scores (unigram overlap)
df["ROUGE-1"] = rouge_1_scores
#add ROUGE-L scores (longest common subsequence overlap)
df["ROUGE-L"] = rouge_L_scores



# --- Print Summary Stats: Average of Each Score ---
avg_sim = df["CosineSimilarity"].mean()
print(f"Average Cosine Similarity: {df['CosineSimilarity'].mean():.4f}")
print(f"Average BLEU Score: {df['BLEU'].mean():.4f}")
print(f"Average ROUGE-1: {df['ROUGE-1'].mean():.4f}")
print(f"Average ROUGE-L: {df['ROUGE-L'].mean():.4f}")



# --- Print Summary Stats: Best/Maximum of Each Score ---
print(f"Best Cosine Similarity: {df['CosineSimilarity'].max():.4f}")
print(f"Best BLEU Score: {df['BLEU'].max():.4f}")
print(f"Best ROUGE-1: {df['ROUGE-1'].max():.4f}")
print(f"Best ROUGE-L: {df['ROUGE-L'].max():.4f}")



# --- Save to New File ---
#save all scores with outputs
"""df.to_csv("results/batch_affirmations_evaluated.csv", index=False)"""
#save the DataFrame with all new evaluation columns
df.to_csv(output_file, index=False)