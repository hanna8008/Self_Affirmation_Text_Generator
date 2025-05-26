# ----------------------------------------------------------------------------
# combine_affirmations_tweets_datasets.py
# ----------------------------------------------------------------------------
#
# This script downloads and preprocesses affirmations and emotional tweets datasets.
# It assigns tags to tweets using keywords and emotion mappings, then uses semantic
# similarity to pair each tweet with a relevant affirmation. The resulting dataset
# is saved for use in fine-tuning the text generation model.



# --- Imports ---
import os
import re
import pandas as pd
import numpy as np
import kagglehub
import shutil
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util



# --- Download Datasets from KaggleHub ---
print("Downloading datasets from Kagglehub...")
#download affirmation dataset
path_affirmations = kagglehub.dataset_download("pratiksharm/positive-affirmations-with-tags")
#download emotion tweets dataset
path_tweets = kagglehub.dataset_download("pashupatigupta/emotion-detection-from-text")



# -- Move and Rename Datasets ---
#destination folder for affirmations
affirmations_new_path = "data/og-positive-affirmations-dataset"
#destination folder for tweets
tweets_new_path = "data/og-emotion-detection-tweets-dataset"
#ensure that "data" folder exists
os.makedirs("data", exist_ok=True)

#move tweet dataset if it hasn't already been moved
if not os.path.exists(affirmations_new_path):
    shutil.move(path_affirmations, affirmations_new_path)
else:
    print(f"Folder already exists: {affirmations_new_path}. Skipping move.")

if not os.path.exists(tweets_new_path):
    shutil.move(path_tweets, tweets_new_path)
else:
    print(f"Folder already exists: {tweets_new_path}. Skipping move.")  

print("Affirmations dataset moved to:", affirmations_new_path)
print("Tweets dataset moved to:", tweets_new_path)



# --- Load Datasets ---
print("Loading datasets...")
#load both affirmation CSV files (some are misname or encoded differently )
affirmations_df_1 = pd.read_csv(os.path.join(affirmations_new_path, "positive_affirmations"), encoding='latin1', on_bad_lines='skip')
affirmations_df_2 = pd.read_csv(os.path.join(affirmations_new_path, "possitive_affirmation.csv"))
#load tweet dataset
tweets_df = pd.read_csv(os.path.join(tweets_new_path, "tweet_emotions.csv"))



# --- Combine Affirmations Datasets ---
print("Combining affirmations datasets...")
#merge both affirmation files and drop any duplicates
affirmations_df = pd.concat([affirmations_df_1, affirmations_df_2]).drop_duplicates()
#normalize tag column to lowercase and strip spaces
affirmations_df['Tag'] = affirmations_df['Tag'].str.lower().str.strip()



# --- Normalize Sentiment Labels ---
print("Normalizing sentiment labels...")
#normalize tweet sentiments to lowercase and strip whitespace
tweets_df['sentiment'] = tweets_df['sentiment'].str.lower().str.strip()



# --- Emotion-to-Tag Mapping ---
print("Mapping tweet sentinments to affirmation tags...")
#maps each emotion label to a set of possible affirmation tags
emotion_to_tag = {
    'sadness': ['happiness', 'love', 'gratitude', 'blessing'],
    'worry': ['health', 'spiritual'],
    'anger': ['health', 'gratitude'],
    'relief': ['gratitude', 'spiritual'],
    'happiness': ['happiness', 'love', 'blessing'],
    'love': ['love', 'gratitude'],
    'hate': ['love', 'spiritual'],
    'surprise': ['blessing', 'happiness'],
    'empty': ['spiritual', 'gradtitude', 'blessing'],
    'boredom': ['gratitude', 'spiritual'],
    'enthusiasm': ['happiness', 'love'],
    'neutral': ['gratitude', 'spiritual', 'blessing', 'happiness', 'love'],
    'fun': ['happiness', 'love', 'gratitude', 'blessing']
}



# --- Define Keyword-Based Tag Detection (Keyword or Both) ---
#dictionary mapping general topics to associated keywords for content filtering
keyword_map = {
    'money': ['broke', 'poor', 'rent', 'bank', 'debt', 'financial', 'money', 'wealth', 'rich', 'investment', 'savings', 'wallet', 'finance', 'credit', 'cash', 'income', 'salary', 'earnings', 'funds'],
    'sleep': ['tried', 'sleepy', 'insomnia', 'nap', 'awake', "can't sleep", 'restless', 'sleepless', 'sleeping', 'tired', 'exhausted', 'fatigue', 'drowsy', 'slumber', 'doze', 'repose', 'siesta', 'snooze'],
    'beauty': ['ugly', 'acne', 'pimple', 'blemish', 'flaw', 'imperfection', 'skin', 'appearance', 'looks', 'self-esteem', 'confidence', 'attractiveness', 'radiance', 'glow', 'charm', 'allure', 'pretty', 'beautiful', 'mirror', 'look'],
    'health': ['sick', 'illness', 'disease', 'pain', 'injury', 'health', 'wellness', 'fitness', 'exercise', 'nutrition', 'diet', 'medicine', 'treatment', 'recovery', 'therapy', 'rehabilitation', 'unwell', 'headache', 'flu', 'healing'],
    'gratitude': ['thankful', 'appreciation', 'grateful', 'blessing', 'gratitude', 'thankfulness', 'acknowledgment', 'recognition', 'appreciative', 'gratified', 'indebtedness', 'thankfulness', 'appreciate', 'blessed'],
    'spiritual': ['soul', 'pray', 'divine', 'spirit', 'universe', 'faith', 'belief', 'religion', 'soul', 'spirituality', 'meditation', 'prayer', 'sacred', 'divine', 'enlightenment', 'transcendence', 'sacredness', 'worship', 'sacredness']
}



# --- Semantic Similarity Model ---
#load pretrained sentence embedding model for computing semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')



# ---- Define Function to Detect Tags Based on Keywords ---
def assign_tags(text, sentiment):
    #normalize to lowercase
    text_lower = text.lower()
    assigned_tags = []
    #loop through each keyword group
    for tag, keywords in keyword_map.items():
        if any(keyword in text_lower for keyword in keywords):
            assigned_tags.append(tag)
    #if no keywords matched, fall back to tag mapping from sentiment
    if not assigned_tags:
        assigned_tags = emotion_to_tag.get(sentiment, [])
    #remove duplicate tags
    return list(set(assigned_tags))



# --- Tweet Filter ---
def is_clean_tweet(text):
    text = text.lower()
    #filter out tweets that contain URLs, mentions, spam, or social media noise
    return not any(x in text for x in ['@', 'http', 'https', 'www', 'rt', 're:', 're:', 'reply', 'like', 'follow', 'subscribe', 'retweet', 'wtf', '#'])



# --- Expand Tweets Based on Keyword and Emotion Taps ---
print("Expanding tweets with keyword and emotion tags...")
expanded_tweets = []
#loop through all tweets and assign tags (can be multiple per tweet)
for _, row in tweets_df.iterrows():
    tags = assign_tags(row['content'], row['sentiment'])
    for tag in tags:
        new_row = row.copy()
        new_row['Mapped_Tag'] = tag
        expanded_tweets.append(new_row)

#convert to DataFrame
tweets_expanded_df = pd.DataFrame(expanded_tweets)



# --- Tag-Restricted Semantic Pairing ---
print("Joining expanded tweets with affirmations using semantic similarity...")
paired_rows = []
#loop through expanded tweet-tag pairs
for _, row in tqdm(tweets_expanded_df.iterrows(), total=len(tweets_expanded_df)):
    tweet = row['content']
    tag = row['Mapped_Tag']
    sentiment = row['sentiment']
    #filter affirmations with matching tag
    aff_pool = affirmations_df[affirmations_df['Tag'] == tag]
    if aff_pool.empty:
        continue
    #encode tweet and candidate affirmations
    tweet_emb = model.encode(tweet, convert_to_tensor=True)
    candidates = aff_pool['Affirmation'].tolist()
    aff_embs = model.encode(candidates, convert_to_tensor=True)
    #compute cosine similarity and select the best match
    scores = util.cos_sim(tweet_emb, aff_embs)[0]
    best_idx = scores.argmax().item()
    #store the matched pair
    paired_rows.append({
        'Input': tweet,
        'Output': candidates[best_idx],
        'Emotion_Label': sentiment,
        'Affirmation_Tag': tag
    })



# --- Create DataFrame from Paired Rows ---
#create DataFrame from matches
paired_df = pd.DataFrame(paired_rows)
#remove duplicates
paired_df.drop_duplicates(subset=['Input', 'Output'], inplace=True)
#shuffle
paired_df = paired_df.sample(frac=1, random_state=42).reset_index(drop=True)



# --- Save Final Dataset ---
print("Saving final dataset to paired_affirmations.csv...")
paired_df.to_csv("data/paired_affirmations.csv", index=False)
print("Done. Dataset saved to data/paired_affirmations.csv")



# --- Dataset Preview ---
print("\nSample Rows from Paired Dataset:")
#take random 5-row sample
preview_df = paired_df.sample(5, random_state=42)
print(preview_df[['Input', 'Output', 'Emotion_Label', 'Affirmation_Tag']])



# --- Save a Sample CSV ---
#save preview
paired_df.sample(20, random_state=42).to_csv("data/sample_preview.csv", index=False)



# --- Preview Tag Distribution ---
print("\nSample count by Affirmation_Tag:")
#display distribution of tags
print(paired_df['Affirmation_Tag'].value_counts())



# --- Checking for Missing or Empty Values ---
print("\nChecking for missing or empty values in the dataset...")
#check for null values in each column
print(paired_df.isnull().sum())



# --- Character Count Preview ---
print("\nCharacter count in Input and Output columns:")
print("Input column character count:")
#stats on tweet length
print(paired_df['Input'].str.len().describe())
print("Output column character count:")
#stats on affirmation length
print(paired_df['Output'].str.len().describe())