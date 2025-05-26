# ----------------------------------------------------------------------------
# split_dataset.py
# ----------------------------------------------------------------------------
#
# This script splits the cleaned, paired dataset into train, validation, and
# test sets. The splits are 80% training, 10% validation, and 10% test. It saves
# each subset to separate CSV files for use in model training and evaluation



# --- Imports ---
import pandas as pd
from sklearn.model_selection import train_test_split



# --- Load the Full Dataset ---
#load the full dataset of paired input-output affirmations
df = pd.read_csv("data/paired_affirmations.csv")



# --- Split into train (80%) and temp (20%) ---
#first, hold out 20% of the data into a temporary set (which will later become val + test)
train_df, temp_df = train_test_split(
    df, 
    test_size=0.2, 
    #ensures reproducibility
    random_state=42
)



# --- Split temp into validation (10%) and test 10%)
#split the temporary 20% evently into 10% validation and 10% test sets
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)



# --- Save splits ---
#export each DataFrame to CSV for downstream use by training and evaluation scripts
#save 80% training data
train_df.to_csv("data/train.csv", index=False)
#save 10% validation data
val_df.to_csv("data/val.csv", index=False)
#save 10% test data
test_df.to_csv("data/test.csv", index=False)



# --- Output ---
#print confirmation and size of each dataset split
print("Split Complete! Files saved in /data:")
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")