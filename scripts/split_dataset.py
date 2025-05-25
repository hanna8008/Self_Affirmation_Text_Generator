# ----------------------------------------------------------------------------
# split_dataset.py
# ----------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split

# --- Load the Full Dataset ---
df = pd.read_csv("../data/paired_affirmations.csv")

# --- Split into train (80%) and temp (20%) ---
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# --- Split temp into validation (10%) and test 10%)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# --- Save splits ---
train_df.to_csv("../data/train.csv", index=False)
val_df.to_csv("../data/val.csv", index=False)
test_df.to_csv("../data/test.csv", index=False)

# --- Output ---
print("Split Complete! Files saved in /data:")
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")