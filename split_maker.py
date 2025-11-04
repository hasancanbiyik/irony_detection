import pandas as pd
from sklearn.model_selection import train_test_split

# Load your balanced dataset (change maybe needed!)
df = pd.read_csv("/Users/hasancan/Desktop/irony_detection/datasets/balanced_dataset_adjusted.csv")

X = df["text"]
y = df["label"]

# 80% train, 20% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Split the 20% into validation (10%) and test (10%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Combine back into DataFrames
train_df = pd.DataFrame({"text": X_train, "label": y_train})
val_df   = pd.DataFrame({"text": X_val, "label": y_val})
test_df  = pd.DataFrame({"text": X_test, "label": y_test})

# Save splits
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")
print(train_df['label'].value_counts(), "\n")
print(val_df['label'].value_counts(), "\n")
print(test_df['label'].value_counts())

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    fold_train = df.iloc[train_idx]
    fold_test  = df.iloc[test_idx]

    # Split training fold into train/val (90/10, stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        fold_train["text"], fold_train["label"],
        test_size=0.1, stratify=fold_train["label"],
        random_state=fold
    )

    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    val_df   = pd.DataFrame({"text": X_val, "label": y_val})
    test_df  = fold_test.copy()

    train_df.to_csv(f"train_{fold+1}.csv", index=False)
    val_df.to_csv(f"val_{fold+1}.csv", index=False)
    test_df.to_csv(f"test_{fold+1}.csv", index=False)

    print(f"Fold {fold+1}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print("Train label balance:\n", train_df['label'].value_counts())
    print("Val label balance:\n", val_df['label'].value_counts())
    print("Test label balance:\n", test_df['label'].value_counts())
    print("-" * 50)


# EXAMPLE OUTPUT
# Fold 1: train=529, val=59, test=66
Train label balance:
#  label
# 1    265
# 0    264
# Name: count, dtype: int64
# Val label balance:
#  label
# 0    30
# 1    29
# Name: count, dtype: int64
# Test label balance:
#  label
# 1    33
# 0    33
# Name: count, dtype: int64

