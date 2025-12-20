import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import random
import time

data_path = "/home/cc/Amazon-Product-Review-Sentiment-Analysis-using-RNN-Dataset.csv"
df = pd.read_csv(data_path)

# Check class distribution
#print(df["Sentiment"].value_counts())

# ---- BALANCE THE DATA INTO TWO EQUAL TABLES ---- #

# 1. Group by sentiment
groups = {label: group for label, group in df.groupby("Sentiment")}

# 2. Find the minimum class count (so all classes contribute equally)
min_count = min(len(g) for g in groups.values())

#print("Minimum class size:", min_count)

# 3. Sample each sentiment class equally
balanced_df = pd.concat([g.sample(min_count, random_state=42) for g in groups.values()],
                        ignore_index=True)

# 4. Shuffle
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Split into two equal tables
mid = len(balanced_df) // 2
df_part1 = balanced_df.iloc[:mid]
df_part2 = balanced_df.iloc[mid:]

# print("Part 1 shape:", df_part1.shape)
# print("Part 2 shape:", df_part2.shape)
# print(df_part1["Sentiment"].value_counts())
# print(df_part2["Sentiment"].value_counts())

# import pandas as pd

# data_path = "/home/cc/Amazon-Product-Review-Sentiment-Analysis-using-RNN-Dataset.csv"
# df = pd.read_csv(data_path)

# Column with class labels
label_col = "Sentiment"

# How many from each class
N_PER_CLASS = 20

small_parts = []
remaining_parts = []

def split_n_per_class(df, label_col, n_per_class=20, seed=42):
    small_parts = []
    remaining_parts = []

    for label, group in df.groupby(label_col):
        # select 20 samples from each class
        if len(group) >= n_per_class:
            small = group.sample(n=n_per_class, random_state=seed)
        else:
            small = group  # if fewer than 20 exist
        remaining = group.drop(small.index)

        small_parts.append(small)
        remaining_parts.append(remaining)

    df_small = pd.concat(small_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    df_rest = pd.concat(remaining_parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    return df_small, df_rest
# df1_20, df1_remaining = split_n_per_class(df_part1, "Sentiment", 20)

# # For table 2
# df2_20, df2_remaining = split_n_per_class(df_part2, "Sentiment", 20)

# For table 1
df1_20, df1_remaining = split_n_per_class(df_part1, "Sentiment", 20)
df1_20.reset_index(drop=True, inplace=True)
df1_remaining.reset_index(drop=True, inplace=True)

# For table 2
df2_20, df2_remaining = split_n_per_class(df_part2, "Sentiment", 20)
df2_20.reset_index(drop=True, inplace=True)
df2_remaining.reset_index(drop=True, inplace=True)

print("Table 1 - 20-per-class index:", df1_20.index[:5])
print("Table 1 - remaining index:", df1_remaining.index[:5])
print("Table 2 - 20-per-class index:", df2_20.index[:5])
print("Table 2 - remaining index:", df2_remaining.index[:5])


# print("Table 1 - 20-per-class:", df1_20.shape)
# print(df1_20["Sentiment"].value_counts())

# print("\nTable 1 - remaining:", df1_remaining.shape)
# print(df1_remaining["Sentiment"].value_counts())

# print("\nTable 2 - 20-per-class:", df2_20.shape)
# print(df2_20["Sentiment"].value_counts())

# print("\nTable 2 - remaining:", df2_remaining.shape)
# print(df2_remaining["Sentiment"].value_counts())

Encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

a = list(df1_20["Review"])
b = list(df2_20["Review"])
print(len(a), len(b))
# Build the two collections as strings (if you still need them for prompting)
string_1 = "Text Collection 1:\n"
for i in range(1, len(a) + 1):
    string_1 += f"{i}. {a[i - 1]}\n"

string_2 = "Text Collection 2:\n"
for i in range(1, len(b) + 1):
    string_2 += f"{i}. {b[i - 1]}\n"


prompt_same = "Find indexes x,y where x is the number of an entry \
in collection 1 and y the number of an entry in \
collection 2 such that they fall under same sentiment category (make sure to catch \
all pairs! e.g. 1,1 or 2,2 or 3,3 or 4,4 or 5,5)! \
Separate index pairs by semicolons. \
Write \"Finished\" after the last pair! \n" + string_1 + "\n " + string_2 + "\n" + "Index pairs:"


# API_KEY = ""
# start = time.time()
# client = OpenAI(api_key=API_KEY)

# response = client.responses.create(
#     model="gpt-4o",
#     input=prompt_same
# )
# end = time.time()
# print(response.output_text)
# print(response.usage)
# print(f"{end-start}")

start = time.time()
#pair_str = "1,1; 2,2; 3,3; 4,4; 5,5; 6,9; 7,10; 8,7; 9,9; 10,10;"

pair_str = "1,1; 1,8; 1,15; 2,5; 3,12; 3,15; 4,5; 5,5; 6,14; 6,36; 7,2; 7,12; 7,40; 8,3; 9,6; 9,40; 10,14; 10,36; 11,6; 11,40; 12,36; 13,15; 13,94; 13,100; 14,3; 15,29; 17,94; 18,67; 18,79; 19,40; 21,40; 22,3; 22,94; 22,100; 25,15; 25,79; 33,44; 33,97; 35,92; 37,3; 47,16; 47,50; 48,40; 50,14; 50,29; 51,14; 51,36; 53,14; 53,36; 55,34; 61,90; 62,91; 65,79; 67,66; 70,94; 72,14; 73,40; 75,15; 75,79; 91,94; 93,90; 93,94; 95,31; 95,85; 98,36; 98,83;"

all_pairs = []
for o in pair_str.split(";"):
    o = o.strip()
    if o:
        all_pairs.append(o)

positive_pairs = []
for item in all_pairs:
    item = item.strip()
    if not item:
        continue
    i_str, j_str = item.split(",")
    i = int(i_str) - 1
    j = int(j_str) - 1
    positive_pairs.append((i, j))

print("Number of positive pairs:", len(positive_pairs))

train_embeddings = Encoder.encode(df1_20["Review"], batch_size=64, show_progress_bar=True)
test_embeddings = Encoder.encode(df2_20["Review"], batch_size=64, show_progress_bar=True)
table_1_arr = np.asarray(train_embeddings, dtype=np.float32)
table_2_arr = np.asarray(test_embeddings, dtype=np.float32)
pos_set = set(positive_pairs)

X_pos = []
y_pos = []
for i, j in positive_pairs:
    #print(i,j)
    emb_a = table_1_arr[i]
    emb_b = table_2_arr[j]
    pair_emb = np.concatenate([emb_a, emb_b], axis=0)
    X_pos.append(pair_emb)
    y_pos.append(1)

num_pos = len(X_pos)

# Build negative examples by sampling pairs not in pos_set
all_neg_candidates = []
for i in range(len(table_1_arr)):
    for j in range(len(table_2_arr)):
        if (i, j) not in pos_set:
            all_neg_candidates.append((i, j))

random.shuffle(all_neg_candidates)
neg_pairs = all_neg_candidates[:num_pos]

X_neg = []
y_neg = []
for i, j in neg_pairs:
    emb_a = table_1_arr[i]
    emb_b = table_2_arr[j]
    pair_emb = np.concatenate([emb_a, emb_b], axis=0)
    X_neg.append(pair_emb)
    y_neg.append(0)

# Combine positives + negatives
X = np.vstack(X_pos + X_neg)   # shape: (2 * num_pos, 2d)
y = np.array(y_pos + y_neg, dtype=np.float32)
print("Dataset shape:", X.shape, y.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

print("Train/test sizes:", X_train.shape, X_test.shape)
input_dim = X_train.shape[1]

# =======================
# MLP model
# =======================

class PairMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # binary logit
        )
        self.deeper_net = nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.1),

        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.1),

        nn.Linear(256, 128),
        nn.ReLU(),

        nn.Linear(128, 64),
        nn.ReLU(),

        nn.Linear(64, 32),
        nn.ReLU(),

        nn.Linear(32, 1),  # binary logit
    )


    def forward(self, x):
        return self.net(x).squeeze(1)  # (N,)


model = PairMLP(input_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)


# # =======================
# # Training loop
# # =======================

n_epochs = 250
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            preds = (torch.sigmoid(test_logits) > 0.5).float()
            acc = (preds == y_test).float().mean().item()
        print(f"Epoch {epoch + 1:3d} | Loss: {loss.item():.4f} | Test Acc: {acc:.3f}")

# =======================
# Helper to score a single pair
# =======================



#table_1_arr = np.load("table_1_arr.npy")
#table_2_arr = np.load("table_2_arr.npy")

train_embeddings = Encoder.encode(df1_remaining["Review"], batch_size=64, show_progress_bar=True)
test_embeddings = Encoder.encode(df2_remaining["Review"], batch_size=64, show_progress_bar=True)
table_1_arr = np.asarray(train_embeddings, dtype=np.float32)
table_2_arr = np.asarray(test_embeddings, dtype=np.float32)

def score_pair(i: int, j: int) -> float:
    """
    i, j are 0-based indices into table_1_arr and table_2_arr.
    Returns probability that the pair is in the positive class.
    """
    emb_a = table_1_arr[i]
    emb_b = table_2_arr[j]
    pair_emb = np.concatenate([emb_a, emb_b], axis=0)
    x = torch.from_numpy(pair_emb).float().unsqueeze(0)  # (1, 2d)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()
    #return prob
    return 1 if prob >= 0.5 else 0

#print("Example P(1,3 is positive/same):", score_pair(0, 2))
#print("Example P(5,1 is negative/different):", score_pair(4, 0))
#print(df_test.head(5))
#print(df_train.head(5))

# same sentiment evaluation on rows after `length`
y_true = []
y_pred = []
for idx in range(100):
    if idx % 20 == 0:
        print("Index:", idx)
    for idx2 in range(100):
        #print("Score same sentiment P:", score_pair(idx, idx))
        tab_1_label = df1_remaining.loc[idx, "Sentiment"]
        tab_2_label = df2_remaining.loc[idx, "Sentiment"]
        pred = score_pair(idx, idx2)
        y_pred.append(pred)
        if tab_1_label == tab_2_label:
            actual = 1
            y_true.append(actual)
            
        else:
            actual = 0
            y_true.append(actual)

end = time.time()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# #### Values over (500x500) comparisons
# # Accuracy: 0.407108
# # Precision: 0.9667724211095209
# # Recall: 0.39998757763975157
# # F1 Score: 0.5658593450238272
# # ### need to compute acc, precision, recall, F1 on remainder of data after length


#ResponseUsage(input_tokens=19456, input_tokens_details=InputTokensDetails(cached_tokens=0), output_tokens=331, output_tokens_details=OutputTokensDetails(reasoning_tokens=0), total_tokens=19787)
#13.793205976486206

# total_time = end-start + 2.6394989490509033
# print(total_time)
