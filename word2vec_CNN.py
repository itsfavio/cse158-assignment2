import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from gensim.models import Word2Vec
import re
from tqdm import tqdm
import random
import os

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------
# 1) Load data
# ---------------------------
print("Loading data...")
with open("filter_all_t.json", "r") as f:
    data = json.load(f)

train_df = pd.DataFrame(data["train"])
val_df   = pd.DataFrame(data.get("val", []))
test_df  = pd.DataFrame(data.get("test", []))

# Keep only the columns we need
train_df = train_df[['user_id', 'business_id', 'rating', 'review_text', 'history_reviews']]
val_df   = val_df[['user_id', 'business_id', 'rating', 'review_text', 'history_reviews']]
test_df  = test_df[['user_id', 'business_id', 'rating', 'review_text', 'history_reviews']]

# ---------------------------
# 2) Preprocessing helpers
# ---------------------------
def preprocess_text(text):
    """Very fast preprocessing: lowercase, remove non-alpha, split."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    return tokens

# ---------------------------
# 3) Train Word2Vec on TRAIN ONLY (no leakage)
# ---------------------------
print("Preparing Word2Vec training corpus (TRAIN only)...")
train_texts = train_df['review_text'].fillna('').tolist()
sentences = [preprocess_text(t) for t in train_texts if isinstance(t, str) and len(t) > 0]
sentences = [s for s in sentences if len(s) > 3]  # remove very short reviews

print(f"Training Word2Vec on {len(sentences)} sentences (train set only)...")
word2vec = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=3,
    min_count=5,
    workers=4,
    sg=1,
    epochs=8
)

# Build word->idx mapping (PAD=0, UNK=1)
word_to_idx = {'<PAD>': 0, '<UNK>': 1}
for i, word in enumerate(word2vec.wv.key_to_index.keys(), start=2):
    word_to_idx[word] = i
vocab_size = len(word_to_idx)
print(f"Vocab size (with PAD/UNK): {vocab_size}")

# ---------------------------
# 4) Global user/item mappings from TRAIN and UNK indices (cold-start)
# ---------------------------
train_users = train_df['user_id'].unique()
train_items = train_df['business_id'].unique()

user_to_idx_global = {uid: i for i, uid in enumerate(train_users)}
item_to_idx_global = {bid: i for i, bid in enumerate(train_items)}

UNK_USER = len(user_to_idx_global)    # index for unseen users
UNK_ITEM = len(item_to_idx_global)    # index for unseen items

# Sizes for embeddings (add 1 for UNK)
NUM_USERS = len(user_to_idx_global) + 1
NUM_ITEMS = len(item_to_idx_global) + 1

print(f"Num train users: {len(user_to_idx_global)}, num train items: {len(item_to_idx_global)}")
print(f"NUM_USERS (with UNK): {NUM_USERS}, NUM_ITEMS (with UNK): {NUM_ITEMS}")

# ---------------------------
# 5) Dataset (uses GLOBAL mappings; includes review_length)
# ---------------------------
class SimpleRestaurantDataset(Dataset):
    def __init__(self, df, word_to_idx, max_length=100):
        self.df = df.reset_index(drop=True)
        self.word_to_idx = word_to_idx
        self.max_length = max_length

        # Use global mappings created above
        self.user_to_idx = user_to_idx_global
        self.item_to_idx = item_to_idx_global

        # Precompute token indices and review lengths
        self.sequences = []
        self.review_lengths = []
        for text in self.df['review_text'].fillna(''):
            tokens = preprocess_text(text)
            self.review_lengths.append(min(len(tokens), max_length))  # capped length

            tokens = tokens[:max_length]
            indices = [word_to_idx.get(tok, 1) for tok in tokens]  # UNK=1
            if len(indices) < max_length:
                indices = indices + [0] * (max_length - len(indices))
            self.sequences.append(indices)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        u = self.user_to_idx.get(row['user_id'], UNK_USER)
        i = self.item_to_idx.get(row['business_id'], UNK_ITEM)

        return {
            'user_idx': torch.tensor(u, dtype=torch.long),
            'item_idx': torch.tensor(i, dtype=torch.long),
            'text_seq': torch.tensor(self.sequences[idx], dtype=torch.long),
            'review_length': torch.tensor(self.review_lengths[idx], dtype=torch.float32),
            'rating': torch.tensor(row['rating'], dtype=torch.float32)
        }

# ---------------------------
# 6) Model: include review_length
# ---------------------------
import torch.nn.functional as F

class SimpleRecommenderCNN(nn.Module):
    def __init__(self, num_users, num_items, vocab_size, embedding_dim=100):
        super().__init__()
        
        # User & item embeddings
        self.user_emb = nn.Embedding(num_users, 32)
        self.item_emb = nn.Embedding(num_items, 32)
        
        # Word embeddings
        self.word_emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self._init_word_embeddings(word2vec, word_to_idx)
        
        # 1D CNN text encoder
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout_text = nn.Dropout(0.2)
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(32 + 32 + 128, 128),  # user + item + CNN text
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def _init_word_embeddings(self, word2vec, word_to_idx):
        """Initialize word embeddings with Word2Vec."""
        embedding_matrix = np.zeros((len(word_to_idx), 100))
        for word, idx in word_to_idx.items():
            if word in word2vec.wv:
                embedding_matrix[idx] = word2vec.wv[word]
            elif word == '<UNK>':
                embedding_matrix[idx] = np.random.normal(scale=0.1, size=(100,))
        self.word_emb.weight.data.copy_(torch.from_numpy(embedding_matrix))
        print(f"Initialized {np.sum(np.any(embedding_matrix != 0, axis=1))} words")
    
    def forward(self, user_idx, item_idx, text_seq):
        # User & item embeddings
        user_emb = self.user_emb(user_idx)
        item_emb = self.item_emb(item_idx)
        
        # Text embeddings (CNN)
        word_embs = self.word_emb(text_seq)          # [batch, seq_len, embed_dim]
        x = word_embs.transpose(1, 2)               # [batch, embed_dim, seq_len]
        x = F.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)                # [batch, 128]
        x = self.dropout_text(x)
        
        # Combine embeddings
        combined = torch.cat([user_emb, item_emb, x], dim=1)
        rating = self.predictor(combined).squeeze()
        return rating

# ---------------------------
# 7) Training & evaluation utilities
# ---------------------------
def evaluate_model(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            user_idx = batch['user_idx'].to(device)
            item_idx = batch['item_idx'].to(device)
            text_seq = batch['text_seq'].to(device)
            review_length = batch['review_length'].to(device)
            rating = batch['rating'].to(device)

            out = model(user_idx, item_idx, text_seq)
            preds.extend(out.cpu().numpy())
            targets.extend(rating.cpu().numpy())

    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    return rmse, mae, np.array(preds), np.array(targets)

# ---------------------------
# 8) Training loop (fast-ish)
# ---------------------------
def train_simple(num_epochs=8, batch_size=128, lr=1e-3, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    train_dataset = SimpleRestaurantDataset(train_df, word_to_idx, max_length=100)
    val_dataset   = SimpleRestaurantDataset(val_df, word_to_idx, max_length=100)
    test_dataset  = SimpleRestaurantDataset(test_df, word_to_idx, max_length=100)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SimpleRecommenderCNN(num_users=NUM_USERS, num_items=NUM_ITEMS, vocab_size=vocab_size).to(device)
    print("Model params:", sum(p.numel() for p in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    patience = 3

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            user_idx = batch['user_idx'].to(device)
            item_idx = batch['item_idx'].to(device)
            text_seq = batch['text_seq'].to(device)
            review_length = batch['review_length'].to(device)
            rating = batch['rating'].to(device)

            optimizer.zero_grad()
            out = model(user_idx, item_idx, text_seq)
            loss = criterion(out, rating)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_preds.extend(out.detach().cpu().numpy())
            train_targets.extend(rating.cpu().numpy())

        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        train_mae  = mean_absolute_error(train_targets, train_preds)

        val_rmse, val_mae, _, _ = evaluate_model(model, val_loader, device)

        print(f"\nEpoch {epoch} summary:")
        print(f"  Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
        print(f"  Val   RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")

        # Early stopping & saving
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), "simple_best_model.pth")
            print("  Saved best model.")
            patience = 3
        else:
            patience -= 1
            print(f"  Patience left: {patience}")
            if patience <= 0:
                print("Early stopping.")
                break

    # Test
    model.load_state_dict(torch.load("simple_best_model.pth"))
    test_rmse, test_mae, test_preds, test_targets = evaluate_model(model, test_loader, device)

    # Save prediction CSV
    out_df = test_df.copy().reset_index(drop=True)
    out_df['predicted'] = test_preds
    out_df['error'] = np.abs(out_df['rating'] - out_df['predicted'])
    out_df[['user_id', 'business_id', 'rating', 'predicted', 'error']].to_csv('simple_predictions.csv', index=False)

    return test_rmse, test_mae

# ---------------------------
# 9) Baselines (simple)
# ---------------------------
def run_baselines():
    print("\nBaselines:")
    global_mean = train_df['rating'].mean()

    val_preds_global = [global_mean] * len(val_df)
    test_preds_global = [global_mean] * len(test_df)
    val_rmse_global = np.sqrt(mean_squared_error(val_df['rating'], val_preds_global))
    test_rmse_global = np.sqrt(mean_squared_error(test_df['rating'], test_preds_global))
    print(" Global mean - Val RMSE: {:.4f}, Test RMSE: {:.4f}".format(val_rmse_global, test_rmse_global))

    user_means = train_df.groupby('user_id')['rating'].mean()
    val_preds_user = val_df['user_id'].map(user_means).fillna(global_mean)
    test_preds_user = test_df['user_id'].map(user_means).fillna(global_mean)
    print(" User mean  - Val RMSE: {:.4f}, Test RMSE: {:.4f}".format(
        np.sqrt(mean_squared_error(val_df['rating'], val_preds_user)),
        np.sqrt(mean_squared_error(test_df['rating'], test_preds_user))
    ))

    item_means = train_df.groupby('business_id')['rating'].mean()
    val_preds_item = val_df['business_id'].map(item_means).fillna(global_mean)
    test_preds_item = test_df['business_id'].map(item_means).fillna(global_mean)
    print(" Item mean  - Val RMSE: {:.4f}, Test RMSE: {:.4f}".format(
        np.sqrt(mean_squared_error(val_df['rating'], val_preds_item)),
        np.sqrt(mean_squared_error(test_df['rating'], test_preds_item))
    ))

# ---------------------------
# 10) Main
# ---------------------------
if __name__ == "__main__":
    run_baselines()
    test_rmse, test_mae = train_simple(num_epochs=8, batch_size=128, lr=1e-3)
    print("\nFinal results:")
    print(f" Simple Word2Vec model - Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}")
