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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
train_df = train_df[['user_id', 'business_id', 'rating', 'history_reviews']]
val_df   = val_df[['user_id', 'business_id', 'rating', 'history_reviews']]
test_df  = test_df[['user_id', 'business_id', 'rating', 'history_reviews']]

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

# ---------------------------
# 2) Preprocessing helpers
# ---------------------------
def preprocess_text(text):
    """Fast preprocessing: lowercase, remove non-alpha, split."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    return tokens

def extract_history_text(history_reviews, max_reviews=5):
    """Extract text from history reviews (could be list, array, or string)."""
    # Handle None or NaN
    if history_reviews is None or (isinstance(history_reviews, float) and pd.isna(history_reviews)):
        return ''
    
    # Handle empty string
    if isinstance(history_reviews, str) and history_reviews.strip() == '':
        return ''
    
    # Handle list or array
    if isinstance(history_reviews, (list, np.ndarray)):
        if len(history_reviews) == 0:
            return ''
        # Take the most recent reviews
        recent = history_reviews[-max_reviews:] if len(history_reviews) > max_reviews else history_reviews
        return ' '.join([str(r) for r in recent if r and str(r).strip()])
    
    # Handle string or other types
    return str(history_reviews)

# ---------------------------
# 3) Train Word2Vec on TRAIN history reviews only
# ---------------------------
print("Preparing Word2Vec training corpus from TRAIN history reviews...")
train_history_texts = []
for history in train_df['history_reviews']:
    history_text = extract_history_text(history)
    if history_text:
        train_history_texts.append(history_text)

sentences = [preprocess_text(t) for t in train_history_texts]
sentences = [s for s in sentences if len(s) > 3]  # remove very short sequences

print(f"Training Word2Vec on {len(sentences)} history review sequences...")
word2vec = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=3,
    min_count=5,
    workers=1,  # Single worker for reproducibility
    sg=1,
    epochs=8,
    seed=SEED
)

# Build word->idx mapping (PAD=0, UNK=1)
word_to_idx = {'<PAD>': 0, '<UNK>': 1}
for i, word in enumerate(word2vec.wv.key_to_index.keys(), start=2):
    word_to_idx[word] = i
vocab_size = len(word_to_idx)
print(f"Vocab size (with PAD/UNK): {vocab_size}")

# ---------------------------
# 4) Global user/item mappings from TRAIN
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
# 5) Dataset using HISTORY reviews
# ---------------------------
class RecommendationDataset(Dataset):
    def __init__(self, df, word_to_idx, max_length=200, max_history_reviews=5):
        self.df = df.reset_index(drop=True)
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        self.max_history_reviews = max_history_reviews

        # Use global mappings
        self.user_to_idx = user_to_idx_global
        self.item_to_idx = item_to_idx_global

        # Precompute token indices from HISTORY reviews
        self.sequences = []
        self.history_lengths = []
        
        for history in self.df['history_reviews']:
            history_text = extract_history_text(history, max_reviews=max_history_reviews)
            tokens = preprocess_text(history_text)
            self.history_lengths.append(min(len(tokens), max_length))

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
            'history_seq': torch.tensor(self.sequences[idx], dtype=torch.long),
            'history_length': torch.tensor(self.history_lengths[idx], dtype=torch.float32),
            'rating': torch.tensor(row['rating'], dtype=torch.float32)
        }

# ---------------------------
# 6) Model: Predict based on user history + user/item embeddings
# ---------------------------
import torch.nn.functional as F

class HistoryBasedRecommender(nn.Module):
    def __init__(self, num_users, num_items, vocab_size, embedding_dim=100):
        super().__init__()
        
        # User & item embeddings
        self.user_emb = nn.Embedding(num_users, 32)
        self.item_emb = nn.Embedding(num_items, 32)
        
        # Word embeddings for history reviews
        self.word_emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self._init_word_embeddings(word2vec, word_to_idx)
        
        # 1D CNN to encode user's history
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout_text = nn.Dropout(0.2)
        
        # Predictor: combines user, item, and user history
        self.predictor = nn.Sequential(
            nn.Linear(32 + 32 + 128 + 1, 128),  # user + item + history + length
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
        print(f"Initialized {np.sum(np.any(embedding_matrix != 0, axis=1))} words from Word2Vec")
    
    def forward(self, user_idx, item_idx, history_seq, history_length):
        # User & item embeddings
        user_emb = self.user_emb(user_idx)  # [batch, 32]
        item_emb = self.item_emb(item_idx)  # [batch, 32]
        
        # History text embeddings (CNN)
        word_embs = self.word_emb(history_seq)      # [batch, seq_len, embed_dim]
        x = word_embs.transpose(1, 2)               # [batch, embed_dim, seq_len]
        x = F.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)                # [batch, 128]
        x = self.dropout_text(x)
        
        # Normalize history length
        history_length_norm = (history_length / 200.0).unsqueeze(1)  # [batch, 1]
        
        # Combine all features
        combined = torch.cat([user_emb, item_emb, x, history_length_norm], dim=1)
        rating = self.predictor(combined).squeeze(-1)
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
            history_seq = batch['history_seq'].to(device)
            history_length = batch['history_length'].to(device)
            rating = batch['rating'].to(device)

            out = model(user_idx, item_idx, history_seq, history_length)
            preds.extend(out.cpu().numpy())
            targets.extend(rating.cpu().numpy())

    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    return rmse, mae, np.array(preds), np.array(targets)

# ---------------------------
# 8) Training loop
# ---------------------------
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_recommender(num_epochs=10, batch_size=128, lr=1e-3, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    train_dataset = RecommendationDataset(train_df, word_to_idx, max_length=200)
    val_dataset   = RecommendationDataset(val_df, word_to_idx, max_length=200)
    test_dataset  = RecommendationDataset(test_df, word_to_idx, max_length=200)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = HistoryBasedRecommender(
        num_users=NUM_USERS, 
        num_items=NUM_ITEMS, 
        vocab_size=vocab_size
    ).to(device)
    
    print("Model params:", sum(p.numel() for p in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    patience = 3
    patience_counter = 3

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            user_idx = batch['user_idx'].to(device)
            item_idx = batch['item_idx'].to(device)
            history_seq = batch['history_seq'].to(device)
            history_length = batch['history_length'].to(device)
            rating = batch['rating'].to(device)

            optimizer.zero_grad()
            out = model(user_idx, item_idx, history_seq, history_length)
            loss = criterion(out, rating)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_preds.extend(out.detach().cpu().numpy())
            train_targets.extend(rating.cpu().numpy())

        train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        train_mae  = mean_absolute_error(train_targets, train_preds)

        val_rmse, val_mae, _, _ = evaluate_model(model, val_loader, device)

        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
        print(f"  Val   RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict().copy()
            print(" New best model")
            patience_counter = patience
        else:
            patience_counter -= 1
            print(f"  Patience left: {patience_counter}")
            if patience_counter <= 0:
                print("Early stopping triggered.")
                break

    # Test with best model
    print("\n" + "="*50)
    print("Evaluating best model on test set...")
    model.load_state_dict(best_model_state)
    test_rmse, test_mae, test_preds, test_targets = evaluate_model(model, test_loader, device)

    return test_rmse, test_mae

# ---------------------------
# 9) Main
# ---------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("TRAINING HISTORY-BASED RECOMMENDER")
    print("="*50)
    
    test_rmse, test_mae = train_recommender(num_epochs=10, batch_size=128, lr=1e-3)
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")
    print("="*50)