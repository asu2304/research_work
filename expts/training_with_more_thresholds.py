import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
from tqdm import tqdm

# --- Configuration ---
CONFIG = {
    'vocab_set1': 10,      # Tokens 0-9 (Signal)
    'vocab_set2': 10,      # Tokens 10-19 (Noise)
    'total_vocab': 20,
    'seq_len': 200,
    'attn_dim': 64,
    'num_heads': 1,
    'batch_size': 128,
    'lr': 1e-2,
    'num_epochs': 200,      # Adjusted for multi-run efficiency
    'train_size': 20000,
    'test_size': 5000,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'thresholds': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'output_dir': 'experiment_artifacts'
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# --- 1. Data Generation (Modified for Threshold x) ---
def generate_dataset(num_samples, seq_len, threshold_x):
    x_data, y_data = [], []
    for _ in range(num_samples):
        transitions = {i: np.random.choice(CONFIG['vocab_set1'], size=2, replace=False) 
                       for i in range(CONFIG['vocab_set1'])}
        seq = []
        last_ruled_token = None
        for _ in range(seq_len + 1):
            # x is the probability of selecting a ruled token
            if np.random.rand() < threshold_x:
                token = np.random.randint(0, CONFIG['vocab_set1']) if last_ruled_token is None \
                        else np.random.choice(transitions[last_ruled_token])
                last_ruled_token = token
            else:
                token = np.random.randint(CONFIG['vocab_set1'], CONFIG['total_vocab'])
            seq.append(token)
        x_data.append(seq[:-1])
        y_data.append(seq[1:])
    return torch.tensor(x_data, dtype=torch.long), torch.tensor(y_data, dtype=torch.long)

# --- 2. Model Architecture (Same as provided) ---
class ConcatAttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.W_q = nn.Linear(input_dim, output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, output_dim, bias=False)
        self.W_o = nn.Linear(output_dim, output_dim, bias=False)
        
    def forward(self, x, mask=None):
        B, T, D = x.shape
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.output_dim ** 0.5)
        if mask is not None: scores += mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = self.W_o(torch.bmm(attn_weights, V))
        return torch.cat([x, attn_out], dim=-1), attn_weights
        
class DisentangledTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_dim = CONFIG['total_vocab'] + CONFIG['seq_len']
        self.layer1 = ConcatAttentionBlock(self.initial_dim, CONFIG['attn_dim'])
        self.layer2 = ConcatAttentionBlock(self.initial_dim + CONFIG['attn_dim'], CONFIG['attn_dim'])
        self.fc_out = nn.Linear(self.initial_dim + 2 * CONFIG['attn_dim'], CONFIG['total_vocab'], bias=False)
    
    def forward(self, x, return_attn=False):
        B, T = x.shape
        tokens_oh = F.one_hot(x, num_classes=CONFIG['total_vocab']).float()
        pos_oh = F.one_hot(torch.arange(T, device=x.device), num_classes=CONFIG['seq_len']).float().unsqueeze(0).repeat(B, 1, 1)
        x_stream = torch.cat([tokens_oh, pos_oh], dim=-1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        
        x_stream, a1 = self.layer1(x_stream, mask)
        x_stream, a2 = self.layer2(x_stream, mask)
        logits = self.fc_out(x_stream)
        return (logits, [a1, a2]) if return_attn else logits

# --- 3. Multi-Threshold Training Loop ---
results = {}

for x_thresh in CONFIG['thresholds']:
    
    print(f"\n>>> Training for Probability Threshold x = {x_thresh}")
    train_x, train_y = generate_dataset(CONFIG['train_size'], CONFIG['seq_len'], x_thresh)
    test_x, test_y = generate_dataset(CONFIG['test_size'], CONFIG['seq_len'], x_thresh)
    # loader = data.DataLoader(data.TensorDataset(train_x, train_y), batch_size=CONFIG['batch_size'], shuffle=True)
    loader = data.DataLoader(
    data.TensorDataset(train_x, train_y), 
    batch_size=CONFIG['batch_size'], 
    shuffle=True,
    num_workers=4,  # Add this if the CPU is the bottleneck
    pin_memory=True # Recommended for GPU training
    )
    model = DisentangledTransformer().to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss()
    
    history = {'loss': [], 'acc': []}
    
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        l_sum = 0
        for xb, yb in loader:
            xb, yb = xb.to(CONFIG['device']), yb.to(CONFIG['device'])
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits.view(-1, CONFIG['total_vocab']), yb.view(-1))
            loss.backward()
            optimizer.step()
            l_sum += loss.item()
        
        # Eval
        model.eval()
        with torch.no_grad():
            tx, ty = test_x[:500].to(CONFIG['device']), test_y[:500].to(CONFIG['device'])
            t_logits = model(tx)
            acc = (t_logits.argmax(-1) == ty).float().mean().item()
            history['loss'].append(l_sum/len(loader))
            history['acc'].append(acc)
            
    # Save Model Weights
    torch.save(model.state_dict(), f"{CONFIG['output_dir']}/model_x_{x_thresh}.pth")
    results[x_thresh] = history

# Save metrics to JSON
with open(f"{CONFIG['output_dir']}/metrics.json", 'w') as f:
    json.dump(results, f)
    
# Plot Training Comparison
plt.figure(figsize=(12, 5))
for x_t, h in results.items():
    plt.plot(h['acc'], label=f'x={x_t}')
plt.title("Test Accuracy across different Probability Thresholds (x)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(f"{CONFIG['output_dir']}/accuracy_comparison.png")
plt.show()

print(f"\nDone! All artifacts (models, metrics, plots) are in /{CONFIG['output_dir']}")


