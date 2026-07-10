import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

# 1. CONFIGURATION & SETUP
CONFIG = {
    'total_vocab': 20,
    'vocab_set1': 10,
    'seq_len': 200,
    'attn_dim': 64,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'output_dir': 'experiment_results_2',
    'models': {
            0.1: "experiment_artifacts/model_x_0.1.pth",
            0.2: "experiment_artifacts/model_x_0.2.pth",
            0.3: "experiment_artifacts/model_x_0.3.pth",
            0.4: "experiment_artifacts/model_x_0.4.pth",
            0.5: "experiment_artifacts/model_x_0.5.pth",
            0.6: "experiment_artifacts/model_x_0.6.pth",
            0.7: "experiment_artifacts/model_x_0.7.pth",
            0.8: "experiment_artifacts/model_x_0.8.pth",
            0.9: "experiment_artifacts/model_x_0.9.pth"
    }
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# 2. MODEL DEFINITION
class ConcatAttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.W_q = nn.Linear(input_dim, output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, output_dim, bias=False)
        self.W_o = nn.Linear(output_dim, output_dim, bias=False)

    def forward(self, x, mask=None):
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

    def forward(self, x, return_intermediates=False):
        B, T = x.shape
        tokens_oh = F.one_hot(x, num_classes=CONFIG['total_vocab']).float()
        pos_oh = F.one_hot(torch.arange(T, device=x.device), num_classes=CONFIG['seq_len']).float().unsqueeze(0).repeat(B, 1, 1)
        x_stream = torch.cat([tokens_oh, pos_oh], dim=-1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        x_stream, attn1 = self.layer1(x_stream, mask)
        hidden_state = x_stream.clone()
        x_stream, attn2 = self.layer2(x_stream, mask)
        logits = self.fc_out(x_stream)
        if return_intermediates:
            return logits, hidden_state, [attn1, attn2]
        return logits

# 3. DATA GENERATION
def generate_tracked_test_data(num_samples, threshold_x):
    x_data, y_data, ruled_mask = [], [], []
    for _ in range(num_samples):
        transitions = {i: np.random.choice(CONFIG['vocab_set1'], size=2, replace=False) for i in range(CONFIG['vocab_set1'])}
        seq, is_ruled_token = [], []
        last_ruled = None
        for _ in range(CONFIG['seq_len'] + 1):
            if np.random.rand() < threshold_x:
                token = np.random.randint(0, CONFIG['vocab_set1']) if last_ruled is None else np.random.choice(transitions[last_ruled])
                last_ruled = token
                is_ruled_token.append(True)
            else:
                token = np.random.randint(CONFIG['vocab_set1'], CONFIG['total_vocab'])
                is_ruled_token.append(False)
            seq.append(token)
        x_data.append(seq[:-1])
        y_data.append(seq[1:])
        ruled_mask.append(is_ruled_token[1:]) 
    return torch.tensor(x_data), torch.tensor(y_data), torch.tensor(ruled_mask)

# 4. EXPERIMENTS
def run_exp1_context_warmup(models_dict):
    print("\n--- Running Exp 1: Context Warm-up Phase Transition ---")
    plt.figure(figsize=(12, 6))
    for x_val, model in models_dict.items():
        tx, ty, mask = generate_tracked_test_data(1000, x_val)
        tx, ty = tx.to(CONFIG['device']), ty.to(CONFIG['device'])
        with torch.no_grad():
            preds = model(tx).argmax(-1).cpu()
        acc_at_t = []
        for t in range(CONFIG['seq_len']):
            ruled_at_t = mask[:, t]
            if ruled_at_t.sum() > 0:
                acc = (preds[ruled_at_t, t] == ty.cpu()[ruled_at_t, t]).float().mean().item()
            else:
                acc = np.nan
            acc_at_t.append(acc)
        smoothed_acc = np.convolve(acc_at_t, np.ones(10)/10, mode='valid')
        plt.plot(smoothed_acc, label=f'x={x_val}')
    plt.title("Exp 1: Rule Accuracy vs. Sequence Position (Warm-up)")
    plt.xlabel("Sequence Position (t)")
    plt.ylabel("Accuracy on Ruled Tokens")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{CONFIG['output_dir']}/exp1_context_warmup.png")
    plt.close()

def run_exp2_attention_sparsity(models_dict):
    print("--- Running Exp 2: Attention Sparsity Analysis ---")
    sparsity_results = {}
    for x_val, model in models_dict.items():
        tx, ty, mask = generate_tracked_test_data(200, x_val)
        tx = tx.to(CONFIG['device'])
        with torch.no_grad():
            _, _, attns = model(tx, return_intermediates=True)
            attn2 = attns[1]
        noise_mask = ~mask
        avg_attention_to_noise = []
        for b in range(tx.shape[0]):
            for t in range(10, CONFIG['seq_len']):
                if mask[b, t]:
                    weights = attn2[b, t, :t]
                    prev_noise_mask = noise_mask[b, :t]
                    attention_to_noise = weights[prev_noise_mask].sum().item()
                    avg_attention_to_noise.append(attention_to_noise)
        sparsity_results[x_val] = np.mean(avg_attention_to_noise)
    plt.figure(figsize=(8, 5))
    plt.plot(list(sparsity_results.keys()), list(sparsity_results.values()), marker='o', color='purple')
    plt.title("Exp 2: Attention Wasted on Noise Tokens")
    plt.xlabel("Data Generation Threshold (x)")
    plt.ylabel("Total Attention Mass on Noise")
    plt.grid(alpha=0.3)
    plt.savefig(f"{CONFIG['output_dir']}/exp2_attention_sparsity.png")
    plt.close()

def run_exp3_linear_probing(models_dict):
    print("--- Running Exp 3: Geometric Disentanglement (Linear Probing) ---")
    probe_accuracies = {}
    for x_val, model in models_dict.items():
        tx, ty, mask = generate_tracked_test_data(1000, x_val)
        tx = tx.to(CONFIG['device'])
        with torch.no_grad():
            _, hidden_states, _ = model(tx, return_intermediates=True)
        X_probe = hidden_states.view(-1, hidden_states.shape[-1]).cpu().numpy()
        Y_probe = mask.view(-1).numpy().astype(int)
        X_tensor = torch.tensor(X_probe, dtype=torch.float32).to(CONFIG['device'])
        Y_tensor = torch.tensor(Y_probe, dtype=torch.float32).unsqueeze(1).to(CONFIG['device'])
        probe = nn.Linear(X_tensor.shape[-1], 1).to(CONFIG['device'])
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        for _ in range(100):
            optimizer.zero_grad()
            out = probe(X_tensor)
            loss = criterion(out, Y_tensor)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            preds = torch.sigmoid(probe(X_tensor)).round()
            acc = (preds == Y_tensor).float().mean().item()
            probe_accuracies[x_val] = acc
    plt.figure(figsize=(8, 5))
    plt.plot(list(probe_accuracies.keys()), list(probe_accuracies.values()), marker='s', color='green')
    plt.title("Exp 3: Probe Accuracy (Detecting Signal vs Noise in Hidden State)")
    plt.xlabel("Probability Threshold (x)")
    plt.ylabel("Probe Accuracy")
    plt.axhline(y=1.0, color='gray', linestyle=':')
    plt.grid(alpha=0.3)
    plt.savefig(f"{CONFIG['output_dir']}/exp3_linear_probing.png")
    plt.close()

def run_exp4_signal_efficiency(models_dict):
    print("\n--- Running Exp 4: Accuracy vs. Signal Count (Normalizing Noise) ---")
    plt.figure(figsize=(12, 6))
    for x_val, model in models_dict.items():
        tx, ty, mask = generate_tracked_test_data(1000, x_val)
        tx, ty = tx.to(CONFIG['device']), ty.to(CONFIG['device'])
        with torch.no_grad():
            preds = model(tx).argmax(-1).cpu()
        
        # Track accuracy based on signal count seen so far
        signal_counts = mask.cumsum(dim=1)
        max_signals = signal_counts.max().item()
        
        acc_per_count = []
        counts_axis = np.arange(1, int(max_signals))
        
        for c in counts_axis:
            # Mask for tokens that are exactly the c-th signal seen in their sequence
            c_mask = (signal_counts == c) & mask 
            if c_mask.sum() > 0:
                acc = (preds[c_mask] == ty.cpu()[c_mask]).float().mean().item()
                acc_per_count.append(acc)
            else:
                acc_per_count.append(np.nan)
        
        plt.plot(counts_axis, acc_per_count, label=f'x={x_val}')

    plt.title("Exp 4: Accuracy vs. Number of Signal Tokens Encountered")
    plt.xlabel("Count of Signal Tokens Seen So Far")
    plt.ylabel("Accuracy on Current Ruled Token")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{CONFIG['output_dir']}/exp4_signal_efficiency.png")
    plt.close()
    
    
def run_exp_specific_token_frequency(models_dict):
    print("\n--- Running Experiment: Accuracy vs. Specific Token Appearance Count ---")
    plt.figure(figsize=(12, 7))
    
    # Define colors for the 9 thresholds for better visual separation
    colors = plt.cm.viridis(np.linspace(0, 1, len(models_dict)))
    
    for (x_val, model), color in zip(sorted(models_dict.items()), colors):
        # Using a larger sample size to ensure we get high occurrence counts
        tx, ty, mask = generate_tracked_test_data(2000, x_val)
        tx, ty = tx.to(CONFIG['device']), ty.to(CONFIG['device'])
        
        with torch.no_grad():
            preds = model(tx).argmax(-1).cpu()
        
        targets = ty.cpu()
        ruled_mask = mask.cpu()
        input_tokens = tx.cpu()
        
        # We need to track how many times the CURRENT input token has appeared 
        # BEFORE this position in this specific sequence.
        appearance_counts = torch.zeros_like(input_tokens)
        for b in range(input_tokens.shape[0]):
            # Use a dictionary to track counts for tokens 0-9 in this batch
            counts = {k: 0 for k in range(CONFIG['vocab_set1'])}
            for t in range(input_tokens.shape[1]):
                token = input_tokens[b, t].item()
                if token < CONFIG['vocab_set1']: # Only track signal tokens
                    appearance_counts[b, t] = counts[token]
                    counts[token] += 1
        
        # Calculate accuracy grouped by appearance count
        max_appearances = int(appearance_counts.max().item())
        acc_per_count = []
        count_axis = []
        
        for c in range(max_appearances + 1):
            # Mask: Is it a ruled token AND is this the c-th time we've seen it?
            c_mask = ruled_mask & (appearance_counts == c)
            
            if c_mask.sum() > 30: # Minimum sample threshold for stability
                acc = (preds[c_mask] == targets[c_mask]).float().mean().item()
                acc_per_count.append(acc)
                count_axis.append(c)
        
        if len(acc_per_count) > 2:
            plt.plot(count_axis, acc_per_count, label=f'x={x_val}', color=color, marker='o', markersize=4, alpha=0.8)

    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Theoretical Max')
    plt.title("Accuracy vs. How Many Times the Specific Token was Seen in Context", fontsize=14)
    plt.xlabel("Appearance Count of the Specific Signal Token", fontsize=12)
    plt.ylabel("Accuracy on Ruled Prediction", fontsize=12)
    plt.legend(title="Threshold (x)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/accuracy_vs_token_frequency.png")
    plt.show()
    
    
def run_exp_single_sequence_per_threshold(models_dict):
    print("\n--- Running Exp: Learning Curves for 9 Specific Sequences ---")
    plt.figure(figsize=(12, 7))
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(models_dict)))
    
    for (x_val, model), color in zip(sorted(models_dict.items()), colors):
        tx, ty, mask = generate_tracked_test_data(1, x_val)
        tx, ty = tx.to(CONFIG['device']), ty.to(CONFIG['device'])
        
        with torch.no_grad():
            preds = model(tx).argmax(-1).cpu().squeeze(0)
        
        targets = ty.cpu().squeeze(0)
        ruled_mask = mask.cpu().squeeze(0)
        input_tokens = tx.cpu().squeeze(0)
        
        appearance_counts = []
        accuracies = []
        # FIX: Initialize for total_vocab (20) instead of vocab_set1 (10)
        token_counts = {k: 0 for k in range(CONFIG['total_vocab'])}
        
        for t in range(CONFIG['seq_len']):
            token = input_tokens[t].item()
            if ruled_mask[t]:
                # Now token_counts has keys for 0-19, so no more KeyErrors
                count = token_counts[token]
                is_correct = 1.0 if preds[t] == targets[t] else 0.0
                
                appearance_counts.append(count)
                accuracies.append(is_correct)
                token_counts[token] += 1
        
        counts_arr = np.array(appearance_counts)
        acc_arr = np.array(accuracies)
        unique_counts = np.unique(counts_arr)
        
        curve_y = []
        for c in unique_counts:
            curve_y.append(np.mean(acc_arr[counts_arr == c]))
            
        if len(curve_y) > 1:
            plt.plot(unique_counts, curve_y, label=f'x={x_val}', color=color, linewidth=2, marker='o', markersize=3)

    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
    plt.title("Learning Trajectory: 1 Sequence per Threshold")
    plt.xlabel("Appearance Count of Specific Token")
    plt.ylabel("Accuracy")
    plt.legend(title="Threshold (x)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/single_sequence_curves.png")
    plt.show()
    
    
# 5. DOCUMENT GENERATOR
def generate_documentation():
    print("--- Generating Documentation Report ---")
    doc_content = """# Mechanistic Interpretability Experiments Report (V2)
... [Exp 1-3 content omitted for brevity] ...

---

### Experiment 4: Accuracy vs. Signal Count (Normalizing Noise)
* **File:** `exp4_signal_efficiency.png`
* **Why it was performed:** This experiment "normalizes" for the noise density. In Exp 1, low $x$ looks "slower" simply because signal tokens are farther apart in time. Exp 4 asks: "If we only look at the amount of useful information (signals) seen, does the model learn at the same speed regardless of noise?"
* **Implementation Strategy:** We use a cumulative sum on the ruled-mask to assign each token a "Signal Index." We then group accuracy by this index rather than by position $t$.
* **What to look for:** If all lines overlap, the model is perfectly noise-robust. If low $x$ lines are lower than high $x$ lines on this plot, it proves that "distractor" noise tokens cause catastrophic forgetting or structural degradation in the attention mechanism.
"""
    with open(f"{CONFIG['output_dir']}/experiment_rationale.md", "w") as f:
        f.write(doc_content)

if __name__ == "__main__":
    loaded_models = {}
    print("Loading models...")
    for x_val, path in CONFIG['models'].items():
        if os.path.exists(path):
            model = DisentangledTransformer().to(CONFIG['device'])
            model.load_state_dict(torch.load(path, map_location=CONFIG['device']))
            model.eval()
            loaded_models[x_val] = model
    if loaded_models:
        # run_exp1_context_warmup(loaded_models)
        # run_exp2_attention_sparsity(loaded_models)
        # run_exp3_linear_probing(loaded_models)
        # run_exp4_signal_efficiency(loaded_models)
        # run_exp_specific_token_frequency(loaded_models)
        run_exp_single_sequence_per_threshold(loaded_models)
        # generate_documentation()
        print(f"\nAll tasks complete. Artifacts saved in ./{CONFIG['output_dir']}/")
        
        