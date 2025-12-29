import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Set standard float32 precision
torch.set_default_dtype(torch.float32)

def naive_cross_entropy(logits, target_index):
    """
    Naive: Softmax -> Log -> NegativeLikelihood
    Mathematically correct, but does calculations in a specific order
    that accumulates floating point noise differently.
    """
    probs = torch.softmax(logits, dim=0)
    log_probs = torch.log(probs)
    loss = -log_probs[target_index]
    return loss.item()

def stable_cross_entropy(logits, target_index):
    """
    Stable: Fused CrossEntropyLoss
    Uses LogSumExp trick internally.
    """
    loss_fn = nn.CrossEntropyLoss()
    # Unsqueeze to add batch dimension required by PyTorch loss functions
    loss = loss_fn(logits.unsqueeze(0), torch.tensor([target_index]))
    return loss.item()

# ---------------------------------------------------------
# 1. APPLY THE LIMIT HERE
# We set the range to 0-10.
# ---------------------------------------------------------
x_values = np.linspace(0, 10, 200)

naive_losses = []
stable_losses = []
differences = []

target = 0

for x in x_values:
    # Logits: [variable, 0.0]
    logits = torch.tensor([x, 0.0])
    
    n_loss = naive_cross_entropy(logits, target)
    s_loss = stable_cross_entropy(logits, target)
    
    naive_losses.append(n_loss)
    stable_losses.append(s_loss)
    
    # We calculate the tiny difference between the two
    differences.append(abs(n_loss - s_loss))

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: The Main Loss Curves
ax1.plot(x_values, stable_losses, label="PyTorch CrossEntropyLoss (Stable)", 
         color='green', linewidth=5, alpha=0.4)
ax1.plot(x_values, naive_losses, label="Naive Log(Softmax)", 
         color='red', linestyle='--', linewidth=2)
ax1.set_title("Loss Value (Logits 0 to 10)")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.text(2, 2, "Visual Overlap:\nBoth methods work perfectly here", fontsize=10)

# Plot 2: The Microscopic Difference
ax2.plot(x_values, differences, color='purple', linewidth=1)
ax2.set_title("Floating Point Discrepancy (Naive minus Stable)")
ax2.set_ylabel("Absolute Diff (Log Scale)")
ax2.set_xlabel("Logit Value")
ax2.set_yscale("log") # Log scale is necessary to see 1e-7 errors
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()