import json
import matplotlib.pyplot as plt
import os

# 1. Robustly find the file relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'report_data', 'model_train_metrics.json')

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find file at {file_path}")
    exit(1)

def plot_comparison(models_data, metric_key, ylabel, filename, log_scale=True):
    plt.clf() # Clear the figure
    plt.figure(figsize=(10, 6))
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    
    if log_scale:
        plt.yscale('log')
        plt.grid(True, which="both", linestyle='--', alpha=0.5)
    else:
        plt.grid(True, linestyle='--', alpha=0.5)
    
    # Models to compare with specific labels
    # Mapping internal model names to display names
    target_models = {
        'Q1_1_NoSoftmax': 'Logit', 
        'Q1_1_YesSoftmax': 'Softmax'
    }
    
    found_data = False
    for model_name, display_label in target_models.items():
        if model_name in models_data:
            metrics = models_data[model_name]
            
            if metric_key in metrics:
                values = metrics[metric_key]
                epochs = len(values)
                x = list(range(1, epochs + 1))
                
                plt.plot(x, values, label=display_label)
                found_data = True
            else:
                print(f"Warning: {metric_key} not found for {model_name}")
        else:
            print(f"Warning: Model {model_name} not found in JSON")

    if found_data:
        plt.legend()
        
        # Save to the same directory as the script
        save_path = os.path.join(script_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        print(f"Skipping plot {filename}: No valid data found.")

# Plot 1: Training Loss with Logarithmic Scale
plot_comparison(data, 'train_loss', 'Training Loss', 'training_loss_scale.pdf', log_scale=False)

# Plot 2: Validation Accuracy with Logarithmic Scale
plot_comparison(data, 'val_accuracy', 'Validation Accuracy', 'validation_accuracy_scale.pdf', log_scale=False)