import json
import matplotlib.pyplot as plt
import os

# 1. robustly find the file relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'model_train_metrics_save.json')

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find file at {file_path}")
    exit(1)

def plot(epochs, val_acc, test_acc, ylabel='Accuracy', name=''):
    plt.clf() # Clear the figure to avoid overwriting
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)

    x = list(range(1, epochs + 1))

    # Plot both lines
    plt.plot(x, val_acc, label='Validation Accuracy')
    plt.plot(x, test_acc, label='Test Accuracy')
    
    plt.legend()
    
    # Save to the same directory as the script
    save_path = os.path.join(script_dir, f'{name}.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved plot to {save_path}")

# Iterate through the experiments and plot
for experiment_name, metrics in data.items():
    # Check if necessary data exists for this experiment
    if 'val_accuracy' in metrics and 'test_accuracy' in metrics:
        val_acc = metrics['val_accuracy']
        test_acc = metrics['test_accuracy']
        
        # Assume number of epochs matches the length of the data
        epochs = len(val_acc)
        
        plot(epochs, val_acc, test_acc, name=experiment_name)