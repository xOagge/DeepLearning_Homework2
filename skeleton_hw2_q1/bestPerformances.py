import json

file_name = 'model_train_metrics_save.json'

# Read the JSON file
with open(file_name, 'r') as f:
    data = json.load(f)

# Iterate through each model
for model_name, metrics in data.items():
    val_acc = metrics['val_accuracy']
    test_acc = metrics['test_accuracy']
    
    # Find the highest validation accuracy and its index (epoch)
    max_val_acc = max(val_acc)
    # Note: lists are 0-indexed, so we add 1 to get the epoch number (1st epoch is index 0)
    best_epoch_index = val_acc.index(max_val_acc)
    best_epoch_number = best_epoch_index + 1
    
    # Get the corresponding test accuracy for that epoch
    best_test_acc = test_acc[best_epoch_index]
    
    # Print the results
    print(f"Model: {model_name}")
    print(f"  Best Epoch: {best_epoch_number}")
    print(f"  Validation Accuracy: {max_val_acc}")
    print(f"  Test Accuracy: {best_test_acc}")
    print("-" * 30)