import torch
import matplotlib.pyplot as plt
import os


results_dir = 'results'

num_epochs = 25
epochs = range(num_epochs)


print("Generating Learning Rate plot...")


lr_files = {
    'LR = 0.01': 'history_resnet18_sgd_lr0.01_mom0.9.pth',
    'LR = 0.001 (Baseline)': 'history_resnet18_sgd_lr0.001_mom0.9.pth',
    'LR = 0.0001': 'history_resnet18_sgd_lr0.0001_mom0.9.pth'
}

# Create a new plot
plt.figure(figsize=(10, 6))

# Loop over each file, load it, and plot its validation accuracy
for label, filename in lr_files.items():
    file_path = os.path.join(results_dir, filename)
    if os.path.exists(file_path):
        # Load the history object from the file
        history = torch.load(file_path)
        
        # Get the validation accuracy list
        val_acc = [acc * 100 for acc in history['val_acc']] # Convert to percentage
        
        # Plot the data
        plt.plot(epochs, val_acc, label=label)
    else:
        print(f"Warning: File not found, skipping. {file_path}")

# Add plot labels and title
plt.title('Validation Accuracy vs. Epochs (Different Learning Rates)')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.legend()
plt.grid(True)

# Save the plot as an image file in your 'results' folder
plt.savefig(os.path.join(results_dir, 'plot_learning_rates.png'))
print(f"Saved plot to {os.path.join(results_dir, 'plot_learning_rates.png')}")



print("\nGenerating Momentum plot...")

momentum_files = {
    'Momentum = 0.8': 'history_resnet18_sgd_lr0.001_mom0.8.pth',
    'Momentum = 0.9 (Baseline)': 'history_resnet18_sgd_lr0.001_mom0.9.pth',
    'Momentum = 0.95': 'history_resnet18_sgd_lr0.001_mom0.95.pth'
}

# Create a second, new plot
plt.figure(figsize=(10, 6))

# Loop over each file, load it, and plot
for label, filename in momentum_files.items():
    file_path = os.path.join(results_dir, filename)
    if os.path.exists(file_path):
        history = torch.load(file_path)
        val_acc = [acc * 100 for acc in history['val_acc']] # Convert to percentage
        plt.plot(epochs, val_acc, label=label)
    else:
        print(f"Warning: File not found, skipping. {file_path}")

# Add plot labels and title
plt.title('Validation Accuracy vs. Epochs (Different Momentum Values)')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.legend()
plt.grid(True)

# Save the second plot as an image
plt.savefig(os.path.join(results_dir, 'plot_momentum.png'))
print(f"Saved plot to {os.path.join(results_dir, 'plot_momentum.png')}")

print("\nDone. Check your 'results' folder for 'plot_learning_rates.png' and 'plot_momentum.png'")
