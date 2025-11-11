import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import random

def visualize_model_predictions(model, dataloader, device, num_images=4):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(10, 10))
    
    class_names = dataloader.dataset.classes

    with torch.no_grad():
        # Getting a random batch of data
        inputs, labels = next(iter(dataloader))
        
        # Select 4 random indices from the batch
        batch_size = inputs.size()[0]
        indices = random.sample(range(batch_size), k=num_images)
        
        selected_inputs = inputs[indices].to(device)
        selected_labels = labels[indices].to(device)

        outputs = model(selected_inputs)
        _, preds = torch.max(outputs, 1)

        for j in range(selected_inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            
            
            title = f'True: {class_names[selected_labels[j]]} | Predicted: {class_names[preds[j]]}'
            ax.set_title(title, color=("green" if preds[j] == selected_labels[j] else "red"))
            
            
            inp = selected_inputs.cpu().data[j].numpy().transpose((1, 2, 0))
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            inp = std * inp + mean
            inp = inp.clip(0, 1)
            plt.imshow(inp)

    model.train(mode=was_training)
    return fig


best_model_path = 'results/model_resnet50.pth' 
results_dir = 'results'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = 'dataset'
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)


model = models.resnet50(weights=None)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Loading the weights
model.load_state_dict(torch.load(best_model_path, map_location=device))
model = model.to(device)

# Plot
prediction_fig = visualize_model_predictions(model, val_dataloader, device, num_images=4)
prediction_fig.savefig(os.path.join(results_dir, 'plot_predictions.png'))
print(f"Saved plot to {os.path.join(results_dir, 'plot_predictions.png')}")