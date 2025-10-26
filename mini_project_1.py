import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms,models
from torchvision.models import ResNet18_Weights

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25):
    
    start_time = time.time()

    # Keeping track of the best model's weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Storing loss and accuracy
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)


        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Get the total number of images in this phase
            dataset_size = len(dataloaders[phase].dataset)

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Reset the gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # Get the class prediction
                    loss = criterion(outputs, labels)

                    # Backward pass & optimizing
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculating epoch loss and accuracy
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item()) # .item() to get Python number
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # Save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history



if __name__ == "__main__":
    # Data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Loading datasets
    data_dir = 'dataset'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    # Creating batches
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

# Checking to see if everything is set up correctly

    # print(f"Classes found: {class_names}")
    # print(f"Training images: {dataset_sizes['train']}, Validation images: {dataset_sizes['val']}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    inputs, classes = next(iter(dataloaders['train']))

    # print(f"Batch shape (Input): {inputs.shape}")
    # print(f"Batch shape (Classes): {classes.shape}") 
    # print(f"Class labels for this batch: {classes}")

# Building a baseline model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, len(class_names))

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Training param: {name}")

    model = model.to(device)

# Training the model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.95)

    print("Starting baseline model training...")
    # Call the function to start training
    trained_model_8, baseline_history_8 = train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25)
    torch.save(  baseline_history_8, 'results/history_resnet18_lr0001_mom095.pth')

    print("Baseline training finished.")
    print(  baseline_history_8['val_acc'])

  