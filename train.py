import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, image_files, image_labels, transform=None):
        self.image_files = image_files
        self.image_labels = image_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        label = self.image_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
def load_data(data_dir, image_size, batch_size):
    # Define subfolders for train, validation, and test
    subfolders = ['train', 'val', 'test']
    image_files = {subfolder: [] for subfolder in subfolders}
    image_labels = {subfolder: [] for subfolder in subfolders}

    labels = [d.name for d in os.scandir(os.path.join(data_dir, 'train')) if d.is_dir()]
    if not labels:
        raise ValueError("No subdirectories found in the train directory. Please ensure the directory contains labeled subfolders.")

    label_map = {label: idx for idx, label in enumerate(labels)}

    for subfolder in subfolders:
        subfolder_path = os.path.join(data_dir, subfolder)
        for label in labels:
            label_dir = os.path.join(subfolder_path, label)
            for image_file in os.listdir(label_dir):
                image_files[subfolder].append(os.path.join(label_dir, image_file))
                image_labels[subfolder].append(label)

        image_labels[subfolder] = [label_map[label] for label in image_labels[subfolder]]

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(image_files['train'], image_labels['train'], transform=transform)
    val_dataset = CustomDataset(image_files['val'], image_labels['val'], transform=transform)
    test_dataset = CustomDataset(image_files['test'], image_labels['test'], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(labels)
    return train_loader, val_loader, num_classes  # Adjusted return values

class CustomModel(nn.Module):
    def __init__(self, base_model_names, num_classes):
        super(CustomModel, self).__init__()

        # Dictionary to map model names to functions and weights
        self.model_dict = {
            'alexnet': (models.alexnet, models.AlexNet_Weights.DEFAULT),
            'convnext_tiny': (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
            'densenet121': (models.densenet121, models.DenseNet121_Weights.DEFAULT),
            'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            'efficientnet_v2_s': (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.DEFAULT),
            'googlenet': (models.googlenet, models.GoogLeNet_Weights.DEFAULT),
            'inception_v3': (models.inception_v3, models.Inception_V3_Weights.DEFAULT),
            'mnasnet1_0': (models.mnasnet1_0, models.MNASNet1_0_Weights.DEFAULT),
            'mobilenet_v2': (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
            'mobilenet_v3_small': (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
            'regnet_y_400mf': (models.regnet_y_400mf, models.RegNet_Y_400MF_Weights.DEFAULT),
            'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT),
            'resnext50_32x4d': (models.resnext50_32x4d, models.ResNeXt50_32X4D_Weights.DEFAULT),
            'shufflenet_v2_x1_0': (models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
            'squeezenet1_0': (models.squeezenet1_0, models.SqueezeNet1_0_Weights.DEFAULT),
            'vgg16': (models.vgg16, models.VGG16_Weights.DEFAULT),
            'wide_resnet50_2': (models.wide_resnet50_2, models.Wide_ResNet50_2_Weights.DEFAULT),
        }

        self.models = nn.ModuleList()
        self.base_model_names = base_model_names

        for base_model_name in base_model_names:
            if base_model_name not in self.model_dict:
                raise ValueError(f"Unsupported model name: {base_model_name}")

            base_model_func, weight_func = self.model_dict[base_model_name]
            base_model = base_model_func(weights=weight_func)

            # Freeze the base model layers
            for param in base_model.parameters():
                param.requires_grad = False

            # Modify the classifier or head based on the model type
            if hasattr(base_model, 'classifier'):
                if isinstance(base_model.classifier, nn.Sequential):
                    num_ftrs = base_model.classifier[0].in_features
                else:
                    num_ftrs = base_model.classifier.in_features

                base_model.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(num_ftrs, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(512, num_classes)
                )
            elif hasattr(base_model, 'fc'):
                num_ftrs = base_model.fc.in_features
                base_model.fc = nn.Sequential(
                    nn.Linear(num_ftrs, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(512, num_classes)
                )
            elif hasattr(base_model, 'head'):
                num_ftrs = base_model.head.in_features
                base_model.head = nn.Sequential(
                    nn.Linear(num_ftrs, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(512, num_classes)
                )
            else:
                raise NotImplementedError(f"Modification for {base_model_name} not implemented")

            self.models.append(base_model)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return outputs


def create_and_train_model(base_model_names, num_classes, train_loader, val_loader, log_dir, model_dir,
                           epochs, optimizer_name, learning_rate, step_gamma, alpha, batch_size, patience):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for base_model_name in base_model_names:
        print(f"Training with base model: {base_model_name}")

        model = CustomModel([base_model_name], num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()

        if optimizer_name.lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=alpha)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        scheduler = ExponentialLR(optimizer, gamma=step_gamma)

        best_acc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs[0], labels)  # Only use the output from the first model
                _, preds = torch.max(outputs[0], 1)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            print(f"Training Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            model.eval()
            val_loss = 0.0
            val_corrects = 0

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc="Validation"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs[0], labels)
                    _, preds = torch.max(outputs[0], 1)
                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)

            val_loss /= len(val_loader.dataset)
            val_acc = val_corrects.double() / len(val_loader.dataset)
            print(f"Validation Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                model_path = os.path.join(model_dir, f"{base_model_name}_best_model.pth")
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break

            scheduler.step()

        print(f"Finished Training {base_model_name} with best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Custom Models')
    parser.add_argument('--data_path', type=str, required=True, help='Directory containing the data')
    parser.add_argument('--shape', type=int, default=224, help='Image size (default: 224)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (default: 20)')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer (default: adam)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--step_gamma', type=float, default=0.9, help='Learning rate scheduler gamma (default: 0.9)')
    parser.add_argument('--alpha', type=float, default=0.9, help='Momentum for SGD (default: 0.9)')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping (default: 5)')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for TensorBoard logs (default: logs)')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models (default: models)')
    parser.add_argument('--base_model_names', type=str, nargs='+', required=True, help='Base model names')
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    train_loader, val_loader, num_classes = load_data(args.data_path, args.shape, args.batch_size)
    create_and_train_model(args.base_model_names, num_classes, train_loader, val_loader,
                           args.log_dir, args.model_dir, args.epochs, args.optimizer, args.learning_rate,
                           args.step_gamma, args.alpha, args.batch_size, args.patience)
