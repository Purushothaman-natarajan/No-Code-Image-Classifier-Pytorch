import torch
import torch.nn as nn
from torchvision import models

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
