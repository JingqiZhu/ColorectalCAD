import torch
import torch.nn as nn

def create_resnet50(pretrained_source='imagenet', num_classes=3, num_fc_layers=2):

    if pretrained_source == 'gastronet':
        # Load GastroNet-pretrained weights
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=None)
        pretrained_weights = torch.load('/cs/student/projects3/aibh/2023/jingqzhu/code/GastroNet_pretrained_ResNet50.pth')
        state_dict = pretrained_weights['teacher']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    elif pretrained_source == 'imagenet':
        # Load ImageNet-pretrained weights
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='DEFAULT')
    else:
        raise ValueError("Invalid pretrained_source. Choose 'gastronet' or 'imagenet'.")

    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final fully connected layer(s)
    num_ftrs = model.fc.in_features
    if num_fc_layers == 2:
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )
    elif num_fc_layers == 1:
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Invalid num_fc_layers. Choose 1 or 2.")

    # Enable gradients only for the fully connected layers
    for param in model.fc.parameters():
        param.requires_grad = True

    for name, param in model.named_parameters(): 
        if param.requires_grad:
            print(name, param.requires_grad)

    return model