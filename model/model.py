import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, \
    wide_resnet50_2, wide_resnet101_2, mnasnet1_0, mobilenet_v2, alexnet, vgg16, squeezenet1_1, densenet161

torchvision_models = [resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d,
                      wide_resnet50_2, wide_resnet101_2, mnasnet1_0, mobilenet_v2, alexnet, vgg16, squeezenet1_1,
                      densenet161]
torchvision_models_names = [x.__name__ for x in torchvision_models]
torchvision_models_dict = {x.__name__: x for x in torchvision_models}


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self._num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def _freeze_layers(model, layers_to_freeze):
    for i, child in enumerate(model.children()):
        print("Freezing layer {}".format(i))
        for param in child.parameters():
            param.requires_grad = False
        if i + 1 >= layers_to_freeze:
            return


def torchvision_model(model_name, num_class=10, pretrained=False, frozen=0):
    unknown_model_msg = "Unknown model name. Model name should be in {}".format(torchvision_models_names)
    assert model_name in torchvision_models_names, unknown_model_msg
    model = torchvision_models_dict[model_name](pretrained=pretrained)
    if pretrained and frozen:
        _freeze_layers(model, frozen)
    if 'resnet' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_class)
    elif model_name == "alexnet":
        """ Alexnet
        """
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_class)

    elif "vgg" in model_name:
        """ VGG
        """
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_class)

    elif "squeezenet" in model_name:
        """ Squeezenet
        """
        model.classifier[1] = nn.Conv2d(512, num_class, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_class

    elif "densenet" in model_name:
        """ Densenet
        """
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_class)
    elif "mobile" in model_name:
        """ Mobilenet
        """
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_class)
    elif "mnas" in model_name:
        """ Mnasnet
        """
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_class)
    else:
        print("Invalid model name, exiting...")
        exit(1)
    return model


def resnet_model(num_classes=10, pretrained=False, frozen=0, which=0):
    print("Is pretrained", pretrained)
    resnet_models = [resnet18, resnet34, resnet50, resnet101, resnet152]
    assert which in range(len(resnet_models))
    model = resnet_models[which](pretrained=pretrained)
    print(resnet_models[which])
    if pretrained and frozen:
        for i, child in enumerate(model.children()):
            print("Freezing layer {}".format(i))
            for param in child.parameters():
                param.requires_grad = False
            if i + 1 >= frozen:
                break
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    print("Number of parameters:", sum([param.nelement() * param.element_size() for param in model.parameters()]))
    # import sys
    # sys.exit(1)
    return model
