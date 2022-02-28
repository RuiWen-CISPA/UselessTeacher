import sys
sys.path.append('/home/useless_teacher')
from models import *

def get_model_based_on_name(model_name,num_class, shadow_class=8):
    '''

    :param model_name: model structure name
    :param num_class: how many classes in the dataset, or task
    :return: corresponding model
    '''
    # ResNet 18 / 34 / 50 ****************************************
    if model_name == 'resnet18':
        model = ResNet18(num_class=num_class)
    elif model_name == 'resnet34':
        model = ResNet34(num_class=num_class)
    elif model_name == 'resnet50':
        model = ResNet50(num_class=num_class)

    # ResNet 18 / 34 / 50 ****************************************
    elif model_name == 'advresnet18':
        model = AdvResNet18(num_class=num_class)
    elif model_name == 'censorresnet18':
        model = censorResNet18(num_class=num_class, shadow_classes = shadow_class)
    elif model_name == 'advresnet34':
        model = AdvResNet34(num_class=num_class)
    elif model_name == 'advresnet50':
        model = AdvResNet50(num_class=num_class)

    # PreResNet(ResNet for CIFAR-10)  20/32/56/110 ***************
    elif model_name.startswith('preresnet20'):
        model = PreResNet(depth=20, num_classes=num_class)
    elif model_name.startswith('preresnet32'):
        model = PreResNet(depth=32, num_classes=num_class)
    elif model_name.startswith('preresnet44'):
        model = PreResNet(depth=44, num_classes=num_class)
    elif model_name.startswith('preresnet56'):
        model = PreResNet(depth=56, num_classes=num_class)
    elif model_name.startswith('preresnet110'):
        model = PreResNet(depth=110, num_classes=num_class)

    # DenseNet *********************************************
    elif model_name == 'densenet121':
        model = densenet121(num_class=num_class)
    elif model_name == 'advdensenet121':
        model = advdensenet121(num_class=num_class)
    elif model_name == 'densenet161':
        model = densenet161(num_class=num_class)
    elif model_name == 'densenet169':
        model = densenet169(num_class=num_class)

    # ResNeXt *********************************************
    elif model_name == 'resnext29':
        model = CifarResNeXt(cardinality=8, depth=29, num_classes=num_class)

    elif model_name == 'mobilenetv2':
        model = MobileNetV2(class_num=num_class)

    elif model_name == 'shufflenetv2':
        model = shufflenetv2(class_num=num_class)
    elif model_name == 'advshufflenetv2':
        model = advshufflenetv2(class_num=num_class)

    # vgg *********************************************
    elif model_name == 'vgg19':
        model = vgg19(num_classes=num_class)
    elif model_name == 'vgg16':
        model = vgg16(num_classes=num_class)
    elif model_name == 'Advvgg16':
        model = Advvgg16(num_classes=num_class)
    elif model_name == 'censorvgg16':
        model = censorvgg16(num_classes=num_class, shadow_classes = shadow_class)

    # alexnet *********************************************
    elif model_name == 'alexnet':
        model = AlexNet(num_classes=num_class)
    elif model_name == 'advalexnet':
        model = advAlexNet(num_classes=num_class)
    elif model_name == 'censoralexnet':
        model = censorAlexNet(num_classes=num_class, shadow_classes = shadow_class)


    # Basic neural network ********************************
    # elif model_name == 'net':
    #     model = Net(num_class, params)

    elif model_name == 'mlp':
        model = MLP(num_class=num_class)

    else:
        model = None
        print('Not support for model ' + str(model_name))
        exit()

    return model