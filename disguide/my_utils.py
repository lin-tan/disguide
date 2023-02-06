import os
import numpy as np
import torch
import torch.nn.functional as F

import network
from network.wrn import WideResNet
from cifar10_models import *
from ensemble import Ensemble

import config


def perm(pop_size, num_samples, device):
    """Use torch.randperm to generate num_samples indices on a tensor of size pop_size."""
    # Source: https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/19
    return torch.randperm(pop_size, device=device)[:num_samples]


def print_and_log(statements):
    """Log the print statements"""
    print(statements)
    config.log_file.write(statements)
    config.log_file.write("\n")
    config.log_file.flush()


# Compute cost_per iteration based on experiment type
def compute_cost_per_iteration(args):
    """Compute the query budget cost based on experiment type and other args"""
    if args.experiment_type == 'dfme':
        cost_per_iteration = args.batch_size * (args.g_iter * (args.grad_m + 1) + args.d_iter)
    elif args.experiment_type == 'disguide':
        cost_per_iteration = args.batch_size * args.d_iter
    elif args.experiment_type == 'hl':  # Legacy, ignore
        if args.dset_size % args.batch_size == 0:
            cost_per_iteration = args.dset_size
        else:
            cost_per_iteration = args.dset_size - (args.dset_size % args.batch_size) + args.batch_size
    else:
        raise KeyError(f"Unexpected experiment type: {args.experiment_type}")
    return cost_per_iteration


# Return the teacher prediction for a given sample with or without correction
def get_teacher_prediction(args, teacher, sample):
    """Get teacher predictions. Enforces that gradients cannot flow through the teacher model and standardizes logits"""
    with torch.no_grad():
        t_logit = teacher(sample)
        # Correction for the fake logits
        if args.loss == "l1" and args.no_logits:
            t_logit = get_standardised_logits(t_logit, args)
        args.current_query_count += t_logit.shape[0]
    return t_logit


def student_loss(s_logit, t_logit, args):
    """Get loss for student logit determined by args.loss"""
    if args.loss == "l1":
        return F.l1_loss(s_logit, t_logit.detach())
    elif args.loss == "kl":
        s_logit = F.log_softmax(s_logit, dim=1)
        t_logit = F.softmax(t_logit, dim=1)
        return F.kl_div(s_logit, t_logit.detach(), reduction="batchmean")
    elif args.loss == "hl":
        assert len(t_logit.shape) == 2
        return F.cross_entropy(s_logit, t_logit.detach().argmax(dim=1))
    else:
        raise ValueError(args.loss)


# Function to transform true logits to a form you could get from softmax outputs.
# This is in order to avoid cheating.
def get_standardised_logits(logit, args):
    """Get standardised logit values. args.logit_correction controls standardisation type. One of min or mean."""
    assert len(logit.shape) == 2
    logit = F.log_softmax(logit, dim=1).detach()
    if args.logit_correction == 'min':
        logit -= logit.min(dim=1).values.view(-1, 1).detach()
    elif args.logit_correction == 'mean':
        logit -= logit.mean(dim=1).view(-1, 1).detach()
    return logit


def assert_is_distribution(x):
    """Verify given input could represent a statistical distribution."""
    assert torch.min(x) >= 0., "Distribution cannot be negative"
    assert torch.max(x) <= 1., "Distribution max cannot be above 1.0"
    dist = torch.sum(x, dim=-1)
    eps = 0.00001
    assert torch.min(dist) + eps > 1.0 > torch.max(dist) - eps, "Distribution sum should be within epsilon of 1.0"


def assert_tensor_shape(x, dim_names, name="tensor", dim_shapes=()):
    """Verify tensor has expected shape. Has verbose printing in case of assertion failure."""
    assert len(x.shape) == len(dim_names), f"Expected {name} to have shape {dim_names}. Actual shape:{x.shape}"
    for i in range(len(dim_shapes)):
        if dim_shapes[i] is not None:
            assert x.shape[i] == dim_shapes[i], f"Expected {name} dim {i} to be {dim_shapes[i]}. Actual:{x.shape[i]}"


def get_dataset_min_max(data_loader):
    """Compute dataset min and max values"""
    it = 0
    for data, target in data_loader:
        if it == 0:
            min_val = data.min()
            max_val = data.max()
        min_val = min(min_val, data.min())
        max_val = max(max_val, data.max())
    return float(min_val), float(max_val)


def get_dataset_histogram(data_loader, bins=10):
    """Compute histogram of dataset features"""
    assert isinstance(bins, int) or (isinstance(bins, torch.Tensor) and len(bins.shape) == 1)
    if isinstance(bins, int):
        data_range = get_dataset_min_max(data_loader)
    histogram = histogram_bins = None
    for data, target in data_loader:
        if isinstance(bins, int):
            data_histogram, histogram_bins = torch.histogram(data, bins, range=data_range)
        else:
            data_histogram, histogram_bins = torch.histogram(data, bins)
        if histogram is None:
            histogram = torch.zeros(data_histogram.shape[0])
        histogram += data_histogram
    return histogram, histogram_bins


def print_dataset_feature_distribution(data_loader):
    """Print histogram of dataset features"""
    min_val, max_val = get_dataset_min_max(data_loader)
    print(f"Dataset min:{min_val}, max:{max_val}")
    histogram, histogram_bin_edges = get_dataset_histogram(data_loader, bins=10)
    print(f"Data Density Histogram: {histogram / histogram.sum()}")
    print(f"Histogram Bin Edges:{histogram_bin_edges}")


def get_model_preds_and_true_labels(model, loader, device="cuda"):
    """Compute the predictions for a model on a dataset and return this together with the true labels"""
    model.eval()
    targets = []
    preds = []
    with torch.no_grad():
        for (data, target) in loader:
            data, target = data.to(device), target.to(device)
            targets.append(target)
            output = model(data)
            assert len(output.shape) == 2 or (len(output.shape) == 3 and isinstance(model, Ensemble))
            preds.append(F.softmax(output, dim=-1))
    targets = torch.cat(targets, dim=0)
    preds = torch.cat(preds, dim=0)
    return preds, targets


def compute_grad_norms(generator, student):
    generator_gradients = []
    for n, p in generator.named_parameters():
        if "weight" in n:
            generator_gradients.append(p.grad.norm().to("cpu"))

    student_gradients = []
    for n, p in student.named_parameters():
        if "weight" in n:
            student_gradients.append(p.grad.norm().to("cpu"))
    return np.mean(generator_gradients), np.mean(student_gradients)


def make_dir_if_not_exists(path):
    """Ensure directory path exists and create it if it doesn't exist yet."""
    if not os.path.exists(path):
        os.makedirs(path)


def init_or_append_to_log_file(path, filename, header, mode="w"):
    with open(os.path.join(path, filename), mode) as f:
        f.write(header + "\n")


def get_classifier(classifier, pretrained=True, num_classes=10, ensemble_size=16, device=None):
    """Initialize classifier based on parameters"""
    if classifier == "wrn-28-10":
        net = WideResNet(
                    num_classes=num_classes,
                    depth=28,
                    widen_factor=10,
                    dropRate=0.3
                )
        if pretrained:
            if device is None:
                raise NotImplementedError
            state_dict = torch.load("cifar100_models/state_dicts/model_best.pt", map_location=device)["state_dict"]
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)

        return net
    elif 'wrn' in classifier and 'kt' not in classifier:
        depth = int(classifier.split("-")[1])
        width = int(classifier.split("-")[2])

        net = WideResNet(depth=depth, num_classes=num_classes, widen_factor=width)
        if pretrained:
            raise ValueError("Cannot be pretrained")
        return net
    elif classifier == "kt-wrn-40-2":
        net = WideResNetKT(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.0)
        if pretrained:
            state_dict = torch.load("cifar10_models/state_dicts/kt_wrn.pt", map_location=device)["state_dict"]
            net.load_state_dict(state_dict)
        return net
    elif classifier == "resnet34_8x":
        if pretrained:
            raise ValueError("Cannot load pretrained resnet34_8x from here")
        return network.resnet_8x.ResNet34_8x(num_classes=num_classes)
    elif classifier == "ensemble_resnet18_8x":
        if pretrained:
            raise ValueError("Cannot load pretrained resnet18_8x from here")
        return Ensemble(ensemble_size=ensemble_size, num_classes=num_classes, student_model="ensemble_resnet18_8x")
    elif classifier == "resnet18_8x":
        if pretrained:
            raise ValueError("Cannot load pretrained resnet18_8x from here")
        return network.resnet_8x.ResNet18_8x(num_classes=num_classes)
    elif classifier == "ensemble_lenet5_half":
        if pretrained:
            raise ValueError("Cannot load pretrained lenet5_half from here")
        return Ensemble(ensemble_size=ensemble_size, num_classes=num_classes, student_model="ensemble_lenet5_half")
    elif classifier == "lenet5":
        if pretrained:
            raise ValueError("Cannot load pretrained lenet5 from here")
        return network.lenet.LeNet5()
    
    elif classifier == "lenet5_half":
        if pretrained:
            raise ValueError("Cannot load pretrained lenet5_half from here")
        return network.lenet.LeNet5Half()
    
    else:
        raise NameError('Please enter a valid classifier')


classifiers = [
    "ensemble_resnet18_8x",
    "ensemble_lenet5_half",
    "resnet34_8x", # Default DFAD
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "densenet121",
    "densenet161",
    "densenet169",
    "mobilenet_v2",
    "googlenet",
    "inception_v3",
    "wrn-28-10",
    "resnet18_8x",
    "kt-wrn-40-2",
    "lenet5",
    "lenet5_half",
]
