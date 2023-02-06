# Code in this file is refactored and slightly modified code from the DFME project.
# Source: https://github.com/cake-lab/datafree-model-extraction

import numpy as np
import torch
import torch.nn.functional as F

from my_utils import get_standardised_logits, assert_is_distribution, assert_tensor_shape
from ensemble import Ensemble


# Sampling from unit sphere is the method 19 from this website:
# http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
def _gen_evaluation_points(x, m, epsilon):
    assert len(x.size()) == 4, "Expected x to be of the form NxCxHxW"
    n, c, h, w = x.size()
    assert h == w
    u = np.random.randn(n * m * c * h * w).reshape(-1, m, c * h * w)  # generate random points from normal distribution

    d = np.sqrt(np.sum(u ** 2, axis=2)).reshape(-1, m, 1)  # map to a uniform distribution on a unit sphere
    u = torch.Tensor(u / d).view(-1, m, c, h, w)
    u = torch.cat((u, torch.zeros(n, 1, c, h, w)), dim=1)  # Shape N, m + 1, h * w
    u = u.view(-1, m + 1, c, h, w)

    return (x.view(-1, 1, c, h, w).cpu() + epsilon * u).view(-1, c, h, w), u


def _get_model_predictions(model, evaluation_points, device):
    assert not model.training, f"Model expected to be in eval mode."
    assert_tensor_shape(evaluation_points, ("num samples", "c", "h", "w"), "evaluation_points")
    assert len(evaluation_points.size()) == 4
    predictions = []
    max_number_points = 32 * 156  # Hardcoded value to split the large evaluation_points tensor to fit in GPU
    for i in (range(evaluation_points.shape[0] // max_number_points + 1)):
        points = evaluation_points[i * max_number_points: (i + 1) * max_number_points]
        points = points.to(device)
        point_predictions = model(points).detach()
        predictions.append(point_predictions)
    return torch.cat(predictions, dim=0).to(device)


def _prep_model_output_for_loss(prediction, args, is_teacher=False):
    assert_tensor_shape(prediction, ("num samples", "num classes"), "prediction")
    if args.loss == "l1":
        if is_teacher and args.no_logits:
            prediction = get_standardised_logits(prediction, args)
    elif args.loss == "kl":
        if is_teacher:
            prediction = F.softmax(prediction.detach(), dim=1)
        else:
            prediction = F.log_softmax(prediction, dim=1)
    else:
        raise ValueError(args.loss)
    return prediction


def _get_loss(pred_teacher, pred_student, args):
    assert_tensor_shape(pred_student, ("Num Samples * (m+1)", "num classes"), "pred_student")
    assert_tensor_shape(pred_teacher, ("Num Samples * (m+1)", "num classes"), "pred_student")
    if args.loss == "l1":
        return -F.l1_loss(pred_student, pred_teacher, reduction='none').mean(dim=1)
    elif args.loss == "kl":
        assert_is_distribution(pred_teacher)
        return -F.kl_div(pred_student, pred_teacher, reduction='none').sum(dim=1)
    else:
        raise ValueError(args.loss)


def _calculate_forward_differences_gradient(clone_model, pred_victim, x, evaluation_points, u, m, epsilon, device, args):
    n, c, h, w = x.size()
    pred_clone = _get_model_predictions(clone_model, evaluation_points, device)
    pred_clone = _prep_model_output_for_loss(pred_clone, args, is_teacher=False)

    # Compute loss
    loss_values = _get_loss(pred_teacher=pred_victim, pred_student=pred_clone, args=args).view(-1, m + 1)
    assert loss_values.shape == (n, m+1)

    # Compute difference following each direction
    differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)
    assert differences.shape == (n, m)
    differences = differences.view(-1, m, 1, 1, 1)

    # Formula for Forward Finite Differences
    gradient_estimates = 1 / epsilon * differences * u[:, :-1]
    if args.forward_differences:
        gradient_estimates *= c * h * w

    gradient_estimates = gradient_estimates.mean(dim=1).view(-1, c, h, w)
    if args.loss == "l1":
        gradient_estimates /= (args.num_classes * n)

    return gradient_estimates, loss_values


def estimate_gradient_objective(args, victim_model, clone_model, x, epsilon=1e-7, m=5, pre_x=False, device="cpu"):
    if pre_x and args.G_activation is None:
        raise ValueError(args.G_activation)
    
    clone_model_training_setting = clone_model.training
    clone_model.eval()
    assert not victim_model.training, "Teacher model should never be in training mode"

    with torch.no_grad():
        evaluation_points, u = _gen_evaluation_points(x, m, epsilon)
        if pre_x: 
            evaluation_points = args.G_activation(evaluation_points)  # Apply args.G_activation function

        pred_victim = _get_model_predictions(victim_model, evaluation_points, device)
        args.current_query_count += pred_victim.shape[0]
        pred_victim = _prep_model_output_for_loss(pred_victim, args, is_teacher=True)
        u = u.to(device)

        if isinstance(clone_model, Ensemble):
            gradient_estimates = []
            loss_values = []
            for idx in range(clone_model.size()):
                model_gradient_estimates, model_loss_values =\
                    _calculate_forward_differences_gradient(clone_model.get_model_by_idx(idx), pred_victim,
                                                            x, evaluation_points, u, m, epsilon, device, args)
                gradient_estimates.append(model_gradient_estimates)
                loss_values.append(model_loss_values)
            gradient_estimates = torch.mean(torch.stack(gradient_estimates, dim=1), dim=1)
            loss_values = torch.mean(torch.stack(loss_values, dim=1), dim=1)
        else:
            gradient_estimates, loss_values = _calculate_forward_differences_gradient(clone_model, pred_victim,
                                                                                      x, evaluation_points, u, m,
                                                                                      epsilon, device, args)
        clone_model.train(clone_model_training_setting)
        generator_loss = loss_values[:, -1].mean()
        return gradient_estimates.detach(), generator_loss


# TODO Update this
def compute_gradient(args, victim_model, clone_model, x, pre_x=False, device="cpu"):
    if pre_x and args.G_activation is None:
        raise ValueError(args.G_activation)
        
    training = clone_model.training
    clone_model.eval()
    x_copy = x.clone().detach().requires_grad_(True)
    x_ = x_copy.to(device)

    if pre_x:
        x_ = args.G_activation(x_)

    pred_victim = victim_model(x_)
    pred_clone = clone_model(x_)

    if args.loss == "l1":
        loss_fn = F.l1_loss
        if args.no_logits:
            pred_victim = get_standardised_logits(pred_victim, args)
    elif args.loss == "kl":
        loss_fn = F.kl_div
        pred_clone = F.log_softmax(pred_clone, dim=1)
        pred_victim = F.softmax(pred_victim, dim=1)
    else:
        raise ValueError(args.loss)

    loss_values = -loss_fn(pred_clone, pred_victim, reduction='mean')
    # print("True mean loss", loss_values)
    loss_values.backward()

    clone_model.train(training)
    
    return x_copy.grad, loss_values


def measure_true_grad_norm(args, x):
    # Compute true gradient of loss wrt x
    true_grad, _ = compute_gradient(args, args.teacher, args.student, x, pre_x=True, device=args.device)
    true_grad = true_grad.view(-1, 3 * 32 * 32)

    # Compute norm of gradients
    norm_grad = true_grad.norm(2, dim=1).mean().cpu()

    return norm_grad
