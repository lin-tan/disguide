# Experiment metric generation and tensorboard storage handling

import os
import torch
from torchmetrics.functional import accuracy, precision, recall, f1_score

from my_utils import assert_is_distribution, assert_tensor_shape

_hparams = {}
_hpmetrics = {}


class _F:
    """Helper class to handle tensorboard metrics"""
    def __init__(self, metric, avg='macro'):
        self.metric = metric
        self.avg = avg
    
    def __call__(self, preds, labels, by_class=False):
        assert(len(preds.shape) == 2 and len(labels.shape) == 1) 
        num_classes = preds.shape[-1]
        if by_class:
            return self.metric(preds, labels, num_classes=num_classes, average=None)
        assert preds.device == labels.device, f"Expected preds and labels on same device. preds:{preds.device} labels:{labels.device}"
        return self.metric(preds, labels, num_classes=num_classes, average=self.avg)


def _one_hot(x, num_classes=-1):
    """Transforms last dimension of tensor input to one hot encoding, assumes last dimension is a class label"""
    return torch.nn.functional.one_hot(x, num_classes=num_classes)


def _probabilities_to_one_hot(x, dim=-1):
    """Transforms last dimension of tensor input to one hot encoding, assumes last dimension is a distribution"""
    return _one_hot(x.argmax(dim=dim), num_classes=x.shape[dim]).to(torch.float32)


def _add_symmetric_quantile(results, metric_stats, lb_name, ub_name, lb_quantile):
    if "Median" not in metric_stats:
        metric_stats["Median"] = torch.quantile(results, 0.5).item()
    metric_stats[lb_name] = torch.quantile(results, lb_quantile).item()
    metric_stats[ub_name] = torch.quantile(results, 1.0-lb_quantile).item()


def _assert_data_fit_for_metrics(preds, labels, expected_pred_dims):
    """Verify data fits expected format"""
    assert len(labels.shape) == 1, f"Expected labels to have shape (N). Actual shape:{labels.shape}"
    assert torch.min(labels) >= 0, f"Expected class labels to be integers from 0 to num_classes-1. " \
                                   f"Actual min:{torch.min(labels)} and max{torch.max(labels)}"
    assert torch.max(labels) < preds.shape[-1],\
        f"Expected preds last dim to be number of classes and larger than max label value. " \
        f"Preds shape:{preds.shape} and Label max:{torch.max(labels)}"
    assert_is_distribution(preds)
    assert preds.device == labels.device,\
        f"Expected preds and labels on same device. preds:{preds.device} labels:{labels.device}"
    if expected_pred_dims == 2:
        assert_tensor_shape(preds, ("N", "num_classes"), "preds")
    else:
        assert_tensor_shape(preds, ("N", "num_models", "num_classes"), "preds")


def _get_class_distribution_quantiles(num_classes):
    if num_classes == 10:
        return [("Min", "Max", 0.0), ("2nd Worst Class", "2nd Best Class", 1.0 / 9.0)]
    elif num_classes == 100:
        return [("Min", "Max", 0.0), ("2nd Worst Class", "2nd Best Class", 1.0 / 99.0)]
    else:
        raise NotImplementedError
    

def _get_primary_model_metrics():
    return [("Accuracy", _F(accuracy, 'micro')), ("Precision", _F(precision)), ("Recall", _F(recall)), ("F1 Score", _F(f1_score))]


def _gen_primary_model_metrics(stats, preds, labels, by_class):
    _assert_data_fit_for_metrics(preds, labels, expected_pred_dims=2)
    metrics = _get_primary_model_metrics()
    
    num_classes = preds.shape[1]
    class_quantiles = _get_class_distribution_quantiles(num_classes)
    stats["By Class"] = dict()
    with torch.no_grad():
        for metric_name, metric in metrics:
            stats[metric_name] = metric(preds, labels).item()
            if not by_class:
                continue
            by_class_metric = metric(preds, labels, by_class=True)
            assert(by_class_metric.shape[0] == num_classes)
            stats["By Class"][metric_name] = dict()
            for lb_name, ub_name, quantile in class_quantiles:
                _add_symmetric_quantile(by_class_metric, stats["By Class"][metric_name], lb_name, ub_name, quantile)


def _get_ensemble_distribution_quantiles():
    return [("Min", "Max", 0.0)]


def _get_ensemble_metrics():
    return [("Accuracy", _F(accuracy, 'micro')), ("Precision", _F(precision)),
            ("Recall", _F(recall)), ("F1 Score", _F(f1_score))]


def _gen_ensemble_metrics(stats, preds, labels):
    _assert_data_fit_for_metrics(preds, labels, expected_pred_dims=3)
    metrics = _get_ensemble_metrics()
    
    ensemble_quantiles = _get_ensemble_distribution_quantiles()
    with torch.no_grad():
        for metric_name, metric in metrics:
            stats[metric_name] = dict()
            results = []
            for i in range(preds.shape[1]):
                results.append(metric(preds[:,i,:], labels))
            results = torch.stack(results, dim=0)
            if len(results.shape) != 1:
                raise NotImplementedError
            for lb_name, ub_name, quantile in ensemble_quantiles:
                _add_symmetric_quantile(results, stats[metric_name], lb_name, ub_name, quantile)


def _gen_individual_model_metrics(stats, preds, labels):
    _assert_data_fit_for_metrics(preds, labels, expected_pred_dims=3)
    
    metrics = _get_primary_model_metrics()
    with torch.no_grad():
        for idx in range(preds.shape[1]):
            model_id = f"Model {idx}"
            stats[model_id] = dict()
            for metric_name, metric in metrics:
                stats[model_id][metric_name] = metric(preds[:,idx,:], labels).item()


def _gen_intra_ensemble_metrics(stats, preds):
    assert_tensor_shape(preds, ("N", "num_models", "num_classes"), "preds")
    if preds.shape[1] < 2:
        return
    stats["Variance"] = torch.mean(torch.var(preds, unbiased=False, dim=1))
    stats["Variance Unbiased"] = torch.mean(torch.var(preds, unbiased=True, dim=1))
    one_hot_preds = _probabilities_to_one_hot(preds, dim=2)
    stats["Class Prediction Variance"] = torch.mean(torch.var(one_hot_preds, unbiased=False, dim=1))
    stats["Class Prediction Variance Unbiased"] = torch.mean(torch.var(one_hot_preds, unbiased=True, dim=1))


def get_ensemble_metrics(preds, labels, by_class=True):
    _assert_data_fit_for_metrics(preds, labels, expected_pred_dims=3)
    
    soft_vote_preds = torch.mean(preds, dim=1)
    hard_vote_preds = torch.mean(_probabilities_to_one_hot(preds, dim=2), dim=1)
    stats = {"Ensemble": dict(), "Soft Vote": dict(), "Hard Vote": dict(),
             "Individual": dict(), "Intra Ensemble": dict()}
    
    _gen_ensemble_metrics(stats["Ensemble"], preds, labels)
    _gen_primary_model_metrics(stats["Soft Vote"], soft_vote_preds, labels, by_class=by_class)
    _gen_primary_model_metrics(stats["Hard Vote"], hard_vote_preds, labels, by_class=by_class)
    _gen_individual_model_metrics(stats["Individual"], preds, labels)
    _gen_intra_ensemble_metrics(stats["Intra Ensemble"], preds)
    return stats


def _prep_hparam_logs(args):
    global _hparams
    global _hpmetrics
    hparam_keys = ['ensemble_size', 'grad_m', 'grad_epsilon', 'epoch_itrs', 'lr_S', 'lr_G', 'g_iter', 'd_iter',
                   'query_budget', 'nz', 'batch_size','momentum', 'weight_decay', 'scheduler', 'loss', 'dataset',
                   'cudnn_deterministic']
    for hparam_key in hparam_keys:
        _hparams[hparam_key] = getattr(args, hparam_key)
    for metric_name, _ in _get_primary_model_metrics():
        _hpmetrics[metric_name] = 0


def _assert_log_dir_path_ok(writer, args):
    head, tail = os.path.split(writer.log_dir)
    assert tail == args.experiment_name, f"Expected writer log directory to be {args.experiment_name} but was {tail} with path={head}"
    head, tail = os.path.split(head)
    assert tail == "general", f"Expected writer log directory parent to be general but was {tail} with path={head}"


def _update_hparam_metrics(stats):
    global _hpmetrics
    for key in _hpmetrics:
        _hpmetrics[key] = max(_hpmetrics[key], stats["Soft Vote"][key], stats["Hard Vote"][key])


def _add_to_custom_layout(layout, scalar_group, metrics, quantiles, metric_prefix=""):
    for metric_name, metric in metrics:
        for lb_name, ub_name, quantile in quantiles:
            description = ['Margin', [f"{scalar_group}/{metric_prefix}{metric_name}/Median",
                                      f"{scalar_group}/{metric_prefix}{metric_name}/{lb_name}",
                                      f"{scalar_group}/{metric_prefix}{metric_name}/{ub_name}"]]
            layout[f"{scalar_group}"][f"{metric_prefix}{metric_name}/{lb_name}-{ub_name}"] = description


def _prep_custom_scalars(writer, num_classes):
    ensemble_metrics = _get_ensemble_metrics()
    ensemble_distribution_quantiles = _get_ensemble_distribution_quantiles()
    primary_model_metrics = _get_primary_model_metrics()
    class_distribution_quantiles = _get_class_distribution_quantiles(num_classes)
    
    layout = {"Ensemble":dict(), "Soft Vote":dict(), "Hard Vote":dict()}
    _add_to_custom_layout(layout, "Ensemble", ensemble_metrics, ensemble_distribution_quantiles)
    _add_to_custom_layout(layout, "Soft Vote", primary_model_metrics, class_distribution_quantiles, "By Class/")
    _add_to_custom_layout(layout, "Hard Vote", primary_model_metrics, class_distribution_quantiles, "By Class/")
    writer.add_custom_scalars(layout)


def _log_multi_model_metrics(writer, stats, log_iter, args):
    # TODO Either remove or fix creation of many log files
    if args.ensemble_size < 2 or True:
        return
    metrics = _get_primary_model_metrics()
    for metric_name, _ in metrics:
        metric_dict = {}
        for idx in range(args.ensemble_size):
            model_name = f"Model {idx}"
            metric_dict[model_name] = stats["Individual"][model_name][metric_name]
        writer.add_scalars(f"Multimodel/{metric_name}", metric_dict, log_iter)


def _log_metrics(writer, stats, idx, prefix):
    if not isinstance(stats, dict):
        writer.add_scalar(prefix, stats, idx)
        return
    for key, value in stats.items():
        _log_metrics(writer, value, idx, prefix + "/" + key)


def eval_and_log_metrics(writer, preds, labels, args):
    """Compute and log metrics with writer for given preds, labels, and args"""
    if not _hparams:
        _prep_custom_scalars(writer, args.num_classes)
        _prep_hparam_logs(args)
    idx = args.current_query_count
    stats = get_ensemble_metrics(preds, labels)
    for key, value in stats.items():
        assert(isinstance(value, dict))
        _log_metrics(writer, value, idx, key)
    _log_multi_model_metrics(writer, stats, idx, args)
    _update_hparam_metrics(stats)
    return stats


def eval_and_log_validation_metrics(writer, preds, teacher_preds, args, tag="Validation"):
    """Compute and log train metrics with writer for given preds, labels, and args"""
    labels = teacher_preds.argmax(dim=1)
    stats = get_ensemble_metrics(preds, labels, by_class=False)
    _log_metrics(writer, stats, args.current_query_count, tag)
    return stats


def log_hparams(writer, args):
    """Log hyperparameters at the end of training"""
    global _hparams
    global _hpmetrics
    hp_store_metric = {}
    for key in _hpmetrics:
        hp_store_metric[f"hparams/{key}"] = _hpmetrics[key]
    _assert_log_dir_path_ok(writer, args)
    writer.add_hparams(_hparams, hp_store_metric, run_name=f"../../hparams/{args.experiment_name}")


def log_generator_distribution(writer, teacher_preds, args):
    """Log class histogram for generated classes. Useful for analyzing a generators' behavior"""
    num_classes = teacher_preds.shape[1]
    class_preds = teacher_preds.argmax(dim=1)
    class_preds_adjusted = torch.cat([class_preds, torch.arange(num_classes, device=args.device)])
    writer.add_histogram("Generator/Class Distribution", class_preds_adjusted, args.current_query_count,
                         bins=num_classes)
