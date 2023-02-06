# Command line parsing helper file.
# This exists to simplify testing completely new methods and reduce the amount of code in primary training scripts

import argparse
import random
import os
from my_utils import classifiers


def _add_generic_args(parser):
    parser.add_argument('--epoch-itrs', type=int, default=50)
    parser.add_argument('--run-idx', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--query-budget', type=float, default=20, metavar='N',
                        help='Query budget for the extraction attack in millions (default: 20M)')
    parser.add_argument('--lr-S', type=float, default=0.1, metavar='LR', help='Student learning rate (default: 0.1)')
    parser.add_argument('--lr-G', type=float, default=1e-4, help='Generator learning rate (default: 0.1)')
    parser.add_argument('--nz', type=int, default=256, help='Size of random noise input to generator')
    parser.add_argument('--suffix', type=str, default=None,
                        help='Optional suffix to append at the end of the experiment name.')

    parser.add_argument('--rep-iter', type=int, default=1, help="Number of consecutive times to sample from replay")
    parser.add_argument('--separate-student-replay', action='store_true',
                        help='Whether to give students different samples from replay.')
    parser.add_argument('--lambda-div', type=float, default=0,
                        help='Penalty weight for generator class diversity in PV training'
                             'TODO: Connect this to DFMS as well')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'kl', 'hl'],
                        help='Whether to train student with l1 loss or kl divergence in soft label setting.'
                             'Alternatively used to specify hard label setting.')
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cosine', "none"], )
    parser.add_argument('--steps', nargs='+', default=[0.1, 0.3, 0.5], type=float,
                        help="Percentage epochs at which to take next step")
    parser.add_argument('--scale', type=float, default=3e-1, help="Fractional decrease in lr")
    parser.add_argument('--replay', type=str, default='Off', choices=['Off', 'Classic'])
    parser.add_argument('--replay-size', type=int, default=100000, help='Experience replay size')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['svhn', 'cifar10', 'mnist', 'cifar100'],
                        help='dataset name (default: cifar10)')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--grayscale', type=int, default=0,
                        help='Control grayscale generator generation. If set to value >= 1, Nth samples to grayscale.')

    parser.add_argument('--model', type=str, default='resnet34_8x', choices=classifiers,
                        help='Target model name (default: resnet34_8x)')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cudnn-deterministic', action='store_true', default=False,
                        help='cudnn determinism setting')
    parser.add_argument('--seed', type=int, default=random.randint(0, 100000), metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/cifar10-resnet34_8x.pt')

    parser.add_argument('--student-load-path', type=str, default=None)
    parser.add_argument('--model-id', type=str, default="debug")
    parser.add_argument('--initial-query-override', type=int, default=0,
                        help='Set to override initial number of queries. Use if training is split between calls.')

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log-dir', type=str, default="results")
    parser.add_argument('--checkpoint-dir', type=str, default="checkpoint")
    parser.add_argument('--store-checkpoints', type=int, default=1)
    parser.add_argument('--generator-store-frequency', type=int, default=4)
    parser.add_argument('--input-space', type=str, default="pre-transform",
                        choices=["pre-transform", "post-transform"],
                        help="Whether inputs are assumed to be pre- (realistic) or post-transform."
                             "Pre-transform assumes attacker does not have access to server side image transforms.")
    parser.add_argument('--experiment-type', type=str, default="disguide",
                        choices=["dfme", "disguide"],
                        help="Which experiment type you want to run")

    parser.add_argument('--student-model', type=str, default='ensemble_resnet18_8x',
                        help='Student model architecture (default: ensemble_resnet18_8x)')
    parser.add_argument('--phase', type=str, default=None, help='Name of stage in multi-stage training.'
                                                                'This option may not make sense for some algorithms.')


def _add_separate_training_args(parser):
    parser.add_argument('--g-iter', type=int, default=1, help="Number of generator iterations per epoch_iter")
    parser.add_argument('--d-iter', type=int, default=5, help="Number of discriminator iterations per epoch_iter")


def _add_approx_grad_args(parser):
    parser.add_argument('--approx-grad', type=int, default=1, help='Always set to 1')
    parser.add_argument('--grad-m', type=int, default=1, help='Number of steps to approximate the gradients')
    parser.add_argument('--grad-epsilon', type=float, default=1e-3)
    parser.add_argument('--forward-differences', type=int, default=1, help='Always set to 1')


def _add_soft_label_args(parser):
    parser.add_argument('--logit-correction', type=str, default='mean', choices=['none', 'mean'])
    parser.add_argument('--rec-grad-norm', type=int, default=1)
    parser.add_argument('--no-logits', type=int, default=1)
    parser.add_argument('--MAZE', type=int, default=0)  # MAZE approach requires soft labels


def _run_post_parsing_processing(args, set_total_query_budget, soft_label):
    if set_total_query_budget:
        args.query_budget *= 10 ** 6
        args.query_budget = int(args.query_budget)
        args.total_query_budget = args.query_budget
        args.current_query_count = 0
        if args.initial_query_override:
            args.current_query_count = args.initial_query_override
            args.query_budget -= args.current_query_count
    else:
        print(f"WARNING: Total query budget not set by parse_args call. Initialization expected outside of "
              f"function call. Please set_total_query_budget to in parse_args argument."
              f"This will be default soon and warning removed.")

    if soft_label and args.MAZE != 0:
        raise NotImplementedError

    if args.student_model not in classifiers:
        raise ValueError("Unknown model")

    if args.student_model not in classifiers:
        if "wrn" not in args.student_model:
            raise ValueError("Unknown model")

    args.use_ensemble = (args.ensemble_size is not None) and args.ensemble_size > 0
    assert (args.ensemble_size is not None) or args.use_ensemble == args.ensemble_size > 0


def _gen_experiment_name_for_idx(run_idx, args):
    if args.experiment_type == "dfme":
        base_name = "tME"
    elif args.experiment_type == "disguide":
        base_name = "tDG"
    else:
        raise ValueError("Unknown experiment type")
    
    experiment_name = f"{base_name}_{args.ensemble_size}_X{args.replay}"
    if args.loss == "hl":
        experiment_name += "_hl"
    if args.input_space != "pre-transform":
        assert args.input_space == "post-transform"
        experiment_name += "_pt"
    if args.suffix is not None and args.suffix.lower() != "none":
        experiment_name += f"_S-{args.suffix}"
    experiment_name += f"_r{run_idx}"
    return experiment_name


def _set_experiment_name_and_log_dir(args):
    if args.run_idx is not None:
        args.experiment_name = _gen_experiment_name_for_idx(args.run_idx, args)
        args.experiment_dir = f"Experiments/{args.experiment_name}"
    else:
        for idx in range(512):
            args.experiment_name = _gen_experiment_name_for_idx(idx, args)
            args.experiment_dir = f"Experiments/{args.experiment_name}"
            if not os.path.exists(args.experiment_dir):
                break  # A suitable run_idx has been found

    args.log_dir = f"{args.experiment_dir}/{args.log_dir}"
    if args.phase:
        args.log_dir += f"/{args.phase}"


def _add_parser_settings_to_args(args, ensemble, approx_grad, separate_training, soft_label,
                                 set_total_query_budget):
    cli_arg_set = {"ensemble":ensemble, "approx_grad": approx_grad,
                   "separate_training": separate_training, "soft_label": soft_label,
                   "set_total_query_budget": set_total_query_budget}
    args.cli_arg_set = cli_arg_set


def parse_args(description='Model Extraction CIFAR', ensemble=True, approx_grad=True,
               separate_training=True, soft_label=True, set_total_query_budget=False):
    """Build and parse CLI arguments. Initializes certain constants (E.G. Experiment name) based on CLI arguments."""
    parser = argparse.ArgumentParser(description=description)
    _add_generic_args(parser)

    if separate_training:
        _add_separate_training_args(parser)

    if ensemble:
        parser.add_argument('--ensemble-size', type=int, default=4,
                            help="Ensemble size to use for student. 0 results in call to baseline.")

    if approx_grad:
        _add_approx_grad_args(parser)

    if soft_label:
        _add_soft_label_args(parser)

    args = parser.parse_args()
    _run_post_parsing_processing(args, set_total_query_budget, soft_label)
    _set_experiment_name_and_log_dir(args)
    return args


