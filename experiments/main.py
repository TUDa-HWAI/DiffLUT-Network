import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import random
import json
import datetime

import numpy as np
import torch
from tqdm import tqdm

from difflogic import LogicLayer, GroupSum, CompiledLogicNet, LutlogicLayer, ResidualBlock, DiffLUTNetLayer
from utils.verilog_seperate import export_verilog_modular, export_difflut_net_verilog
torch.set_num_threads(1)

from experiments.datasets import (
    SUPPORTED_DATASETS,
    input_dim_of_dataset,
    load_dataset,
    load_n,
    num_classes_of_dataset,
)
from utils.train import (
    TempScheduler,
    difflut_eval,
    difflut_train,
    eval,
    packbits_eval,
    train,
)

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64
}

METHOD_NAME = 'DiffLUT-Net'

def get_model(args, train_set_size=10000):
    llkw = dict(grad_factor=args.grad_factor, connections=args.connections,implementation=args.implementation)

    in_dim = input_dim_of_dataset(args.dataset)
    class_count = num_classes_of_dataset(args.dataset)

    logic_layers = []

    arch = args.architecture
    k_list = args.num_neurons
    if not isinstance(k_list, list):
        k_list = [k_list]
    l = args.num_layers

    # Helper function to get layer output dimension
    def get_layer_dim(layer_idx):
        if layer_idx < len(k_list):
            return k_list[layer_idx]
        return k_list[-1]

    ####################################################################################################################

    if arch == 'randomly_connected':
        logic_layers.append(torch.nn.Flatten())
        prev_dim = in_dim
        for i in range(l):
            curr_dim = get_layer_dim(i)
            logic_layers.append(LogicLayer(in_dim=prev_dim, out_dim=curr_dim, **llkw))
            prev_dim = curr_dim

        model = torch.nn.Sequential(
            *logic_layers,
            GroupSum(class_count, args.tau)
        )

    elif arch == 'lut_based':
        logic_layers.append(torch.nn.Flatten())
        prev_dim = in_dim
        for i in range(l):
            curr_dim = get_layer_dim(i)
            logic_layers.append(LutlogicLayer(in_dim=prev_dim, out_dim=curr_dim, **llkw))
            prev_dim = curr_dim

        model = torch.nn.Sequential(
            *logic_layers,
            GroupSum(class_count, args.tau)
        )

    elif arch == 'lut_based_resnet':
        logic_layers.append(torch.nn.Flatten())
        prev_dim = in_dim
        for i in range(l):
            curr_dim = get_layer_dim(i)
            logic_layers.append(ResidualBlock(in_dim=prev_dim, out_dim=curr_dim, alpha=0.5, **llkw))
            prev_dim = curr_dim

        model = torch.nn.Sequential(
            *logic_layers,
            GroupSum(class_count, args.tau)
        )

    elif arch == 'difflut_net':
        logic_layers.append(torch.nn.Flatten())
        prev_dim = in_dim
        for i in range(l):
            curr_dim = get_layer_dim(i)
            logic_layers.append(DiffLUTNetLayer(in_dim=prev_dim, out_dim=curr_dim, **llkw))
            prev_dim = curr_dim

        model = torch.nn.Sequential(
            *logic_layers,
            GroupSum(class_count, args.tau)
        )

    else:
        raise NotImplementedError(arch)

    ####################################################################################################################

    total_num_neurons = sum(map(lambda x: x.num_neurons, logic_layers[1:]))
    print(f'total_num_neurons={total_num_neurons}')
    total_num_weights = sum(map(lambda x: x.num_weights, logic_layers[1:]))
    print(f'total_num_weights={total_num_weights}')
    model = model.to('cuda')

    print(model)

    loss_fn = torch.nn.CrossEntropyLoss()

    if args.optimizer == 'difflut':
        from optimizers import DiffLUTOptimizer
        initial_reweight = args.penalty_start if getattr(args, 'anneal_penalty', False) else getattr(args, 'penalty', 1.0)
        gate_params = [p for n, p in model.named_parameters() if 'mapping' not in n]
        optimizer = DiffLUTOptimizer(model, train_set_size=train_set_size, lr=args.learning_rate, num_samples=args.train_samples, lamda_init=args.lamda_init, reweight=initial_reweight, params=gate_params)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    mapping_optimizer = None
    if getattr(args, 'connections', None) == 'learnable':
        mapping_params = [p for n, p in model.named_parameters() if 'mapping' in n]
        if len(mapping_params) > 0:
            mapping_optimizer = torch.optim.Adam(mapping_params, lr=0.01)

    return model, loss_fn, optimizer, mapping_optimizer





if __name__ == '__main__':

    ####################################################################################################################

    parser = argparse.ArgumentParser(description='Train DiffLUT-Net on the supported datasets.')

    parser.add_argument(
        '--dataset', type=str, choices=SUPPORTED_DATASETS, required=True,
        help='the dataset to use',
    )
    parser.add_argument(
        '--tau', '-t', type=float, default=10,
        help='divisor used by the final class-wise GroupSum aggregation',
    )
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed (default: 0)')
    parser.add_argument('--batch-size', '-bs', type=int, default=100, help='batch size (default: 100)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-7, help='learning rate (default: 1e-7)')
    parser.add_argument('--training-bit-count', '-c', type=int, default=32, help='training bit count (default: 32)')

    parser.add_argument('--implementation', type=str, default='cuda', choices=['cuda', 'python'],
                        help='`cuda` is the fast CUDA implementation and `python` is simpler but much slower '
                        'implementation intended for helping with the understanding.')

    parser.add_argument('--packbits_eval', action='store_true', help='Use the PackBitsTensor implementation for an '
                                                                     'additional eval step.')
    parser.add_argument('--compile_model', action='store_true', help='Compile the final model with C for CPU.')

    parser.add_argument('--num-iterations', '-ni', type=int, default=200_000, help='Number of iterations (default: 200_000)')
    parser.add_argument('--eval-freq', '-ef', type=int, default=1_000, help='Evaluation frequency (default: 1_000)')

    parser.add_argument('--valid-set-size', '-vss', type=float, default=0, help='Fraction of the train set used for validation (default: 0.)')
    parser.add_argument('--extensive-eval', action='store_true', help='Additional evaluation (incl. valid set eval).')

    parser.add_argument(
        '--connections', type=str, default='learnable',
        choices=['random', 'unique', 'lut-based', 'learnable'],
        help='input connection strategy (default: learnable)',
    )
    parser.add_argument(
        '--architecture', '-a', type=str, default='difflut_net',
        choices=['difflut_net', 'randomly_connected', 'lut_based', 'lut_based_resnet'],
        help='model architecture (DiffLUT-Net uses the difflut_net identifier)',
    )

    # Optimizer arguments
    parser.add_argument(
        '--optimizer', type=str, default='difflut', choices=['adam', 'difflut'],
        help='optimizer to use (default: difflut)',
    )
    parser.add_argument('--train_samples', type=int, default=0, help='MC samples used by the DiffLUT optimizer during training (default: 0)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--num_neurons', '-k', type=int, nargs='+', required=True,
        help='Number of neurons per layer (can be a list for custom layer sizes)',
    )
    parser.add_argument('--num_layers', '-l', type=int, required=True)

    parser.add_argument('--grad-factor', type=float, default=1.)
    parser.add_argument(
        '--anneal-temp',
        action='store_true',
        help='Enable temperature annealing for LUT logic layers.'
    )
    parser.add_argument('--temp-start', type=float, default=3.0,
                        help='Initial temperature for softmax over gates.')

    parser.add_argument('--temp-end', type=float, default=1.0,
                        help='Final temperature after annealing.')
                        
    parser.add_argument('--no-anneal-lr', dest='anneal_lr', action='store_false',
                        help='Disable cosine learning rate annealing (enabled by default).')
    parser.set_defaults(anneal_lr=True)
    parser.add_argument('--lr-end', type=float, default=1e-9,
                        help='Final learning rate after annealing (default: 1e-9).')
    parser.add_argument('--warmup-steps', type=int, default=5000,
                        help='Number of warmup steps for learning rate scheduling (default: 5000).')

    parser.add_argument('--anneal-penalty', action='store_true',
                        help='Enable penalty (reweight) annealing for the DiffLUT optimizer.')
    parser.add_argument('--penalty', type=float, default=1.0,
                        help='Fixed penalty weight if not annealing.')
    parser.add_argument('--penalty-start', type=float, default=0.0,
                        help='Initial penalty weight (reweight).')
    parser.add_argument('--penalty-end', type=float, default=1.0,
                        help='Final penalty weight (reweight).')

    parser.add_argument(
        "--export_verilog",
        action="store_true",
        help="Whether to export the trained model as a Verilog file"
    )

    parser.add_argument('--lamda_init', type=float, default=10.0, help='Initial value of lamda (default: 10)')

    args = parser.parse_args()
    if args.architecture == 'difflut_net':
        args.method_name = METHOD_NAME

    ####################################################################################################################

    print(vars(args))

    assert args.num_iterations % args.eval_freq == 0, (
        f'iteration count ({args.num_iterations}) has to be divisible by evaluation frequency ({args.eval_freq})'
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    architecture_name = METHOD_NAME if args.architecture == 'difflut_net' else args.architecture
    exp_name = f"{args.dataset}_{architecture_name}_{timestamp}"

    default_results_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'results',
    ))
    results_root = os.environ.get("DIFFLUT_RESULTS_DIR", default_results_root)
    base_dir = os.path.join(results_root, exp_name)
    METRICS_DIR = os.path.join(base_dir, "metrics")

    os.makedirs(METRICS_DIR, exist_ok=True)

    # Save parameters to log file
    with open(f"{METRICS_DIR}/args.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k:25s}: {v}\n")

    ####################################################################################################################
    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ####################################################################################################################
    # Load dataset and model
    train_loader, validation_loader, test_loader = load_dataset(args)
    train_set_size = len(train_loader.dataset) if train_loader is not None else 10000
    model, loss_fn, optimizer, mapping_optimizer = get_model(args, train_set_size)

    best_acc = 0
    best_test_eval_acc = 0.0
    sched = TempScheduler(
        t_start=args.temp_start,
        t_end=args.temp_end,
        total_steps=args.num_iterations,
        scheme="cosine",
        warmup_ratio=0.02,
        hold_ratio=0.03
    )
    history = {
        'train_acc_eval_mode': [],
        'train_acc_train_mode': [],
        'valid_acc_eval_mode': [],
        'valid_acc_train_mode': [],
        'test_acc_eval_mode': [],
        'test_acc_train_mode': [],
        'step': [],
    }

    lr_scheduler = None
    if getattr(args, 'anneal_lr', False):
        def get_warmup_cosine_lambda(warmup_steps, total_steps, lr_end, lr_init):
            end_ratio = lr_end / lr_init if lr_init > 0 else 0.0
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                else:
                    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    progress = min(max(progress, 0.0), 1.0)
                    return end_ratio + 0.5 * (1.0 - end_ratio) * (1.0 + math.cos(math.pi * progress))
            return lr_lambda

        lr_lambda = get_warmup_cosine_lambda(
            warmup_steps=args.warmup_steps,
            total_steps=args.num_iterations,
            lr_end=args.lr_end,
            lr_init=args.learning_rate
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    penalty_sched = None
    if getattr(args, 'anneal_penalty', False) and args.optimizer == 'difflut':
        penalty_sched = TempScheduler(
            t_start=args.penalty_start,
            t_end=args.penalty_end,
            total_steps=args.num_iterations,
            scheme="cosine",
            warmup_ratio=0.0,
            hold_ratio=0.0
        )

    for i, (x, y) in tqdm(
            enumerate(load_n(train_loader, args.num_iterations)),
            desc='iteration',
            total=args.num_iterations,
    ):
        # Step 1: Update temperature (simulated annealing)
        if args.anneal_temp:
            temperature = sched(i)
            for m in model.modules():
                if hasattr(m, "set_temperature"):
                    m.set_temperature(temperature)

        if penalty_sched is not None:
            optimizer.state['reweight'] = penalty_sched(i)

        # Step 2: Normal training
        x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to('cuda')
        y = y.to('cuda')
        if args.optimizer == 'difflut':
            loss = difflut_train(model, x, y, loss_fn, optimizer, mapping_optimizer)
        else:
            loss = train(model, x, y, loss_fn, optimizer)

        if lr_scheduler is not None:
            lr_scheduler.step()

        if (i + 1) % args.eval_freq == 0:
            # Report loss without storing it in history.json.
            print(f"Iteration[{i + 1:05d}] | Loss: {loss:.4f}")

            # Accuracy evaluation
            if args.optimizer == 'difflut':
                # eval_mode uses deterministic binary weights: mu > 0.5
                train_accuracy_eval_mode = difflut_eval(model, train_loader, optimizer)
                # train_mode evaluates the relaxed model with mu = sigmoid(lambda)
                train_accuracy_train_mode = difflut_eval(model, train_loader, optimizer, soft=True)
                
                test_accuracy_eval_mode = difflut_eval(model, test_loader, optimizer)
                test_accuracy_train_mode = difflut_eval(model, test_loader, optimizer, soft=True)

                if len(validation_loader.dataset) > 0:
                    valid_accuracy_eval_mode = difflut_eval(model, validation_loader, optimizer)
                    valid_accuracy_train_mode = difflut_eval(model, validation_loader, optimizer, soft=True)
                else:
                    valid_accuracy_eval_mode = -1
                    valid_accuracy_train_mode = -1
            else:
                if args.extensive_eval and len(validation_loader.dataset) > 0:
                    train_accuracy_train_mode = eval(model, train_loader, mode=True)
                    valid_accuracy_eval_mode = eval(model, validation_loader, mode=False)
                    valid_accuracy_train_mode = eval(model, validation_loader, mode=True)
                else:
                    train_accuracy_train_mode = eval(model, train_loader, mode=True)
                    valid_accuracy_eval_mode = -1
                    valid_accuracy_train_mode = -1
                train_accuracy_eval_mode = eval(model, train_loader, mode=False)
                test_accuracy_eval_mode = eval(model, test_loader, mode=False)
                test_accuracy_train_mode = eval(model, test_loader, mode=True)

            r = {
                'train_acc_eval_mode': train_accuracy_eval_mode,
                'train_acc_train_mode': train_accuracy_train_mode,
                'valid_acc_eval_mode': valid_accuracy_eval_mode,
                'valid_acc_train_mode': valid_accuracy_train_mode,
                'test_acc_eval_mode': test_accuracy_eval_mode,
                'test_acc_train_mode': test_accuracy_train_mode,
            }

            if args.packbits_eval:
                r['train_acc_eval'] = packbits_eval(model, train_loader)
                r['valid_acc_eval'] = packbits_eval(model, train_loader)
                r['test_acc_eval'] = packbits_eval(model, test_loader)

            print(r)

            if test_accuracy_eval_mode > best_test_eval_acc:
                best_test_eval_acc = test_accuracy_eval_mode
                print(f"NEW BEST TEST EVAL ACCURACY: {best_test_eval_acc:.6f}")

            metric_to_track = valid_accuracy_eval_mode if valid_accuracy_eval_mode >= 0 else test_accuracy_eval_mode
            if metric_to_track > best_acc:
                best_acc = metric_to_track
                print(f'IS THE BEST UNTIL NOW: {best_acc:.6f}')


            # Store only steps and accuracy values.
            history['step'].append(i + 1)
            for k, v in r.items():
                history.setdefault(k, []).append(v)


            # Write to file
            with open(f"{METRICS_DIR}/history.json", "w") as f:
                json.dump(history, f, indent=2)
    ####################################################################################################################

    if args.compile_model:
        print('\n' + '='*80)
        print(' Converting the model to C code and compiling it...')
        print('='*80)

        for opt_level in range(4):

            for num_bits in [
                # 8,
                # 16,
                # 32,
                64
            ]:
                os.makedirs('lib', exist_ok=True)
                save_lib_path = f'lib/{args.dataset}_{args.architecture}_{num_bits}.so'

                compiled_model = CompiledLogicNet(
                    model=model,
                    num_bits=num_bits,
                    cpu_compiler='gcc',
                    # cpu_compiler='clang',
                    verbose=True,
                )

                compiled_model.compile(
                    opt_level=1 if (sum(args.num_neurons) if isinstance(args.num_neurons, list) else args.num_layers * args.num_neurons) < 50_000 else 0,
                    save_lib_path=save_lib_path,
                    verbose=True
                )

                correct, total = 0, 0
                with torch.no_grad():
                    for (data, labels) in torch.utils.data.DataLoader(test_loader.dataset, batch_size=int(1e6), shuffle=False):
                        data = torch.nn.Flatten()(data).bool().numpy()

                        output = compiled_model(data, verbose=True)

                        correct += (output.argmax(-1) == labels).float().sum()
                        total += output.shape[0]

                acc3 = correct / total
                print('COMPILED MODEL', num_bits, acc3)

    ####################################################################################################################

    if args.export_verilog:
        print("Exporting Verilog...")

        Verilogwithoutreg_DIR = os.path.join(base_dir, "Verilogwithoutreg")
        Verilogwithreg_DIR = os.path.join(base_dir, "Verilogwithreg")
        os.makedirs(Verilogwithoutreg_DIR, exist_ok=True)
        os.makedirs(Verilogwithreg_DIR, exist_ok=True)
        Verilogwithoutreg_path = os.path.join(Verilogwithoutreg_DIR, "logic_net.v")
        Verilogwithreg_path = os.path.join(Verilogwithreg_DIR, "logic_net.v")

        if args.architecture == 'difflut_net':
            export_difflut_net_verilog(model, Verilogwithoutreg_path, with_reg=False)
            export_difflut_net_verilog(model, Verilogwithreg_path, with_reg=True)
        else:
            export_verilog_modular(model, Verilogwithoutreg_path, with_reg=False)
            export_verilog_modular(model, Verilogwithreg_path, with_reg=True)

    print("\n" + "="*80)
    print(f" TRAINING COMPLETE! Best Test Eval Accuracy: {best_test_eval_acc:.6f}")
    print("="*80 + "\n")
