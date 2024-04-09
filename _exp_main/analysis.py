import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flgo
import argparse
import importlib
import algorithm
import simulator
import model

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--method', nargs='+', type=str, default=['fedavg'])
args_parser.add_argument('--task', type=str, default='CIFAR100_DIR0.3_N20')
args_parser.add_argument('--simulator', type=str, default='')
args_parser.add_argument('--max_time', type=int, default=10801000)
args_parser.add_argument('--gpu', type=int, default=0)
args = args_parser.parse_args()

dataset = args.task[:args.task.find('_')]

task = f'../task/{args.task}'
Simulator = None if args.simulator=='' else eval('.'.join(['simulator', args.simulator]))

import flgo.experiment.analyzer
# create the analysis plan
analysis_plan = {
    'Selector':{'task': task, 'header':args.method, },
    'Painter':{'Curve':[{'args':{'x':'time', 'y':'val_accuracy'}}]},
    'Table':{'min_value':[{'x':'val_loss'}]},
}

flgo.experiment.analyzer.show(analysis_plan)