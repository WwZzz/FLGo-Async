import flgo
from flgo.simulator import ResponsivenessExampleSimulator
import flgo.benchmark.mnist_classification as mnist
import flgo.benchmark.partition as fbp
import argparse
import importlib

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--method', type=str, default='gradma')
args = args_parser.parse_args()

method = importlib.import_module('.'.join(['algorithm',args.method, ]))
task = './task/test_task'
flgo.gen_task_by_(mnist, fbp.IIDPartitioner(num_clients=100), task)
flgo.init(task, method, {'gpu':0, 'num_epochs':1, 'learning_rate':0.01, 'sample':'uniform_available'}, Simulator=ResponsivenessExampleSimulator).run()
