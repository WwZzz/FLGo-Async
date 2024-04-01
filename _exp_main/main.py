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
args_parser.add_argument('--method', type=str, default='hfl')
args_parser.add_argument('--task', type=str, default='CIFAR100_DIR0.3_N20')
args_parser.add_argument('--simulator', type=str, default='')
args_parser.add_argument('--max_time', type=str, default=10801000)
args = args_parser.parse_args()

task_model = {
    'CIFAR100': model.ResNet18_CIFAR100
}
try:
    method = importlib.import_module('.'.join(['algorithm',args.method, ]))
except:
    method = None
if method==None: method = importlib.import_module('.'.join(['flgo','algorithm',args.method]))

dataset = args.task[:args.task.find('_')]
Model = task_model.get(dataset, None)
if Model is not None:
    def get_model():
        return Model()
    model = flgo.convert_model(get_model, str(Model.__class__))
else:
    model = None

task = f'../task/{args.task}'
Simulator = None if args.simulator=='' else eval('.'.join(['simulator', args.simulator]))
runner = flgo.init(task, method, {'gpu':0, 'num_epochs':5, 'learning_rate':0.01, 'sample':'uniform_available', 'proportion':1.0}, model=model, Simulator=Simulator)

runner.register_exit_condition(lambda server: server.gv.clock.current_time>args.max_time)
runner.run()