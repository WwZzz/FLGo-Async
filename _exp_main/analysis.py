import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flgo
import argparse
import importlib
import algorithm
import simulator
import model
import matplotlib.pyplot as plt
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

import flgo.experiment.analyzer as fea
# create the analysis plan
records = fea.Selector({'task': task, 'header':args.method, }).records[task]
for ri in records:
    rdata = ri.data
    rname = ri.data['option']['algorithm']
    rtest_acc = rdata['test_accuracy']
    rround = list(range(len(rtest_acc)))
    rtime = rdata['time']
    plt.plot(rtime, rtest_acc, label=rname)
plt.legend()
plt.xlabel('time')
plt.ylabel('test_acc')
plt.show()

for ri in records:
    rdata = ri.data
    rname = ri.data['option']['algorithm']
    rtest_acc = rdata['test_accuracy']
    rround = list(range(len(rtest_acc)))
    rtime = rdata['time']
    plt.plot(rround, rtest_acc, label=rname)
plt.legend()
plt.xlabel('round')
plt.ylabel('test_acc')
plt.show()
# analysis_plan = {
#     'Selector':{'task': task, 'header':args.method, },
#     'Painter':{'Curve':[
#         {'args':{'x':'time', 'y':'test_accuracy'}, 'fig_option':{'xlabel':'time', 'ylabel':'test_acc'}},
#         {'args':{'x':'communication_round', 'y':'test_accuracy'}, 'fig_option':{'xlabel':'round', 'ylabel':'test_acc'}},
#     ]},
#     'Table':{'min_value':[{'x':'val_loss'}]},
# }
#
# fea.show(analysis_plan)