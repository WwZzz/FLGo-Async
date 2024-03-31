"""
This is a non-official implementation of 'Measuring the Effects
of Non-Identical Data Distribution for Federated Visual Classification'
(http://arxiv.org/abs/1909.06335).
"""

from flgo.algorithm.fedavg import Client
from flgo.algorithm.fedbase import BasicServer

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'beta': 0.01})
        self.v = self.model.zeros_like()

    def iterate(self):
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)['model']
        new_model = self.aggregate(models)
        self.v = self.beta*self.v + (self.model - new_model)
        self.model = self.model - self.v
        return