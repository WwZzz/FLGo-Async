"""This is a non-official implementation of 'Asynchronous Federated Optimization' (http://arxiv.org/abs/1903.03934). """
from flgo.algorithm.asyncbase import AsyncServer
from flgo.algorithm.fedprox import Client

class Server(AsyncServer):
    def initialize(self):
        self.init_algo_para(
            {'alpha': 0.6, 'mu': 0.005, 'flag': 'poly', 'hinge_a': 10, 'hinge_b': 6, 'poly_a': 0.5})

    def package_handler(self, received_packages:dict):
        if self.is_package_empty(received_packages): return False
        received_models = received_packages['model']
        taus = [m._round for m in received_models]
        alpha_ts = [self.alpha * self.s(self.current_round - tau) for tau in taus]
        currently_updated_models = [(1 - alpha_t) * self.model + alpha_t * model_k for alpha_t, model_k in zip(alpha_ts, received_models)]
        self.model = self.aggregate(currently_updated_models)
        return True

    def s(self, delta_tau):
        if self.flag == 'constant':
            return 1
        elif self.flag == 'hinge':
            return 1 if delta_tau <= self.hinge_b else 1.0 / (self.hinge_a * (delta_tau - self.hinge_b))
        elif self.flag == 'poly':
            return (delta_tau + 1) ** (-self.poly_a)
