"""This is a non-official implementation of 'Stragglers Are Not Disaster: A Hybrid Federated Learning Algorithm with Delayed Gradients' (http://arxiv.org/abs/2102.06329). """
from collections import OrderedDict
import numpy as np
import torch
from flgo.algorithm.fedasync import Server as AsyncServer
from flgo.algorithm.fedbase import BasicClient
import math
import flgo.utils.fmodule as fmodule
import copy

class Server(AsyncServer):
    def initialize(self):
        self.init_algo_para({'lmbd': 0.5, 'buffer_ratio': 0.2})
        self.model_history = []
        self.buffer = []

    def model2numpy(self, model):
        return torch.cat([p.reshape(-1) for p in model.parameters()], dim=0).detach().cpu().numpy()

    def package_handler(self, received_packages:dict):
        if len(self.model_history)<self.current_round: self.model_history.append(self.model2numpy(self.model))
        if self.is_package_empty(received_packages): return False
        received_updates = received_packages['model']
        received_client_taus = [u._round for u in received_updates]
        for cdelta, ctau in zip(received_updates, received_client_taus):
            self.buffer.append((cdelta, ctau))
        if len(self.buffer) >= int(self.buffer_ratio * self.num_clients):
            new_models = []
            for bi in self.buffer:
                update_i, tau_i = bi
                lmbd_i = self.lmbd * math.exp(-(self.current_round-tau_i))
                if tau_i < self.current_round:
                    grad_i = np.expand_dims(self.model2numpy(update_i), -1)
                    # g' = g + R @ (w_t - w_{t-tau})
                    Ri_mul_delta = grad_i@(grad_i.T@(self.model_history[self.current_round-1] - self.model_history[tau_i-1]))/self.learning_rate
                    fixed_part = torch.from_numpy((Ri_mul_delta)).float()
                    update_i = update_i + fmodule._model_from_tensor(fixed_part, update_i.__class__).to(update_i.get_device())
                model_i = (1-lmbd_i)*self.model + lmbd_i*(self.model - update_i)
                new_models.append(model_i)
            self.model = self.aggregate(new_models)
            # clear buffer
            self.buffer = []
            return True
        return False

class Client(BasicClient):
    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        global_model = copy.deepcopy(model)
        self.train(model)
        update = global_model-model
        update._round = model._round
        cpkg = self.pack(update)
        return cpkg