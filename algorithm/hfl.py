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
        self.init_algo_para(
            {'lmbd': 0.5, 'buffer_ratio': 0.2,'period': 1})
        self.tolerance_for_latency = 1000
        self.model_history = []
        self.buffer = []

    def model2numpy(self, model):
        return torch.cat([p.reshape(-1) for p in model.parameters()], dim=0).detach().cpu().numpy()

    def pack(self, client_id, mtype=0, *args, **kwargs):
        return {
            'model': copy.deepcopy(self.model),
            'round': self.current_round,
        }

    def iterate(self):
        # record the global model of each round
        if len(self.model_history)<self.current_round: self.model_history.append(self.model2numpy(self.model))
        self.selected_clients = self.sample() if (self.gv.clock.current_time % self.period) == 0 or self.gv.clock.current_time == 1 else []
        self.selected_clients = [cid for cid in self.selected_clients if cid not in [bi[-1] for bi in self.buffer if bi[1]==self.current_round]]
        if len(self.selected_clients) > 0: self.gv.logger.info('Select clients {} at time {}'.format(self.selected_clients, self.gv.clock.current_time))
        # Check the currently received models
        res = self.communicate(self.selected_clients, asynchronous=True)
        received_client_taus= res['round']
        received_client_ids = res['__cid']
        received_updates = res['update']
        received_client_grads = res['grad']
        if len(received_updates) > 0:
            self.gv.logger.info('Receive new models from clients {} at time {}'.format(received_client_ids, self.gv.clock.current_time))
            for cdelta, ctau, cgrad, cid in zip(received_updates, received_client_taus, received_client_grads, received_client_ids):
                self.buffer.append((cdelta, ctau, cgrad, cid))
            if len(self.buffer) >= int(self.buffer_ratio * self.num_clients):
                new_models = []
                for bi in self.buffer:
                    update_i, tau_i, grad_i = bi
                    lmbd_i = self.lmbd*math.exp(-(self.current_round-tau_i))
                    if tau_i<self.current_round:
                        grad_i = np.expand_dims(grad_i, -1)
                        Ri_mul_delta = grad_i@(grad_i.T@(self.model_history[self.current_round-1] - self.model_history[tau_i-1]))
                        fixed_part = torch.from_numpy((Ri_mul_delta)).float()*self.learning_rate
                        update_i = update_i + fmodule._model_from_tensor(fixed_part, update_i.__class__).to(update_i.get_device())
                    model_i = (1-lmbd_i)*self.model + lmbd_i*(self.model - update_i)
                    new_models.append(model_i)
                self.model = self.aggregate(new_models)
                # clear buffer
                self.buffer = []
                return True
        return False

class Client(BasicClient):
    def unpack(self, received_pkg):
        round = received_pkg['round']
        model = received_pkg['model']
        return model, round

    def pack(self, model, round, grad):
        grad = torch.cat([g.reshape(-1) for g in grad.values()], dim=0).cpu().numpy()
        return {'update':model, 'round':round, 'grad':grad}

    def reply(self, svr_pkg):
        model,round  = self.unpack(svr_pkg)
        global_model = copy.deepcopy(model)
        dataloader = self.calculator.get_dataloader(self.train_data, self.batch_size)
        global_model.to(self.device)
        for batch_id, batch_data in enumerate(dataloader):
            batch_data = self.calculator.to_device(batch_data)
            loss = self.calculator.compute_loss(global_model, batch_data)['loss']*len(batch_data[0])
            loss.backward()
        grad = OrderedDict()
        with torch.no_grad():
            for n,p in global_model.named_parameters():
                grad[n] = p.grad/len(self.train_data)
        self.train(model)
        cpkg = self.pack(global_model-model, round, grad)
        return cpkg