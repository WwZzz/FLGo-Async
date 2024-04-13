"""This is a non-official implementation of 'Tackling the Data Heterogeneity in Asynchronous Federated Learning with Cached Update Calibration' (https://openreview.net/forum?id=4aywmeb97I). """
import torch
from flgo.algorithm.fedasync import Server as AsyncServer
from flgo.algorithm.fedbase import BasicClient
import flgo.utils.fmodule as fmodule
import copy

class Server(AsyncServer):
    def initialize(self):
        self.init_algo_para({'buffer_ratio': 0.1, 'eta': 1.0})
        self.buffer = []
        self.hs = [torch.tensor(0.) for _ in self.clients]
        self.ht = torch.tensor(0.).to(self.device)
        self.delta = self.model.zeros_like()

    def package_handler(self, received_packages:dict):
        if self.is_package_empty(received_packages): return False
        received_updates = received_packages['model']
        received_client_ids = received_packages['__cid']
        for cdelta, cid in zip(received_updates, received_client_ids):
            self.delta += (cdelta - self.hs[cid].to(self.device)) if not isinstance(self.hs[cid],torch.Tensor) else cdelta
            self.hs[cid] = cdelta.to('cpu')
            self.buffer.append(cid)
        if len(self.buffer)>= int(self.buffer_ratio * self.num_clients):
            # aggregate and clear updates in buffer
            vt = self.delta / len(self.buffer) + self.ht.to(self.device) if not isinstance(self.ht, torch.Tensor) else self.delta / len(self.buffer)
            self.model = self.model + self.eta * vt
            self.ht = fmodule._model_sum([h for h in self.hs if not isinstance(h, torch.Tensor)]).to(self.device) / self.num_clients
            self.delta = self.model.zeros_like()
            self.buffer = []
            return True
        return False

class Client(BasicClient):
    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        global_model = copy.deepcopy(model)
        self.train(model)
        update = model - global_model
        cpkg = self.pack(update)
        return cpkg