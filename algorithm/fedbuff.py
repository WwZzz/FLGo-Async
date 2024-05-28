"""This is a non-official implementation of 'Federated Learning with Buffered Asynchronous Aggregation' (http://arxiv.org/abs/2106.06639). """
from flgo.algorithm.asyncbase import AsyncServer
from flgo.algorithm.fedbase import BasicClient
import flgo.utils.fmodule as fmodule
import copy

class Server(AsyncServer):
    def initialize(self):
        self.init_algo_para({'buffer_ratio': 0.1, 'eta': 1.0})
        self.buffer = []

    def package_handler(self, received_packages:dict):
        if self.is_package_empty(received_packages): return False
        received_updates = received_packages['model']
        received_client_taus = [u._round for u in received_updates]
        for cdelta, ctau in zip(received_updates, received_client_taus):
            self.buffer.append((cdelta, ctau))
        if len(self.buffer) >= int(self.buffer_ratio * self.num_clients):
            # aggregate and clear updates in buffer
            taus_bf = [b[1] for b in self.buffer]
            updates_bf = [b[0] for b in self.buffer]
            weights_bf = [(1 + self.current_round - ctau) ** (-0.5) for ctau in taus_bf]
            model_delta = fmodule._model_average(updates_bf, weights_bf) / len(self.buffer)
            self.model = self.model + self.eta * model_delta
            # clear buffer
            self.buffer = []
            return True
        return False

class Client(BasicClient):
    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        global_model = copy.deepcopy(model)
        self.train(model)
        update = model-global_model
        update._round = model._round
        cpkg = self.pack(update)
        return cpkg