"""This is a non-official implementation of 'Federated Learning with Buffered Asynchronous Aggregation' (http://arxiv.org/abs/2106.06639). """
from flgo.algorithm.fedasync import Server as AsyncServer
from flgo.algorithm.fedbase import BasicClient
import flgo.utils.fmodule as fmodule
import copy

class Server(AsyncServer):
    def initialize(self):
        self.init_algo_para({'buffer_ratio': 0.2, 'eta': 1.0, 'period': 1,})
        self.tolerance_for_latency = 1000
        self.buffer = []
        self.hs = [0 for _ in self.clients]
        self.m = 0
        self.delta = self.model.zeros_like()

    def pack(self, client_id, mtype=0, *args, **kwargs):
        return {
            'model': copy.deepcopy(self.model),
            'round': self.current_round,
        }

    def iterate(self):
        # Scheduler periodically triggers the idle clients to locally train the model
        self.selected_clients = self.sample() if (self.gv.clock.current_time % self.period) == 0 or self.gv.clock.current_time == 1 else []
        if len(self.selected_clients) > 0:
            self.gv.logger.info('Select clients {} at time {}'.format(self.selected_clients, self.gv.clock.current_time))
        # filter clients who have already uploaded their latest updates relative to the current model
        self.selected_clients = [cid for cid in self.selected_clients if cid not in [bi[-1] for bi in self.buffer if bi[0]==self.current_round]]
        res = self.communicate(self.selected_clients, asynchronous=True)
        received_updates = res['update']
        received_client_taus = res['round']
        received_client_ids = res['__cid']
        # if reveive client update
        if len(received_updates) > 0:
            self.gv.logger.info('Receive new models from clients {} at time {}'.format(received_client_ids, self.gv.clock.current_time))
            for cdelta, ctau, cid in zip(received_updates, received_client_taus, received_client_ids):
                self.m += 1
                self.delta += (cdelta - (self.hs[cid].to(self.device) if self.hs[cid]!=0 else self.hs[cid]))
                self.hs[cid] = cdelta.to('cpu')
                self.buffer.append((ctau, cid))
            if self.m>=int(self.buffer_ratio * self.num_clients):
                # aggregate and clear updates in buffer
                ht = fmodule._model_average([h for h in self.hs if h!=0]).to(self.device)
                vt = self.delta + ht
                self.model = self.model + self.eta * vt
                self.m = 0
                self.delta = self.model.zeros_like()
                self.buffer = []
                return True
        return False

class Client(BasicClient):
    def unpack(self, received_pkg):
        round = received_pkg['round']
        model = received_pkg['model']
        return model, round

    def pack(self, model, round):
        return {'update':model, 'round':round}

    def reply(self, svr_pkg):
        model,round  = self.unpack(svr_pkg)
        global_model = copy.deepcopy(model)
        self.train(model)
        cpkg = self.pack(model-global_model, round)
        return cpkg