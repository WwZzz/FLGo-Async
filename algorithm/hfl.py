"""This is a non-official implementation of 'Stragglers Are Not Disaster: A Hybrid Federated Learning Algorithm with Delayed Gradients' (http://arxiv.org/abs/2102.06329). """
from flgo.algorithm.fedasync import Server as AsyncServer
from flgo.algorithm.fedbase import BasicClient
import flgo.utils.fmodule as fmodule
import copy

class Server(AsyncServer):
    def initialize(self):
        self.init_algo_para({'buffer_ratio': 0.2, 'eta': 1.0, 'period': 1,})
        self.tolerance_for_latency = 1000
        self.buffer = []

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
        res = self.communicate(self.selected_clients, asynchronous=True)
        received_updates = res['update']
        received_client_taus = res['round']
        received_client_ids = res['__cid']
        # if reveive client update
        if len(received_updates) > 0:
            self.gv.logger.info('Receive new models from clients {} at time {}'.format(received_client_ids, self.gv.clock.current_time))
            for cdelta, ctau in zip(received_updates, received_client_taus):
                self.buffer.append((cdelta, ctau))
            if len(self.buffer)>=int(self.buffer_ratio*self.num_clients):
                # aggregate and clear updates in buffer
                taus_bf = [b[1] for b in self.buffer]
                updates_bf = [b[0] for b in self.buffer]
                weights_bf = [(1+self.current_round-ctau)**(-0.5) for ctau in taus_bf]
                model_delta = fmodule._model_average(updates_bf, weights_bf)/len(self.buffer)
                self.model = self.model + self.eta * model_delta
                # clear buffer
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