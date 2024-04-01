from flgo.simulator.base import BasicSimulator

class StaticUniSimulator(BasicSimulator):
    def initialize(self):
        # Uniform(600s, 10800s) is equal to Uniform(10min, 3h)
        self.client_time_response = {cid: self.random_module.randint(600, 10800) for cid in self.clients}
        self.set_variable(list(self.clients.keys()), 'latency', list(self.client_time_response.values()))

    def update_client_responsiveness(self, client_ids):
        latency = [self.client_time_response[cid] for cid in client_ids]
        self.set_variable(client_ids, 'latency', latency)


class DynamicUniSimulator(BasicSimulator):
    def initialize(self):
        # Uniform(600s, 10800s) is equal to Uniform(10min, 3h)
        client_time_response = {cid: self.random_module.randint(600, 10800) for cid in self.clients}
        self.set_variable(list(self.clients.keys()), 'latency', list(client_time_response.values()))

    def update_client_responsiveness(self, client_ids):
        client_time_response = {cid: self.random_module.randint(600, 10800) for cid in client_ids}
        self.set_variable(client_ids, 'latency', list(client_time_response.values()))