"""
This is a non-official implementation of 'GradMA: A Gradient-Memory-based Accelerated Federated Learning with Alleviated Catastrophic Forgetting'
(http://arxiv.org/abs/2302.14307).
"""
import copy
from collections import OrderedDict

import cvxopt
import numpy as np
import torch
from flgo.algorithm.fedbase import BasicClient
from flgo.algorithm.fedbase import BasicServer
import flgo.utils.fmodule as fmodule

cvxopt.solvers.options['show_progress'] = False

def quadprog(P, q, G=None, h=None, A=None, b=None):
    """
    Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
    Output: Numpy array of the solution
    """
    P = cvxopt.matrix(P.tolist())
    q = cvxopt.matrix(q.tolist(), tc='d').T
    G = cvxopt.matrix(G.tolist()).T if G is not None else None
    h = cvxopt.matrix(h.tolist()).T if h is not None else None
    A = cvxopt.matrix(A.tolist()).T if A is not None else None
    b = cvxopt.matrix(b.tolist(), tc='d') if b is not None else None
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    return np.array(sol['x'])

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'beta1': 0.5, 'beta2':0.5, 'buffer_size':20, 'eta':1.0})
        self.v = self.model.zeros_like()
        self.m = self.model.zeros_like()
        self.counter = [0 for _ in self.clients]
        self.memory_state = [None for _ in self.clients]
        self.buf = [False for _ in self.clients]
        self.buffer_size = min(max(self.clients_per_round, self.buffer_size), self.num_clients)

    def iterate(self):
        self.selected_clients = self.sample()
        new_buf = self.mem_red(self.selected_clients)
        deltas = self.communicate(self.selected_clients)['delta']
        print([cid for cid in range(self.num_clients) if self.buf[cid]])
        self.model = self.aggregate(deltas, new_buf)
        return

    def aggregate(self, deltas, new_buf):
        delta_dict = {cid: cdelta for cid, cdelta in zip(self.selected_clients, deltas)}
        # update memory state: D
        for cid in range(self.num_clients):
            if self.buf[cid]:
                if cid in self.selected_clients:
                    if cid in new_buf:
                        self.memory_state[cid] = delta_dict[cid]
                    else:
                        self.memory_state[cid] = self.beta2 * self.memory_state[cid] + delta_dict[cid]
                else:
                    self.memory_state[cid] = self.beta2 * self.memory_state[cid]
        # compute delta
        new_delta = fmodule._model_average(deltas)
        new_m = self.beta1 * self.m + new_delta
        self.m = self.qp_global(new_m)
        # udpate model
        return self.model - self.eta*self.m

    def qp_global(self, m):
        M = torch.stack([fmodule._model_to_tensor(mi).cpu() for mi in self.memory_state if mi is not None], dim=0).numpy().T
        p = fmodule._model_to_tensor(m).cpu().numpy()
        pM = p@M
        MM = M.T@M
        G = -np.eye(len(pM))
        h = np.zeros(len(pM))
        z = quadprog(MM, np.expand_dims(pM, -1), G, np.expand_dims(h, -1))
        m = fmodule._model_from_tensor(torch.from_numpy((M @ z).squeeze(-1) + p), self.model.__class__)
        return m.to(self.device)

    def mem_red(self, selected_clients):
        """
        m: self.m
        S: self.selected_clients
        c: self.counter
        D: self.memory_state
        buf: self.buf
        """
        new_buf = []
        for cid in selected_clients:
            if self.buf[cid]:
                self.counter[cid] += 1
            else:
                if len([t for t in self.buf if t])>self.buffer_size:
                    old_buf = [j for j in range(self.num_clients) if self.buf[j] and j not in selected_clients]
                    old_buf_counter = [self.counter[j] for j in old_buf]
                    discard_j = old_buf[np.argmin(old_buf_counter)]
                    self.counter[discard_j] = 0
                    self.buf[discard_j] = False
                    self.memory_state[discard_j] = None
                self.counter[cid] += 1
                self.buf[cid] = True
                new_buf.append(cid)
        return new_buf



class Client(BasicClient):
    def pack(self, model):
        return {'delta':model}

    def get_grad(self, model):
        with torch.no_grad():
            res = OrderedDict()
            for name, p in model.named_parameters():
                _g = p.grad.detach()
                if 'bias' not in name:
                    _g += (self.weight_decay*p).detach()
                res[name] = _g
        return res

    @fmodule.with_multi_gpus
    def train(self, model):
        model.train()
        tmp_model = copy.deepcopy(model)
        global_dict = copy.deepcopy(model.state_dict())
        old_dict = copy.deepcopy(model.state_dict())
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, momentum=self.momentum)
        for iter in range(self.num_steps):
            current_dict = copy.deepcopy(model.state_dict())
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            # current grad
            crt_grad = self.get_grad(model)
            # old grad
            tmp_model.load_state_dict(old_dict)
            old_loss = self.calculator.compute_loss(tmp_model, batch_data)['loss']
            old_loss.backward()
            old_grad = self.get_grad(tmp_model)
            tmp_model.zero_grad()
            # global grad
            tmp_model.load_state_dict(global_dict)
            global_loss = self.calculator.compute_loss(tmp_model, batch_data)['loss']
            global_loss.backward()
            global_grad = self.get_grad(tmp_model)
            tmp_model.zero_grad()
            # local diff
            with torch.no_grad():
                diff = OrderedDict()
                for (name, p1), p2 in zip(model.named_parameters(), tmp_model.parameters()):
                    diff[name] = (p1-p2).detach()
                G = [crt_grad, old_grad, global_grad, diff]
                self.qp_local(G, model)
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
            old_dict = current_dict
        tmp_model.load_state_dict(global_dict)
        return tmp_model - model

    def qp_local(self, G, model):
        G = [torch.cat([v.reshape(-1) for v in g.values()], dim=0).cpu() for g in G]
        p = G[0].numpy()
        M = torch.stack(G[1:], dim=0).numpy().T
        pM = p@M
        MM = M.T@M
        G = -np.eye(len(pM))
        h = np.zeros(len(pM))
        z = quadprog(MM, np.expand_dims(pM, -1), G, np.expand_dims(h, -1))
        m = torch.from_numpy((M @ z)).squeeze(-1).float()
        num_paras = 0
        for name, pi in model.named_parameters():
            new_grad = (m[num_paras:num_paras+pi.numel()]).reshape(pi.shape)
            pi.grad = pi.grad + new_grad.to(pi.device)

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        delta = self.train(model)
        cpkg = self.pack(delta)
        return cpkg