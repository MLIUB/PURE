import torch, pickle, io
import torch.nn as nn

import ctrl.dataset as ds
from utils import BNN
from .dynamics import NODE, PETS, DeepPILCO
from .policy import Policy

DEFAULT_PAR_MAP = {'nl_f': 3, 'nn_f': 200, 'act_f': 'elu', 'dropout_f': 0.05, 'n_ens': 10, 'learn_sigma': False,
                   'nl_g': 2, 'nn_g': 200, 'act_g': 'relu',
                   'nl_V': 2, 'nn_V': 200, 'act_V': 'tanh'}


# The CTRL class models control dynamics and policies for interacting with an environment.
# It integrates dynamics models (e.g., NODE, PETS, DeepPILCO) and a policy to enable forward simulations
# and decision-making in a reinforcement learning or control setting.
class CTRL(nn.Module):
    def __init__(self, env, dynamics, **kwargs):
        super().__init__()
        for PAR in DEFAULT_PAR_MAP:
            kwargs[PAR] = kwargs.get(PAR)  # returns value in DEFAULT_PARS or None
        self.kwargs = kwargs
        self.env = env
        self.n_ens = self.kwargs['n_ens']
        self.learn_sigma = self.kwargs['learn_sigma']
        self.dynamics = dynamics
        self.set_solver('dopri5')
        self._g = Policy(self.env, nl=kwargs['nl_g'], nn=kwargs['nn_g'], act=kwargs['act_g'])
        self.make_dynamics_model(nl_f=kwargs['nl_f'], nn_f=kwargs['nn_f'], act_f=kwargs['act_f'],
                                 dropout_f=kwargs['dropout_f'])
        self.V = BNN(self.env.n, 1, n_hid_layers=kwargs['nl_V'], act=kwargs['act_V'], n_hidden=kwargs['nn_V'],
                     bnn=False)
        self.reset_parameters()

    @property
    def device(self):
        return next(self._g.parameters()).device

    @property
    def dtype(self):
        return torch.float32

    @property
    def sn(self):
        return self.logsn.exp()

    @property
    def dynamics_parameters(self):
        if self.learn_sigma:
            return [self.logsn] + list(self._f.parameters())
        else:
            return list(self._f.parameters())

    @property
    def is_cont(self):
        return 'ode' in self.dynamics

    @property
    def name(self):
        return self.env.name + '-' + self.dynamics

    def make_dynamics_model(self, nl_f=2, nn_f=200, act_f='elu', dropout_f=0.05):
        if self.learn_sigma:
            self.logsn = torch.nn.Parameter(-torch.ones(self.env.n + self.env.m) * 3.0, requires_grad=True)
        else:
            self.register_buffer('logsn', -torch.ones(self.env.n + self.env.m) * 3.0)
        if self.is_cont:
            if dropout_f > 0.0:
                print('Dropout is set to 0 since NODE is running')
            self._f = NODE(self.env, self.dynamics, self.n_ens, nl=nl_f, nn=nn_f, act=act_f)
        elif self.dynamics == 'pets':
            if dropout_f > 0.0:
                print('Dropout is set to 0 since PETS is running')
            self._f = PETS(self.env, 'epnn', self.n_ens, nl=nl_f, nn=nn_f, act=act_f)
        elif self.dynamics == 'deep_pilco':
            self._f = DeepPILCO(self.env, 'dbnn', self.n_ens, nl=nl_f, nn=nn_f, act=act_f, dropout=dropout_f)

    def set_solver(self, solver):
        assert solver in ['euler', 'midpoint', 'rk4', 'dopri5', 'rk23', 'rk45']
        self.solver = {}
        self.solver['method'] = solver
        self.solver['step_size'] = self.env.dt / 10  # in case fixed step solvers are used
        self.solver['rtol'] = 1e-3
        self.solver['atol'] = 1e-6

    def draw_f(self, L=1, noise_vec=None, true_rhs=False):
        if self._f.ens_method:
            return self._f._f.draw_f()
        if noise_vec is None:
            noise_vec = self.draw_noise(L)
        return self._f._f.draw_f(L, noise_vec)  # TODO - check if mean is a parameter

    def get_L(self, L=1):
        ''' returns the number of samples from the function
            this is needed as ensembles have a fixed number of possible fnc draws.
        '''
        return self.n_ens if self._f.ens_method else L

    def draw_noise(self, L=1, true_rhs=False):
        return self._f._f.draw_noise(L=L)

    def forward_simulate(self, H_ts, s0, g, f=None, L=10, tau=None, compute_rew=False):
        ''' Performs forward simulation for L different vector fields.
            This method propagates the system state using the dynamics model and policy function.

            Key features:
            - If H_ts is a float, it creates a uniform time grid for simulation.
            - If H_ts is a vector, it uses custom time points for simulation.

            Inputs:
                H_ts - either a float denoting the total simulation time or a tensor specifying time points.
                s0   - [N, n] Initial states of the system.
                g    - Policy function that maps states to actions.
                f    - (Optional) Dynamics function; if None, the method draws one.
                L    - Number of vector fields to simulate (default is 10).
                tau  - (Optional) Auxiliary parameter for the simulation.
                compute_rew - Whether to compute rewards during simulation (default is False).

            Outputs:
                st - [L, N, T, n] Simulated system states over time.
                rt - [L, N, T, n] Rewards associated with the states (if compute_rew=True).
                at - Dictionary mapping times to actions: {[T]: [L, N, m]}.
                t  - [N, T] Time points used during simulation.
        '''
        L = self.get_L(L)
        if f is None:
            f = self.draw_f(L, None)
        # integration time points is a uniform grid
        if isinstance(H_ts, float) or isinstance(H_ts, int):
            return self._f.forward_simulate(solver=self.solver, H=H_ts, s0=s0, f=f, g=g, L=L, \
                                            tau=tau, compute_rew=compute_rew)
        else:
            return self._f.forward_simulate_nonuniform_ts(solver=self.solver, ts=H_ts, s0=s0, f=f, g=g, L=L, \
                                                          tau=tau, compute_rew=compute_rew)

    def reset_parameters(self, w=0.1):
        ''' Initializes or resets model parameters to ensure a fresh start for training.
            This method reinitializes the dynamics model, policy, and value network parameters
            using uniform distributions or other specified initialization strategies.
        '''
        self._f.reset_parameters(w)
        self._g.reset_parameters(w)
        self.V.reset_parameters(w)
        nn.init.uniform_(self.logsn, -1.0, -1.0)

    @staticmethod
    def load(env, fname, verbose=True):
        ''' Loads a previously saved control model and its associated data.
            This method restores the model's state dictionary, configuration, and dynamics type
            from a saved file. If additional data, such as a dataset, is included in the file,
            it will also be loaded and returned.

            Inputs:
                env    - The environment object to which the model will be linked.
                fname  - Filename of the saved model (with or without .pkl extension).
                verbose - If True, prints loading status and additional details.

            Outputs:
                ctrl - The restored CTRL object.
                D    - The associated dataset, if any, otherwise None.
        '''
        fname = fname[:-4] if fname.endswith('.pkl') else fname
        if verbose:
            print('{:s} is loading.'.format(fname))

        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)

        f = open(fname + '.pkl', 'rb')
        stuff = CPU_Unpickler(f).load()
        dynamics, kwargs, state_dict = stuff['dynamics'], stuff['kwargs'], stuff['state_dict']
        ctrl = CTRL(env, dynamics, **kwargs).to(env.device)
        ctrl.load_state_dict(state_dict)
        ctrl.eval()
        if 'D_D' in list(stuff.keys()):
            D, ts = stuff['D_D'], stuff['D_ts']
            D = ds.Dataset(env, D, ts).to(env.device)
            if verbose:
                print(D.shape)
                print(stuff['dynamics'])
        else:
            D = None
        return ctrl.to(env.device), D

    def save(self, D=None, fname=None, verbose=False):
        ''' Serializes the current state of the model and optionally saves associated datasets.
            This method creates a dictionary containing the model's state dictionary, configuration,
            and dynamics. If a dataset (D) is provided, it is also saved in the dictionary.

            Inputs:
                D     - Optional dataset to be saved with the model.
                fname - Filename to save the model (default uses the model's name).
                verbose - If True, prints the name of the saved file.
        '''
        if fname is None:
            fname = self.name
        if verbose:
            print('model save name is {:s}'.format(fname))
        state_dict = self.state_dict()
        save_dict = {}
        save_dict['state_dict'] = state_dict
        save_dict['kwargs'] = self.kwargs
        save_dict['dynamics'] = self.dynamics
        if D is not None:
            save_dict['D_D'] = D.D
            save_dict['D_ts'] = D.ts
        pickle.dump(save_dict, open(fname + '.pkl', 'wb'))

    def __repr__(self):
        text = 'Env solver: ' + str(self.env.solver) + '\n'
        text += 'Model solver: ' + str(self.solver) + '\n'
        text += self._f.__repr__() + '\n'
        text += self._g.__repr__() + '\n'
        text += self.V.__repr__() + '\n'
        return text