import torch
import argparse
import itertools
import numpy as np
from vqe import *
from utils import *
from random import random
from qiskit import pulse, QuantumCircuit, IBMQ
from qiskit.providers.fake_provider import *
from scipy.optimize import minimize, LinearConstraint
import pdb


def IBMQ_ini(backend_str):
    backend = provider.get_backend(backend_str)
    return backend

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend',  type=str,   default='ibmq_quito',help='name of the backend(not a simulator)')
    parser.add_argument('--optimizer',type=str,   default='COBYLA',    help='name of the non-gradient optimizer')
    parser.add_argument('--policy',   type=str,   default='cxrx',      help='name of the pulse growth policy')
    parser.add_argument('--application',  type=str,   default='chemistry',      help='name of the benchmark application')
    parser.add_argument('--pulse_id', type=int,   default=1,           help='indicate the design space at pulse level.')
    parser.add_argument('--molecule', type=str,   default='H2',        help='name of the molecules')
    parser.add_argument('--n_assets', type=int,   default=2,           help='number of assets')
    parser.add_argument('--tune_freq',type=bool,  default=False,       help='specify if frequencies are tuned')
    parser.add_argument('--n_iter',   type=int,   default=100,         help='number of training iterations')
    parser.add_argument('--n_shot',   type=int,   default=1024,        help='number of shots for measurement')
    parser.add_argument('--n_step',   type=int,   default=1,           help='number of pulse_layers')
    parser.add_argument('--max_jobs', type=int,   default=8,           help='number of max_jobs for multiprocessing')
    parser.add_argument('--rhobeg',   type=float, default=0.1 ,        help='rhobeg for non-gradient optimizer')
    #parser.add_argument('--n_parameters',   type=int, default=7,       help='number of parameters in pulse ansatz')

    args = parser.parse_args()
    backend_str = args.backend
    optimizer   = args.optimizer
    policy      = args.policy
    application = args.application
    pulse_id    = args.pulse_id
    molecule    = args.molecule
    assets      = args.n_assets
    tune_freq   = args.tune_freq
    n_iter      = args.n_iter
    n_shot      = args.n_shot
    n_step      = args.n_step
    rhobeg      = args.rhobeg    
    max_jobs    = args.max_jobs
    #parameters  = args.n_parameters

    seed = 40
    np.random.seed(seed)
    if('Fake' in backend_str):
        backend_sim = ""
        exec('backend_sim='+backend_str+'()')
        backend = aer.PulseSimulator.from_backend(backend_sim)
    else:
        backend = IBMQ_ini(backend_str)

    torch.manual_seed(seed)
    if application == 'chemistry':
        if(molecule=='H2'):
            dist_list = [0.735] #works on H2
        elif(molecule=='HeH'):
            dist_list = [1] #works: on HeH
        elif(molecule=='LiH'):
            dist_list = [1.5] #works on: LiH
        elif(molecule=='H2O'):
            dist_list = [1.5] #works on: LiH
        else:
            print('Molecule not Found')
        pauli_dict, n_qubit = pauli_dict_dist(dist=dist_list[0],molecule=molecule)
        if pulse_id == 1:
            parameters = 5*n_qubit - 3
        elif pulse_id == 2:
            parameters = 2*(2*n_qubit - 1)
        elif pulse_id == 3:
            parameters = 6*n_qubit - 3
        elif pulse_id == 4:
            parameters = 5*n_qubit -2
        elif pulse_id == 5:
            parameters = 9 * n_qubit -7
        elif pulse_id == 6:
            parameters = 8*n_qubit - 6
        else:
            print('Please select correct Pulse_ID.')
        params = np.zeros(parameters)
        LC = gen_LC_finance(n_qubit, parameters)
        #pauli_dict = finance_dict(2,seed,0.5)
        finance_res = minimize(vqe,params,args=(pauli_dict,n_qubit,backend,max_jobs,n_shot,pulse_id),method=optimizer,
                            constraints=LC,options={'rhobeg':rhobeg,'maxiter':n_iter,'disp':True})
        print(pauli_dict)
        print('The optimized loss func value: {}'.format(finance_res.fun))
    elif application == 'finance':
        pauli_dict = finance_dict(assets,seed,0.5)
        n_qubit = assets
        if pulse_id == 1:
            parameters = 5*n_qubit - 3
        elif pulse_id == 2:
            parameters = 2*(2*n_qubit - 1)
        elif pulse_id == 3:
            parameters = 6*n_qubit - 3
        elif pulse_id == 4:
            parameters = 5*n_qubit -2
        elif pulse_id == 5:
            parameters = 9 * n_qubit -7
        elif pulse_id == 6:
            parameters = 8*n_qubit - 6
        else:
            print('Please select correct Pulse_ID.')
        params = np.zeros(parameters)
        LC = gen_LC_finance(assets, parameters)
        finance_res = minimize(finance,params,args=(pauli_dict,assets,backend,max_jobs,n_shot,pulse_id),method=optimizer,
                            constraints=LC,options={'rhobeg':rhobeg,'maxiter':n_iter,'disp':True})
        print(pauli_dict)
        print('The optimized loss func value: {}'.format(finance_res.fun))
    else:
        print('Application not found.')
    

