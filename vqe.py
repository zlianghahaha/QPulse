import qiskit
import numpy as np
import pathos.multiprocessing as multiprocessing

from utils import *
from itertools import repeat
from qiskit.test.mock import *
from qiskit import pulse, IBMQ
from qiskit.compiler import assemble

def vqe_napa(tune_param, fixed_param, partial_list, pauli_dict, ansatz_seed, tune_freq, n_qubit, backend, max_jobs, n_shot):
    merged_param = merge(partial_param=tune_param, fixed_param=fixed_param, partial_list=partial_list)
    print("All parameters: ", merged_param) 
    p = multiprocessing.Pool(max_jobs)
    prepulse = ansatz_init(merged_param=merged_param,backend=backend,ansatz_seed=ansatz_seed, option=0, n_qubit=n_qubit)

    keys = [key for key in pauli_dict]
    values = [pauli_dict[key] for key in pauli_dict]

    expect_values = p.starmap(one_in_all,zip(repeat(prepulse),repeat(n_qubit),repeat(n_shot),repeat(backend),keys,values))

    p.close()
    p.join()
    print("E for cur_iter: ",sum(expect_values))
    return sum(expect_values)


#def generate_firstlayer(backend, num_qubits, amp, angle):
#inance,params,args=(pauli_dict,n_qubit,backend,max_jobs,n_shot,)
def finance(params,pauli_dict,n_qubit,backend,max_jobs,n_shot, pulse_id):
    print("params in def finance in vqe.py: ", params)
    # assert(len(params)%2==0)
    width_len = int(len(params)-1*(n_qubit-1))
    split_ind = int(width_len/2)
    amp = np.array(params[:split_ind])
    amp_fixed = np.array(params[:int(len(params)/2)-1*(n_qubit -1)])
    angle_fixed = np.array(params[int(len(params)/2)-1*(n_qubit-1):width_len])*np.pi*2
    angle = np.array(params[split_ind:width_len])*np.pi*2
    width_1 = (np.array(params[width_len:]))
    num_items = (1024 - 256) // 16 + 1
    width_norm = (width_1 - 256) / (1024 - 256)
    width_norm = np.clip(width_norm, 0, 1)
    width = (np.round(width_norm * (num_items - 1)) * 16 + 256).astype(int)
    amp_fixed = amp_fixed.tolist()
    angle_fixed = angle_fixed.tolist()
    amp = amp.tolist()
    angle = angle.tolist()
    width = width.tolist()
    p = multiprocessing.Pool(max_jobs)
    keys = [key for key in pauli_dict]
    values = [pauli_dict[key] for key in pauli_dict]
    if n_qubit == 4:
        if pulse_id ==1:
            prepulse = HE_pulse(backend,amp,angle,width)
        elif pulse_id ==2:
            prepulse = HE_pulsefixedamp(backend,amp_fixed,angle_fixed,width)
        elif pulse_id ==3:
            prepulse = Decaylayer_pulse(backend,amp,angle,width)
        elif pulse_id == 4:
            prepulse = Decaylayer_pulsefixedamp(backend,amp_fixed,angle_fixed,width)
        elif pulse_id == 5:
            prepulse = block_pulse(backend,amp,angle,width)
        elif pulse_id == 6:
            prepulse = block_pulsefixedamp(backend,amp_fixed,angle_fixed,width)
        else:
            print('Pulse ID is wrong. Please type 1 - 6.')
    elif n_qubit == 2:
        if pulse_id ==1:
            prepulse = HE_pulse2q(backend,amp,angle,width)
        elif pulse_id ==2:
            prepulse = HE_pulse2qfixedamp(backend,amp_fixed,angle_fixed,width)
        elif pulse_id ==3:
            prepulse = Decaylayer_pulse2q(backend,amp,angle,width)
        elif pulse_id == 4:
            prepulse = Decaylayer_pulse2qfixedamp(backend,amp_fixed,angle_fixed,width)
        elif pulse_id == 5:
            prepulse = block_dressedpulse2q(backend,amp,angle,width)
        elif pulse_id == 6:
            prepulse = block_dressedpulse2qfixedamp(backend,amp_fixed,angle_fixed,width)
        else:
            print('Pulse ID is wrong. Please type 1 - 6.')
    else: 
        print('Number of qubits is out of range, current we support 2 qubits and 4 qubits applications, more qubits will be adds on.')
    expect_values = p.starmap(finance_one,zip(repeat(prepulse),repeat(n_qubit),repeat(n_shot),repeat(backend),keys,values))

    p.close()
    p.join()
    print("E for cur_iter: ",sum(expect_values))
    return sum(expect_values)

def vqe(params,pauli_dict,n_qubit,backend,max_jobs,n_shot,pulse_id):
    print("params in def chemistry in vqe.py: ", params)
    # assert(len(params)%2==0)
    width_len = int(len(params)-1*(n_qubit-1))
    split_ind = int(width_len/2)
    amp = np.array(params[:split_ind])
    amp_fixed = np.array(params[:int(len(params)/2)-1*(n_qubit-1)])
    angle_fixed = np.array(params[int(len(params)/2)-1*(n_qubit-1):width_len])*np.pi*2
    angle = np.array(params[split_ind:width_len])*np.pi*2
    width_1 = (np.array(params[width_len:]))
    num_items = (1024 - 256) // 16 + 1
    width_norm = (width_1 - 256) / (1024 - 256)
    width_norm = np.clip(width_norm, 0, 1)
    width = (np.round(width_norm * (num_items - 1)) * 16 + 256).astype(int)
    amp_fixed = amp_fixed.tolist()
    angle_fixed = angle_fixed.tolist()
    amp = amp.tolist()
    angle = angle.tolist()
    width = width.tolist()
    p = multiprocessing.Pool(max_jobs)
    keys = [key for key in pauli_dict]
    values = [pauli_dict[key] for key in pauli_dict]
    if n_qubit == 4:
        if pulse_id ==1:
            prepulse = HE_pulse(backend,amp,angle,width)
        elif pulse_id ==2:
            prepulse = HE_pulsefixedamp(backend,amp_fixed,angle_fixed,width)
        elif pulse_id ==3:
            prepulse = Decaylayer_pulse(backend,amp,angle,width)
        elif pulse_id == 4:
            prepulse = Decaylayer_pulsefixedamp(backend,amp_fixed,angle_fixed,width)
        elif pulse_id == 5:
            prepulse = block_pulse(backend,amp,angle,width)
        elif pulse_id == 6:
            prepulse = block_pulsefixedamp(backend,amp_fixed,angle_fixed,width)
        else:
            print('Pulse ID is wrong. Please type 1 - 6.')
    elif n_qubit == 3:
        if pulse_id == 1:
            prepulse = HE_pulse_3q(backend,amp,angle,width)
        else:
            print('Pulse ID is wrong. Please type 1 - 6.')            
    elif n_qubit == 2:
        if pulse_id ==1:
            prepulse = HE_pulse2q(backend,amp,angle,width)
        elif pulse_id ==2:
            prepulse = HE_pulse2qfixedamp(backend,amp_fixed,angle_fixed,width)
        elif pulse_id ==3:
            prepulse = Decaylayer_pulse2q(backend,amp,angle,width)
        elif pulse_id == 4:
            prepulse = Decaylayer_pulse2qfixedamp(backend,amp_fixed,angle_fixed,width)
        elif pulse_id == 5:
            prepulse = block_dressedpulse2q(backend,amp,angle,width)
        elif pulse_id == 6:
            prepulse = block_dressedpulse2qfixedamp(backend,amp_fixed,angle_fixed,width)
        else:
            print('Pulse ID is wrong. Please type 1 - 6.')
    else: 
        print('Number of qubits is out of range, current we support 2 qubits and 4 qubits applications, more qubits will be adds on.')
    expect_values = p.starmap(vqe_one,zip(repeat(prepulse),repeat(n_qubit),repeat(n_shot),repeat(backend),keys,values))

    p.close()
    p.join()
    print("E for cur_iter: ",sum(expect_values))
    return sum(expect_values)
