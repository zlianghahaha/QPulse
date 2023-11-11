import copy
import sched
import qiskit
import itertools
import numpy as np
import pathos.multiprocessing as multiprocessing

from itertools import repeat
from qiskit.providers import aer
from qiskit.providers.fake_provider import *
from qiskit.circuit import Gate
from qiskit.compiler import assemble
from qiskit import pulse, QuantumCircuit, IBMQ
from qiskit.pulse.instructions import Instruction
from qiskit_nature.drivers import UnitsType, Molecule
from scipy.optimize import minimize, LinearConstraint
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.properties.second_quantization.electronic import ParticleNumber
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from typing import List, Tuple, Iterable, Union, Dict, Callable, Set, Optional, Any
from qiskit.pulse import Schedule, GaussianSquare, Drag, Delay, Play, ControlChannel, DriveChannel
from qiskit_nature.mappers.second_quantization import ParityMapper,JordanWignerMapper
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer
from qiskit_nature.drivers.second_quantization import ElectronicStructureDriverType, ElectronicStructureMoleculeDriver


# below for q_finance
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.result import QuasiDistribution
from qiskit_aer.primitives import Sampler
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np
import matplotlib.pyplot as plt
import datetime
from qiskit_optimization.converters import QuadraticProgramToQubo
import pdb

def cr_pulsegate(circ, qubits):
    sched = Schedule(name = 'cr')
    sigma = 64
    square_width = 112
    cr_dur = 368
    amp = (0.11224059989953047+0.0016941472743981524j)
    cr_p = GaussianSquare(cr_dur, amp, sigma, square_width)
    cr_m = GaussianSquare(cr_dur, -amp, sigma, square_width)

    sched += Play(cr_p, ControlChannel(qubits[1]))
    sched += Delay(0, ControlChannel(qubits[1]))
    sched += Play(cr_m, ControlChannel(qubits[1]))

    custom_gate = Gate('cr', 2, [np.pi/4])

    circ.append(custom_gate, qubits)

    circ.add_calibration('cr', qubits, sched, [np.pi/4])
    return circ


def merge(partial_list, fixed_param, partial_param):
    merged_param = []
    j = 0
    k = 0
    for i in range(len(partial_list)):
        if partial_list[i] == 1:
            merged_param.append(partial_param[j])
            j += 1
        elif partial_list[i] == 0:
            merged_param.append(fixed_param[k])
            k += 1
    
    return merged_param

def measurement_pauli(prepulse, pauli_string, backend, n_qubit):
    with pulse.build(backend) as pulse_measure:
        pulse.call(copy.deepcopy(prepulse))
        for ind,pauli in enumerate(pauli_string):
            if(pauli=='X'):
                pulse.u2(0, np.pi, ind)
            if(pauli=='Y'):
                pulse.u2(0, np.pi/2, ind)
        for qubit in range(n_qubit):
            pulse.barrier(qubit)
        pulse.measure(range(n_qubit))
    return pulse_measure

def n_one(bitstring, key):
    results = 0
    for ind,b in enumerate(reversed(bitstring)):
        if((b=='1')&(key[ind]!='I')):
            results+=1
    return results

def expectation_value(counts,shots,key):
    results = 0
    for bitstring in counts:
        if(n_one(bitstring, key)%2==1):
            results -= counts[bitstring]/shots
        else:
            results += counts[bitstring]/shots
    return results

def run_pulse_sim(meas_pulse, key, backend, n_shot):
    # backend_run = aer.PulseSimulator.from_backend(backend)
    backend_run = backend
    pulse_assemble = assemble(meas_pulse, backend = backend_run, shots=n_shot, meas_level=2, meas_return='single')
    results = backend_run.run(pulse_assemble).result()
    counts = results.get_counts()
    expectation = expectation_value(counts,n_shot,key)
    return expectation

def finance_one(prepulse,n_qubit,n_shot,backend,key,value):
    all_Is = True
    for key_ele in key:
        if(key_ele!='I'):
            all_Is = False
    if(all_Is):
        return value
    
    meas_pulse = measurement_pauli(prepulse=prepulse, pauli_string=key, backend=backend, n_qubit=n_qubit)
    return value*run_pulse_sim(meas_pulse, key, backend, n_shot)
def vqe_one(prepulse,n_qubit,n_shot,backend,key,value):
    all_Is = True
    for key_ele in key:
        if(key_ele!='I'):
            all_Is = False
    if(all_Is):
        return value
    
    meas_pulse = measurement_pauli(prepulse=prepulse, pauli_string=key, backend=backend, n_qubit=n_qubit)
    return value*run_pulse_sim(meas_pulse, key, backend, n_shot)
def one_in_all(prepulse, n_qubit, n_shot, backend, key, value):
    all_Is = True
    for key_ele in key:
        if(key_ele!='I'):
            all_Is = False
    if(all_Is):
        return value
    meas_pulse = measurement_pauli(prepulse=copy.deepcopy(prepulse), pauli_string=key, backend=backend, n_qubit=n_qubit)
    return value*run_pulse_sim(meas_pulse, key, backend, n_shot)
def finance_dict(num_assets, seed, risk_factor):
    stocks = [("TICKER%s" % i) for i in range(num_assets)]
    data = RandomDataProvider(
        tickers=stocks,
        start=datetime.datetime(2016, 1, 1),
        end=datetime.datetime(2016, 1, 30),
        seed=seed,)
    
    data.run()
    mu = data.get_period_return_mean_vector()
    sigma = data.get_period_return_covariance_matrix()
    q = risk_factor  # set risk factor
    budget = num_assets // 2  # set budget

    portfolio = PortfolioOptimization(expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget)
    qp = portfolio.to_quadratic_program()
    conv = QuadraticProgramToQubo()
    problem2 = conv.convert(qp)
    qubit_op = problem2.to_ising()
    pauli_dict = {}
    for pauli_op in qubit_op[0].to_pauli_op().oplist:
        pauli_dict[pauli_op.primitive.to_label()] = pauli_op.coeff

    return pauli_dict
def cir2pul(circuit, backend):
    with pulse.build(backend) as pulse_prog:
        pulse.call(circuit)
    return pulse_prog

def gen_LC_finance(n_qubit,parameters):
    lb = np.zeros(parameters)
    ub = np.ones(parameters)
    LC = (LinearConstraint(np.eye(parameters),lb,ub,keep_feasible=False))
    print("lb in gen_LC_finance: ",lb)
    print("ub in gen_LC_finance: ",ub)
    print(LC)
    return LC

def gen_LC(pulse_prog):
    param_list = extract(pulse_prog)
    dim_design = len(param_list)
    lb = np.zeros(dim_design)
    ub = np.ones(dim_design)
    LC = (LinearConstraint(np.eye(dim_design), lb, ub, keep_feasible=False))
    partial_param = np.zeros(param_list.shape)
    return LC,partial_param

def get_from(d: dict, key: str):
    value = 0
    if key in d:
        value = d[key]
    return value

def ansatz_init(merged_param, backend, n_qubit, ansatz_seed=None, option=0):
    GHz = 1*10e9
    if(option==0):
        # assert (len(merged_param)%2)==0
        # modified_list = merged_param[::2]*np.cos(np.array(merged_param[1::2])*2*np.pi) + merged_param[::2]*np.sin(np.array(merged_param[1::2])*2*np.pi)*1j
        with pulse.build(backend) as pulse_ansatz:
            circuit = ansatz_seed
            # for i in range(len(qubits)):
            #     pulse.shift_frequency(freq_shift * GHz,DriveChannel(qubits[i]))
            pulse.call(circuit)
        bak_circ = copy.deepcopy(pulse_ansatz)
        sched = Schedule()
        for inst, amp in zip(bak_circ.blocks[0].operands[0].filter(is_parametric_pulse).instructions, merged_param):
            inst[1].pulse._params['amp'] = amp +0j
        for i in bak_circ.blocks[0].operands[0].instructions:
            if(is_parametric_pulse(i)):
                if(i[1].pulse.amp.real>0.001):
                    sched+=copy.deepcopy(i[1])
            else:
                sched+=copy.deepcopy(i[1])
        return sched



def pauli_dict_dist(dist,molecule):
    if(molecule=='H2'):
        molecule = Molecule(geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, dist]]], charge=0, multiplicity=1)
        nmo=2
    if(molecule=='LiH'):
        molecule = Molecule(geometry=[["Li", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, dist]]], charge=0, multiplicity=1)
        nmo=3
    if(molecule=='H2O'):
        molecule = Molecule(geometry=[["O", [0.0,0.0,0.0], "H", [0.757, 0.586, 0.0]], ["H", [-0.757, 0.586, 0.0]]], charge=0, multiplicity=3)
        nmo=7
    if(molecule=='HeH'):
        molecule = Molecule(geometry=[["He", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, dist]]], charge=1, multiplicity=1)   
        nmo=2
    pauli_dict = {}
    driver = ElectronicStructureMoleculeDriver(molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF)
    properties = driver.run()
    particle_number = properties.get_property(ParticleNumber)
    active_space_trafo = ActiveSpaceTransformer(num_electrons=particle_number.num_particles, num_molecular_orbitals=nmo)
    es_problem = ElectronicStructureProblem(driver, transformers=[active_space_trafo])
    second_q_op = es_problem.second_q_ops()
    # return es_problem.grouped_property_transformed.get_property("ElectronicEnergy").nuclear_repulsion_energy
    qubit_converter = QubitConverter(mapper=ParityMapper(), two_qubit_reduction=True)
    qubit_op = qubit_converter.convert(second_q_op[0], num_particles=es_problem.num_particles)
    print(qubit_op)
    for pauli_op in qubit_op.to_pauli_op().oplist:
        pauli_dict[pauli_op.primitive.to_label()] = pauli_op.coeff
    n_qubit = len(qubit_op.to_pauli_op().oplist[0].primitive.to_label())
    return pauli_dict, n_qubit
def pauli_dict_list(pauli):
    if(pauli=='BeH'):
        pauli_dict = {
            'III': -12.488598,
            'IIZ': -0.85829425,
            'IZI': -0.85829425,
            'IZZ': 0.0230431788,
            'ZII': -0.85829425,
            'ZIZ': 0.0230431788,
            'ZZI': 0.0230431788,
            'ZZZ': 0.642470739,
            'IIX': -0.0434044897,
            'ZZX': -0.0434044897,
            'IXI': -0.0434044897,
            'ZXZ': -0.0434044897,
            'IXX': 0.0121246896,
            'IYY': 0.0121246896,
            'XII': -0.0434044897,
            'XZZ': -0.0434044897,
            'XIX': 0.0121246896,
            'YIY': 0.0121246896,
            'XXI': 0.0121246896,
            'YYI': 0.0121246896
            }
        n_qubit = 3
    return pauli_dict, n_qubit
def extract(pulse_prog):
    amp_list = list(map(lambda x: x[1].pulse.amp, pulse_prog.blocks[0].operands[0].filter(is_parametric_pulse).instructions))
    amp_list = np.array(amp_list)
    ampa_list = np.angle(np.array(amp_list))
    return ampa_list
    # ampn_list = np.abs(np.array(amp_list))
    # amps_list = []
    # for i,j in zip(ampn_list, ampa_list):
    #     amps_list.append(i)
    #     amps_list.append(j)
    # amps_list = np.array(amps_list)
    # return amps_list

def is_parametric_pulse(t0, *inst: Union['Schedule', Instruction]):
    inst = t0[1]
    t0 = t0[0]
    if isinstance(inst, pulse.Play):
        return True
    else:
        return False



def drag_pulse(backend, amp, angle):
  backend_defaults = backend.defaults()
  inst_sched_map = backend_defaults.instruction_schedule_map 
  x_pulse = inst_sched_map.get('x', (0)).filter(channels = [DriveChannel(0)], instruction_types=[Play]).instructions[0][1].pulse
  duration_parameter = x_pulse.parameters['duration']
  sigma_parameter = x_pulse.parameters['sigma']
  beta_parameter = x_pulse.parameters['beta']
  pulse1 = Drag(duration=duration_parameter, sigma=sigma_parameter, beta=beta_parameter, amp=amp, angle=angle)
  return pulse1

def cr_pulse(backend, amp, angle, duration):
  backend_defaults = backend.defaults()
  inst_sched_map = backend_defaults.instruction_schedule_map 
  cr_pulse = inst_sched_map.get('cx', (0, 1)).filter(channels = [ControlChannel(0)], instruction_types=[Play]).instructions[0][1].pulse
  cr_params = {}
  cr_params['duration'] = cr_pulse.parameters['duration']
  cr_params['amp'] = cr_pulse.parameters['amp']
  cr_params['angle'] = cr_pulse.parameters['angle']
  cr_params['sigma'] = cr_pulse.parameters['sigma']
  cr_params['width'] = cr_pulse.parameters['width']
  cr_risefall = (cr_params['duration'] - cr_params['width']) / (2 * cr_params['sigma'])
  angle_parameter = angle
  duration_parameter =  duration
  sigma_parameter = cr_pulse.parameters['sigma']
  width_parameter = int(duration_parameter - 2 * cr_risefall * cr_params['sigma'])
  #declare pulse parameters and build GaussianSquare pulse
  pulse1 = GaussianSquare(duration = duration_parameter, amp = amp, angle = angle_parameter, sigma = sigma_parameter, width=width_parameter)
  return pulse1
def cr_pulsefixedamp(backend, angle, duration):
  backend_defaults = backend.defaults()
  inst_sched_map = backend_defaults.instruction_schedule_map 
  cr_pulse = inst_sched_map.get('cx', (0, 1)).filter(channels = [ControlChannel(0)], instruction_types=[Play]).instructions[0][1].pulse
  cr_params = {}
  cr_params['duration'] = cr_pulse.parameters['duration']
  amp_parameter = cr_pulse.parameters['amp']
  cr_params['angle'] = cr_pulse.parameters['angle']
  cr_params['sigma'] = cr_pulse.parameters['sigma']
  cr_params['width'] = cr_pulse.parameters['width']
  cr_risefall = (cr_params['duration'] - cr_params['width']) / (2 * cr_params['sigma'])
  angle_parameter = angle
  duration_parameter =  duration
  sigma_parameter = cr_pulse.parameters['sigma']
  width_parameter = int(duration_parameter - 2 * cr_risefall * cr_params['sigma'])
  #declare pulse parameters and build GaussianSquare pulse
  pulse1 = GaussianSquare(duration = duration_parameter, amp = amp_parameter, angle = angle_parameter, sigma = sigma_parameter, width=width_parameter)
  return pulse1

def Decaylayer_pulse2q(backend, amp, angle, width):
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          qubits = (0,1)
          for i in range(2):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(qubits[i]))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2: #这里control channel可以根据目标改一下
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulse(backend,amp[2], angle[2], width[0]), uchan)
      sched_list.append(sched2)
      

      with pulse.build(backend) as sched7:

          pulse.play(drag_pulse(backend, amp[3], angle[3]), DriveChannel(1))
      sched_list.append(sched7)
    

    
      with pulse.build(backend) as my_program:
        with pulse.transpiler_settings(initial_layout= [0,1]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)
#              qubits = (0,1)
#              pulse.measure(qubits)

    return my_program


def Decaylayer_pulse2qfixedamp(backend, amp, angle, width):
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          qubits = (0,1)
          for i in range(2):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(qubits[i]))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2: #这里control channel可以根据目标改一下
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulsefixedamp(backend, angle[2], width[0]), uchan)
      sched_list.append(sched2)
      

      with pulse.build(backend) as sched7:

          pulse.play(drag_pulse(backend, amp[2], angle[3]), DriveChannel(1))
      sched_list.append(sched7)
    

    
      with pulse.build(backend) as my_program:
        with pulse.transpiler_settings(initial_layout= [0,1]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)


    return my_program
def HE_pulse2q(backend, amp, angle, width):
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          qubits = (0,1)
          for i in range(2):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(qubits[i]))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2: #这里control channel可以根据目标改一下
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulse(backend,amp[2], angle[2], width[0]), uchan)
      sched_list.append(sched2)


    
      with pulse.build(backend) as my_program:
        with pulse.transpiler_settings(initial_layout= [0,1]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)


    return my_program


def HE_pulse2qfixedamp(backend, amp, angle, width):
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          qubits = (0,1)
          for i in range(2):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(qubits[i]))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2: #这里control channel可以根据目标改一下
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulsefixedamp(backend, angle[2], width[0]), uchan)
      sched_list.append(sched2)


    
      with pulse.build(backend) as my_program:
        with pulse.transpiler_settings(initial_layout= [0,1]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)


    return my_program
def HE_pulse_3q(backend, amp, angle, width):
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          qubits = (0,1,2)
          for i in range(3):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(qubits[i]))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2: #这里control channel可以根据目标改一下
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulse(backend,amp[3], angle[3], width[0]), uchan)
      sched_list.append(sched2)
      

      with pulse.build(backend) as sched4:
          uchan = pulse.control_channels(1, 2)[0]
          pulse.play(cr_pulse(backend, amp[4], angle[4], width[1]), uchan)
      sched_list.append(sched4)

    
      with pulse.build(backend) as my_program:
        with pulse.transpiler_settings(initial_layout= [0,1,2]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)

    return my_program
def HE_pulse(backend, amp, angle, width):
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          qubits = (0,1,2,3)
          for i in range(4):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(qubits[i]))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2: #这里control channel可以根据目标改一下
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulse(backend,amp[4], angle[4], width[0]), uchan)
      sched_list.append(sched2)
      

      with pulse.build(backend) as sched4:
          uchan = pulse.control_channels(1, 2)[0]
          pulse.play(cr_pulse(backend, amp[5], angle[5], width[1]), uchan)
      sched_list.append(sched4)

      with pulse.build(backend) as sched6:
          uchan = pulse.control_channels(2,3)[0]
          pulse.play(cr_pulse(backend, amp[6],angle[6], width[2]), uchan)
      sched_list.append(sched6)

    
      with pulse.build(backend) as my_program:
        with pulse.transpiler_settings(initial_layout= [0,1,2,3]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)

    return my_program
def block_dressedpulse2qfixedamp(backend, amp, angle, width):
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          for i in range(2):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(i))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2: #这里control channel可以根据目标改一下
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulsefixedamp(backend, angle[2], width[0]), uchan)
      sched_list.append(sched2)
      
      with pulse.build(backend) as sched3:
          qubits = (0,1)
          for i in qubits:
              pulse.play(drag_pulse(backend, amp[2+i], angle[3+i]), DriveChannel(i))
      sched_list.append(sched3)

      with pulse.build(backend) as my_program:
        with pulse.transpiler_settings(initial_layout= [0,1]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)
             
             

    return my_program

def block_dressedpulse2q(backend, amp, angle, width):
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          for i in range(2):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(i))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2: #这里control channel可以根据目标改一下
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulse(backend, amp[2], angle[2], width[0]), uchan)
      sched_list.append(sched2)
      
      with pulse.build(backend) as sched3:
          qubits = (0,1)
          for i in qubits:
              pulse.play(drag_pulse(backend, amp[3+i], angle[3+i]), DriveChannel(i))
      sched_list.append(sched3)

      with pulse.build(backend) as my_program:
        with pulse.transpiler_settings(initial_layout= [0,1]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)
    

    return my_program


def HE_pulsefixedamp(backend, amp, angle, width):
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          qubits = (0,1,2,3)
          for i in range(4):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(qubits[i]))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2: #这里control channel可以根据目标改一下
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulsefixedamp(backend, angle[4], width[0]), uchan)
      sched_list.append(sched2)
      

      with pulse.build(backend) as sched4:
          uchan = pulse.control_channels(1, 2)[0]
          pulse.play(cr_pulsefixedamp(backend, angle[5], width[1]), uchan)
      sched_list.append(sched4)

      with pulse.build(backend) as sched6:
          uchan = pulse.control_channels(2,3)[0]
          pulse.play(cr_pulsefixedamp(backend,angle[6], width[2]), uchan)
      sched_list.append(sched6)

    
      with pulse.build(backend) as my_program:
        with pulse.transpiler_settings(initial_layout= [0,1,2,3]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)
    return my_program

def block_pulse(backend, amp, angle, width):
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          for i in range(2):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(i))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2: #这里control channel可以根据目标改一下
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulse(backend,amp[2], angle[2], width[0]), uchan)
      sched_list.append(sched2)
      
      with pulse.build(backend) as sched3:
          qubits = (0,1,2)
          for i in range(3):
              pulse.play(drag_pulse(backend, amp[3+i], angle[3+i]), DriveChannel(qubits[i]))
      sched_list.append(sched3)

      with pulse.build(backend) as sched4:
          uchan = pulse.control_channels(1, 2)[0]
          pulse.play(cr_pulse(backend, amp[6], angle[6], width[1]), uchan)
      sched_list.append(sched4)

      with pulse.build(backend) as sched5:
          qubits = (1,2,3)
          for i in range(3):
              pulse.play(drag_pulse(backend, amp[7+i], angle[7+i]), DriveChannel(qubits[i]))
      sched_list.append(sched5)

      with pulse.build(backend) as sched6:
          uchan = pulse.control_channels(2,3)[0]
          pulse.play(cr_pulse(backend, amp[10],angle[10], width[2]), uchan)
      sched_list.append(sched6)

      with pulse.build(backend) as sched7:
          qubits = (2,3)
          for i in range(2):
              pulse.play(drag_pulse(backend, amp[11+i], angle[11+i]), DriveChannel(qubits[i]))
      sched_list.append(sched7)
    
      with pulse.build(backend) as my_program:
        with pulse.transpiler_settings(initial_layout= [0,1,2,3]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)

    return my_program

def block_pulsefixedamp(backend, amp, angle, width):
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          for i in range(2):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(i))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2: #这里control channel可以根据目标改一下
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulsefixedamp(backend, angle[2], width[0]), uchan)
      sched_list.append(sched2)
      
      with pulse.build(backend) as sched3:
          qubits = (0,1,2)
          for i in range(3):
              pulse.play(drag_pulse(backend, amp[2+i], angle[3+i]), DriveChannel(qubits[i]))
      sched_list.append(sched3)

      with pulse.build(backend) as sched4:
          uchan = pulse.control_channels(1, 2)[0]
          pulse.play(cr_pulsefixedamp(backend, angle[6], width[1]), uchan)
      sched_list.append(sched4)

      with pulse.build(backend) as sched5:
          qubits = (1,2,3)
          for i in range(3):
              pulse.play(drag_pulse(backend, amp[5+i], angle[7+i]), DriveChannel(qubits[i]))
      sched_list.append(sched5)

      with pulse.build(backend) as sched6:
          uchan = pulse.control_channels(2,3)[0]
          pulse.play(cr_pulsefixedamp(backend, angle[10], width[2]), uchan)
      sched_list.append(sched6)

      with pulse.build(backend) as sched7:
          qubits = (2,3)
          for i in range(2):
              pulse.play(drag_pulse(backend, amp[8+i], angle[11+i]), DriveChannel(qubits[i]))
      sched_list.append(sched7)
    
      with pulse.build(backend) as my_program:
        with pulse.transpiler_settings(initial_layout= [0,1,2,3]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)

    return my_program

def Decaylayer_pulse(backend, amp, angle, width):
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          qubits = (0,1,2,3)
          for i in range(4):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(qubits[i]))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2: #这里control channel可以根据目标改一下
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulse(backend,amp[4], angle[4], width[0]), uchan)
          uchan1 = pulse.control_channels(3,2)[0]
          pulse.play(cr_pulse(backend, amp[5],angle[5], width[1]), uchan1)
      sched_list.append(sched2)
      

      with pulse.build(backend) as sched7:
          qubits = (1,2)
          for i in range(2):
              pulse.play(drag_pulse(backend, amp[6+i], angle[6+i]), DriveChannel(qubits[i]))
      sched_list.append(sched7)
    
      with pulse.build(backend) as sched4:
          uchan = pulse.control_channels(1, 2)[0]
          pulse.play(cr_pulse(backend, amp[8], angle[8], width[2]), uchan)
      sched_list.append(sched4)

    
      with pulse.build(backend) as my_program:
        with pulse.transpiler_settings(initial_layout= [0,1,2,3]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)

    return my_program

def Decaylayer_pulsefixedamp(backend, amp, angle, width):
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          qubits = (0,1,2,3)
          for i in range(4):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(qubits[i]))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2: #这里control channel可以根据目标改一下
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulsefixedamp(backend, angle[4], width[0]), uchan)
          uchan1 = pulse.control_channels(3,2)[0]
          pulse.play(cr_pulsefixedamp(backend,angle[5], width[1]), uchan1)
      sched_list.append(sched2)
      

      with pulse.build(backend) as sched7:
          qubits = (1,2)
          for i in range(2):
              pulse.play(drag_pulse(backend, amp[4+i], angle[6+i]), DriveChannel(qubits[i]))
      sched_list.append(sched7)
    
      with pulse.build(backend) as sched4:
          uchan = pulse.control_channels(1, 2)[0]
          pulse.play(cr_pulsefixedamp(backend,angle[8], width[2]), uchan)
      sched_list.append(sched4)

    
      with pulse.build(backend) as my_program:
        with pulse.transpiler_settings(initial_layout= [0,1,2,3]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)

    return my_program

