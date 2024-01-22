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

def generate_numbers(n):
    return list(range(n))

def measurement_pauli(prepulse, pauli_string, backend, n_qubit):
    # n = n_qubit
    # numbers = generate_numbers(n)
    with pulse.build(backend) as pulse_measure:
        pulse.call(copy.deepcopy(prepulse))
        # pulse.barrier(*numbers)
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
    pauli_dict = {}
    n_qubit = 0
    if(pauli=='BeH'):
        pauli_dict = {
            'III': -12.488598002651747,
            'IIZ': -0.8582942496558701,
            'IZI': -0.8582942496558701,
            'IZZ': 0.02304317881962037,
            'ZII': -0.8582942496558702,
            'ZIZ': 0.02304317881962037,
            'ZZI': 0.02304317881962037,
            'ZZZ': 0.6424707389584848,
            'IIX': -0.04340448973076078,
            'ZZX': -0.04340448973076078,
            'IXI': -0.04340448973076078,
            'ZXZ': -0.04340448973076078,
            'IXX': 0.01212468961058548,
            'IYY': 0.01212468961058548,
            'XII': -0.04340448973076078,
            'XZZ': -0.04340448973076078,
            'XIX': 0.01212468961058548,
            'YIY': 0.01212468961058548,
            'XXI': 0.01212468961058548,
            'YYI': 0.01212468961058548
            }
        n_qubit = 3
    elif(pauli=='H2_3-21G'):
        pauli_dict = {
            'IIII': 2.86,
            'IIIZ': -1.116,
            'IIZI': -1.116,
            'IIZZ': 0.289,
            'IZII': -1.582,
            'IZIZ': 0.052,
            'IZZI': 0.052,
            'ZIII': -0.44,
            'ZIIZ': -0.305,
            'ZIZI': 0.234,
            'ZIZZ': -0.44,
            'ZZII': -0.103,
            'ZZIZ': 0.598,
            'ZZZZ': -0.103,
            'IIIX': -0.02,
            'IZZX': -0.02,
            'ZIZX': -0.02,
            'ZZIX': -0.02,
            'IIXI': -0.028,
            'IIXZ': 0.058,
            'IZXZ': -0.104,
            'ZIXI': -0.058,
            'ZIXZ': 0.028,
            'ZZXI': 0.104,
            'IIXX': 0.033,
            'IIYY': -0.033,
            'ZZXX': 0.033,
            'ZZYY': -0.033,
            'IXII': 0.057,
            'ZXIZ': 0.057,
            'IXIX': -0.002,
            'IYIY': -0.002,
            'ZXZX': -0.002,
            'ZYZY': -0.002,
            'IXXX': 0.036,
            'IXYY': -0.036,
            'IYXY': 0.036,
            'IYYX': 0.036,
            'XIII': -0.023,
            'XIZZ': -0.023,
            'XZIZ': -0.023,
            'XZZI': -0.023,
            'XIIX': -0.028,
            'XIZX': 0.058,
            'XZZX': -0.104,
            'YIIY': -0.028,
            'YIZY': 0.058,
            'YZZY': -0.104,
            'XIXI': -0.02,
            'XZXZ': -0.02,
            'YIYI': 0.02,
            'YZYZ': 0.02,
            'XIYY': 0.036,
            'YIYX': -0.036,
            'XXII': -0.009,
            'XXZZ': -0.009,
            'YYII': -0.009,
            'YYZZ': -0.009,
            'XXXI': -0.002,
            'XYYI': -0.002,
            'YXYI': 0.002,
            'YYXI': -0.002
            }
        n_qubit = 4
    elif(pauli=='NH'):
        pauli_dict = {
            'IIII': -51.99474022263152,
            'IIIZ': 0.35672725289327645,
            'IZII': 0.6759674464334343,
            'IZIZ': 0.8815743038588083,
            'ZIII': 0.6759687393467231,
            'ZIIZ': 0.8815745954966026,
            'ZZII': 0.3230574110776512,
            'ZZIZ': 0.2941393870513714,
            'IIIX': -0.023966777845131182,
            'IZIX': 0.02396678099725029,
            'ZIIX': 0.023966777845131182,
            'ZZIX': -0.02396678099725029,
            'IIXI': 0.0411185184723139,
            'IXII': 0.015622914899415839,
            'IXIZ': -0.015622238828915232,
            'ZXII': 0.05468197366233466,
            'ZXIZ': -0.05468129744499403,
            'IXIX': -0.014792316398698721,
            'IYIY': -0.014792005171018535,
            'ZXIX': 0.014792316398698721,
            'ZYIY': 0.014792005171018535,
            'XIII': -0.01562223688013262,
            'XIIZ': 0.01562223688013262,
            'XZII': -0.054680930653391,
            'XZIZ': 0.054680930653391,
            'XIIX': 0.014791638379415505,
            'XZIX': -0.01479220679313444,
            'YIIY': 0.014791638379415505,
            'YZIY': -0.01479220679313444,
            'YYII': -0.06399736836915412,
            'YYIZ': 0.06399736836915412,
            'XXIX': -0.03452814024579984,
            'XYIY': -0.03452843188359385,
            'YXIY': -0.03452814024579984,
            'YYIX': 0.03452843188359385
            }
        n_qubit = 4
    elif(pauli=='BeH+'):
        pauli_dict = {
            'IIIII': -12.436191341883138,
            'IIIIZ': -0.6064933312208174,
            'IIIZI': -0.606493331154895,
            'IIIZZ': 0.2583925428543069,
            'IIZII': -0.856046307528914,
            'IIZIZ': 0.1805169482770208,
            'IIZZI': 0.1805169482770208,
            'IZIII': -0.856046307528914,
            'IZIIZ': 0.1805169482770208,
            'IZIZI': 0.1805169482770208,
            'IZZII': 0.3892355930337422,
            'IZZIZ': -0.15184739578570627,
            'IZZZI': -0.15184739607301742,
            'ZIIII': -0.37015772409612363,
            'ZIIIZ': 0.15173753720327138,
            'ZIIZI': 0.1752684711882569,
            'ZIIZZ': -0.3701577245556942,
            'ZIZII': 0.1866004758975273,
            'ZIZIZ': -0.34861115038365303,
            'ZIZZZ': 0.18660047613703334,
            'ZZIII': 0.1866004758975273,
            'ZZIIZ': -0.34861115038365303,
            'ZZIZZ': 0.18660047613703334,
            'ZZZII': -0.23364280391756703,
            'ZZZIZ': 0.46210182765773133,
            'ZZZZZ': -0.23364280391756703,
            'IIIIX': 0.011451012557330793,
            'IZZZX': 0.011451012557330793,
            'ZIIZX': 0.011451012557330793,
            'ZZZIX': 0.011451012557330793,
            'IIIXI': 0.028039531880619493,
            'IIIXZ': 0.04136144411713835,
            'IIZXZ': -0.01205966499300103,
            'IZIXZ': -0.01205966499300103,
            'IZZXZ': 0.016622529824795222,
            'ZIIXI': -0.04136144411713835,
            'ZIIXZ': -0.028039531880619493,
            'ZIZXI': 0.01205966499300103,
            'ZZIXI': 0.01205966499300103,
            'ZZZXI': -0.016622529824795222,
            'IIIXX': 0.030372505622554034,
            'IIIYY': -0.030372505622554034,
            'ZZZXX': 0.030372505622554034,
            'ZZZYY': -0.030372505622554034,
            'IIXII': -0.021479567258249697,
            'ZZXIZ': -0.021479567258249697,
            'IIXIX': -0.0037175632507374175,
            'IIYIY': -0.0037175632507374175,
            'ZIXZX': -0.0037175632507374175,
            'ZIYZY': -0.0037175632507374175,
            'IIXXX': -0.006677525207565469,
            'IIXYY': 0.006677525207565469,
            'IIYXY': -0.006677525207565469,
            'IIYYX': -0.006677525207565469,
            'IXIII': -0.021479567258249697,
            'ZXZIZ': -0.021479567258249697,
            'IXIIX': -0.0037175632507374175,
            'IYIIY': -0.0037175632507374175,
            'ZXIZX': -0.0037175632507374175,
            'ZYIZY': -0.0037175632507374175,
            'IXIXX': -0.006677525207565469,
            'IXIYY': 0.006677525207565469,
            'IYIXY': -0.006677525207565469,
            'IYIYX': -0.006677525207565469,
            'IXXII': 0.012124689610585477,
            'IYYII': 0.012124689610585477,
            'XIIII': -0.008380307801670097,
            'XIIZZ': -0.008380307749909404,
            'XZZIZ': -0.008380307749909404,
            'XZZZI': -0.008380307801670097,
            'XIIIX': 0.02803953188719743,
            'XIIZX': 0.0413614448433547,
            'XIZZX': -0.01205966511173538,
            'XZIZX': -0.01205966511173538,
            'XZZZX': 0.016622529824795222,
            'YIIIY': 0.028039531985268908,
            'YIIZY': 0.04136144427494598,
            'YIZZY': -0.01205966499300103,
            'YZIZY': -0.01205966499300103,
            'YZZZY': 0.016622530057662635,
            'XIIXI': 0.011451012526575878,
            'XZZXZ': 0.011451012557330793,
            'YIIYI': -0.011451012557330793,
            'YZZYZ': -0.011451012526575878,
            'XIIYY': 0.023530933984985476,
            'YIIYX': -0.023530933984985476,
            'XIXII': 0.013044728047586102,
            'XIXZZ': 0.013044728175954718,
            'YIYII': 0.013044728175954718,
            'YIYZZ': 0.013044728047586102,
            'XIXXI': -0.003717563335420835,
            'XIYYI': -0.003717563335420835,
            'YIXYI': 0.0037175632507374175,
            'YIYXI': -0.0037175632507374175,
            'XXIII': 0.013044728047586102,
            'XXIZZ': 0.013044728175954718,
            'YYIII': 0.013044728175954718,
            'YYIZZ': 0.013044728047586102,
            'XXIXI': -0.003717563335420835,
            'XYIYI': -0.003717563335420835,
            'YXIYI': 0.0037175632507374175,
            'YYIXI': -0.0037175632507374175
            }
        n_qubit = 5   
    elif(pauli=='F2'):
        pauli_dict = {
            'IIIIII': -186.64341821351084,
            'IIIIIZ': 2.3759526779155298,
            'IIIIZI': 2.577678525597112,
            'IIIIZZ': 0.6498038715260552,
            'IIIZII': 2.577678525597115,
            'IIIZIZ': 0.6498038715260552,
            'IIIZZI': 0.6168577049250952,
            'IIZIII': 1.3569401298819703,
            'IIZIIZ': 0.28433130911068116,
            'IIZIZI': 0.30533640177206156,
            'IIZZII': 0.30533640177206156,
            'IIZZZZ': -0.2554743123306841,
            'IZIIII': 1.3569401298819703,
            'IZIIIZ': 0.28433130911068116,
            'IZIIZI': 0.30533640177206156,
            'IZIZII': 0.30533640177206156,
            'IZIZZZ': -0.2554743123306841,
            'IZZIII': 0.34173892830297087,
            'ZIIIII': 3.2940310168997247,
            'ZIIIIZ': 0.610982058183648,
            'ZIIIZI': 0.5832181596455184,
            'ZIIZII': 0.5832181596455184,
            'ZIIZZZ': -0.2669700188979873,
            'ZIZIII': 0.2839399006319595,
            'ZIZIZZ': -0.3033913513199572,
            'ZIZZIZ': -0.3033913513199572,
            'ZIZZZI': -0.3273700391518314,
            'ZIZZZZ': -1.8177176642041462,
            'ZZIIII': 0.2839399006319595,
            'ZZIIZZ': -0.3033913513199572,
            'ZZIZIZ': -0.3033913513199572,
            'ZZIZZI': -0.3273700391518314,
            'ZZIZZZ': -1.8177176642041442,
            'ZZZZZZ': -0.3208782050383475,
            'IIIIIX': -0.02475074194750153,
            'IZZIIX': -0.02475074194750153,
            'ZIZZZX': -0.02475074194750153,
            'ZZIZZX': -0.02475074194750153,
            'IIIIXI': -0.020565680365610173,
            'IZZIXI': -0.020565680365610173,
            'ZIZZXZ': -0.020565680365610173,
            'ZZIZXZ': -0.020565680365610173,
            'IIIIXX': 0.017107071987866287,
            'IIIIYY': 0.017107071987866287,
            'IIIXII': -0.020565680365610173,
            'IZZXII': -0.020565680365610173,
            'ZIZXZZ': -0.020565680365610173,
            'ZZIXZZ': -0.020565680365610173,
            'IIIXIX': 0.017107071987866287,
            'IIIYIY': 0.017107071987866287,
            'IIIXXI': 0.014335370222937254,
            'IIIYYI': 0.014335370222937254,
            'IIXIII': -0.016889791581695342,
            'IZXIII': 0.066484887210813,
            'IZXIIZ': 0.02252387579900742,
            'IZXIZI': 0.017264149072982373,
            'IZXZII': 0.017264149072982373,
            'IZXZZZ': 0.030093350196134124,
            'ZIXZZZ': -0.016889791581695342,
            'ZZXIII': 0.030093350196134124,
            'ZZXIZZ': 0.017264149072982373,
            'ZZXZIZ': 0.017264149072982373,
            'ZZXZZI': 0.02252387579900742,
            'ZZXZZZ': 0.06648488721081298,
            'IIXIIX': 0.01385844081057167,
            'IIYIIY': 0.01385844081057167,
            'ZZXZZX': 0.01385844081057167,
            'ZZYZZY': 0.01385844081057167,
            'IIXIXI': -0.004886736590463941,
            'IIYIYI': -0.004886736590463941,
            'ZZXZXZ': -0.004886736590463941,
            'ZZYZYZ': -0.004886736590463941,
            'IIXXII': -0.004886736590463941,
            'IIYYII': -0.004886736590463941,
            'ZZXXZZ': -0.004886736590463941,
            'ZZYYZZ': -0.004886736590463941,
            'IXIIII': 0.016889791581695342,
            'IXZIII': -0.066484887210813,
            'IXZIIZ': -0.02252387579900742,
            'IXZIZI': -0.017264149072982373,
            'IXZZII': -0.017264149072982373,
            'IXZZZZ': -0.030093350196134124,
            'ZXIZZZ': 0.016889791581695342,
            'ZXZIII': -0.030093350196134124,
            'ZXZIZZ': -0.017264149072982373,
            'ZXZZIZ': -0.017264149072982373,
            'ZXZZZI': -0.02252387579900742,
            'ZXZZZZ': -0.06648488721081301,
            'IXIIIX': -0.01385844081057167,
            'IYIIIY': -0.01385844081057167,
            'ZXZZZX': -0.01385844081057167,
            'ZYZZZY': -0.01385844081057167,
            'IXIIXI': 0.004886736590463941,
            'IYIIYI': 0.004886736590463941,
            'ZXZZXZ': 0.004886736590463941,
            'ZYZZYZ': 0.004886736590463941,
            'IXIXII': 0.004886736590463941,
            'IYIYII': 0.004886736590463941,
            'ZXZXZZ': 0.004886736590463941,
            'ZYZYZZ': 0.004886736590463941,
            'IYYIII': -0.05390818614036015,
            'ZYYZZZ': -0.05390818614036015,
            'IXXIIX': -0.06777522614894527,
            'IXYIIY': -0.06777522614894527,
            'IYXIIY': -0.06777522614894527,
            'IYYIIX': 0.06777522614894527,
            'IXXIXI': -0.007538461819129052,
            'IXYIYI': -0.007538461819129052,
            'IYXIYI': -0.007538461819129052,
            'IYYIXI': 0.007538461819129052,
            'IXXXII': -0.007538461819129052,
            'IXYYII': -0.007538461819129052,
            'IYXYII': -0.007538461819129052,
            'IYYXII': 0.007538461819129052,
            'XIIIII': -0.05772853261876541,
            'XIZZZZ': -0.05772853261876541,
            'XZIZZZ': -0.05772853261876541,
            'XZZIII': -0.05772853261876541,
            'XIIIIX': 0.05000314744516725,
            'YIIIIY': 0.05000314744516725,
            'XIIIXI': 0.054198697403198756,
            'YIIIYI': 0.054198697403198756,
            'XIIXII': 0.054198697403198756,
            'YIIYII': 0.054198697403198756,
            'XIXIII': -0.029667590340709152,
            'XZXZZZ': -0.029667590340709152,
            'YIYIII': -0.029667590340709152,
            'YZYZZZ': -0.029667590340709152,
            'XXIIII': 0.029667590340709152,
            'XXZZZZ': 0.029667590340709152,
            'YYIIII': 0.029667590340709152,
            'YYZZZZ': 0.029667590340709152,
            'XXXIII': -0.03520116793901029,
            'XYYIII': 0.03520116793901029,
            'YXYIII': -0.03520116793901029,
            'YYXIII': -0.03520116793901029
            }
        n_qubit = 6   
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

      with pulse.build(backend) as sched2:  
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

      with pulse.build(backend) as sched2:  
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

      with pulse.build(backend) as sched2:  
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

      with pulse.build(backend) as sched2:  
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
      n_qubit = 3
      numbers = generate_numbers(n_qubit)
      sched_list = []
      with pulse.build(backend) as sched1:
          qubits = (0,1,2)
          for i in range(3):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(qubits[i]))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2:  
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulse(backend,amp[3], angle[3], width[0]), uchan)
      sched_list.append(sched2)
      

      with pulse.build(backend) as sched4:
          uchan = pulse.control_channels(1, 2)[0]
          pulse.play(cr_pulse(backend, amp[4], angle[4], width[1]), uchan)
      sched_list.append(sched4)

    
      with pulse.build(backend) as my_program:
        # with pulse.transpiler_settings(initial_layout= [0,1,2]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)
              pulse.barrier(*numbers)

    return my_program
def HE_pulse(backend, amp, angle, width):
    n_qubit = 4
    numbers = generate_numbers(n_qubit)
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          qubits = (0,1,2,3)
          for i in range(4):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(qubits[i]))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2:  
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
        # with pulse.transpiler_settings(initial_layout= [0,1,2]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)
              pulse.barrier(*numbers)

    return my_program

def HE_pulse5q(backend, amp, angle, width):
    qubits = (0,1,2,3,5)
    numbers = qubits
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          qubits = (0,1,2,3,5)
          for i in range(5):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(qubits[i]))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2:  
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulse(backend,amp[5], angle[5], width[0]), uchan)
      sched_list.append(sched2)
      

      with pulse.build(backend) as sched4:
          uchan = pulse.control_channels(1, 2)[0]
          pulse.play(cr_pulse(backend, amp[6], angle[6], width[1]), uchan)
      sched_list.append(sched4)

      with pulse.build(backend) as sched6:
          uchan = pulse.control_channels(2,3)[0]
          pulse.play(cr_pulse(backend, amp[7],angle[7], width[2]), uchan)
      sched_list.append(sched6)
      with pulse.build(backend) as sched7:
          uchan = pulse.control_channels(3,5)[0]
          pulse.play(cr_pulse(backend, amp[8],angle[8], width[3]), uchan)
      sched_list.append(sched7)
    
      with pulse.build(backend) as my_program:
        # with pulse.transpiler_settings(initial_layout= [0,1,2]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)
              pulse.barrier(*numbers)

    return my_program

def HE_pulse6q(backend, amp, angle, width):
    qubits = (0,1,2,3,5,8)
    numbers = qubits
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          qubits = (0,1,2,3,5,8)
          for i in range(6):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(qubits[i]))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2:  
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulse(backend,amp[6], angle[6], width[0]), uchan)
      sched_list.append(sched2)
      

      with pulse.build(backend) as sched4:
          uchan = pulse.control_channels(1, 2)[0]
          pulse.play(cr_pulse(backend, amp[7], angle[7], width[1]), uchan)
      sched_list.append(sched4)

      with pulse.build(backend) as sched6:
          uchan = pulse.control_channels(2,3)[0]
          pulse.play(cr_pulse(backend, amp[8],angle[8], width[2]), uchan)
      sched_list.append(sched6)
      with pulse.build(backend) as sched7:
          uchan = pulse.control_channels(3,5)[0]
          pulse.play(cr_pulse(backend, amp[9],angle[9], width[3]), uchan)
      sched_list.append(sched7)
      with pulse.build(backend) as sched8:
          uchan = pulse.control_channels(5,8)[0]
          pulse.play(cr_pulse(backend, amp[10],angle[10], width[4]), uchan)
      sched_list.append(sched8)
    
      with pulse.build(backend) as my_program:
        # with pulse.transpiler_settings(initial_layout= [0,1,2]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)
              pulse.barrier(*numbers)

    return my_program
def block_dressedpulse2qfixedamp(backend, amp, angle, width):
    with pulse.build(backend) as my_program1:
      # layer 1
      sched_list = []
      with pulse.build(backend) as sched1:
          for i in range(2):
              pulse.play(drag_pulse(backend, amp[i], angle[i]), DriveChannel(i))
      sched_list.append(sched1)

      with pulse.build(backend) as sched2:  
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

      with pulse.build(backend) as sched2:  
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

      with pulse.build(backend) as sched2:  
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

      with pulse.build(backend) as sched2: 
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

      with pulse.build(backend) as sched2: 
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

      with pulse.build(backend) as sched2: 
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

      with pulse.build(backend) as sched2: 
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

