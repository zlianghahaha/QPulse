import pdb
from turtle import width
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from numpy.linalg import matrix_rank
import random
import qiskit
import numpy as np
from qiskit.circuit.library import TwoLocal, RealAmplitudes
from qiskit import pulse, QuantumCircuit,circuit,transpile, providers
from qiskit.circuit import Gate
from qiskit.pulse import library, Schedule,GaussianSquare, ControlChannel, Play, Drag,DriveChannel,Delay,DriveChannel, SymbolicPulse, ShiftPhase
from qiskit.pulse import transforms
from qiskit.pulse.transforms import block_to_schedule,remove_directives
from qiskit.pulse import filters
from qiskit.pulse.filters import composite_filter, filter_instructions
from typing import List, Tuple, Iterable, Union, Dict, Callable, Set, Optional, Any
from qiskit.tools.visualization import circuit_drawer
from qiskit.visualization.pulse_v2 import draw, IQXSimple,IQXDebugging
from qiskit.compiler import assemble, schedule
from qiskit import IBMQ
import paddle
import paddle_quantum as pq
from paddle_quantum.ansatz.circuit import Circuit
from paddle_quantum.qinfo import state_fidelity, partial_trace, purity
import warnings

import qiskit.quantum_info as qi


"""## 2.2 Scan amp"""

import random
import decimal
import numpy as np
import paddle
import paddle_quantum as pq
from paddle_quantum.ansatz.circuit import Circuit
from paddle_quantum.visual import plot_state_in_bloch_sphere
from paddle_quantum.linalg import haar_unitary
from paddle_quantum.qinfo import state_fidelity
from paddle_quantum.state.common import to_state
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import integrate
from qiskit.providers.fake_provider.backends.athens.fake_athens import FakeAthens
import copy
import qiskit
# import multiset
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit import pulse, QuantumCircuit, IBMQ, visualization,execute, Aer
from qiskit.pulse import library
from qiskit.visualization.pulse_v2.stylesheet import IQXDebugging
from qiskit.visualization.pulse_v2 import draw
from qiskit.providers.fake_provider import FakeQuito, FakeBelem, FakeSantiago, FakeManila, FakeLagos, FakeLima,FakeJakarta, FakeAthens
from qiskit.pulse import transforms
from qiskit.pulse.transforms import block_to_schedule
from qiskit.pulse import filters
from qiskit.pulse.filters import composite_filter, filter_instructions
from typing import List, Tuple, Iterable, Union, Dict, Callable, Set, Optional, Any
from qiskit.pulse.instructions import Instruction
from qiskit.compiler import assemble, schedule
from qiskit.providers.aer.pulse import PulseSystemModel
from qiskit.providers.aer import PulseSimulator
from qiskit_dynamics import Solver, DynamicsBackend
from qiskit_dynamics.backend import default_experiment_result_function
from qiskit_dynamics.array import Array
import jax
backend = FakeQuito()

backend.configuration().hamiltonian['qub']

backend.configuration().hamiltonian['qub'] = {'0': 2,'1': 2,'2': 2,'3': 2,'4': 2 }
backend_model = PulseSystemModel.from_backend(backend)
backend_sim = PulseSimulator(system_model=backend_model)
backend_config = backend.configuration().to_dict()
num_qubits = int(backend_config['n_qubits'])
f = backend.properties().frequency
freq = [f(i) for i in range(num_qubits)]
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

Array.set_default_backend("jax")

# set a small value for epsilon to use in calculating the gradient
epsilon = 1e-2
def drag_pulse(backend, amp, angle):
  backend_defaults = backend.defaults()
  inst_sched_map = backend_defaults.instruction_schedule_map 
  x_pulse = inst_sched_map.get('x', (0)).filter(channels = [DriveChannel(0)], instruction_types=[Play]).instructions[0][1].pulse
  duration_parameter = x_pulse.parameters['duration']
  sigma_parameter = x_pulse.parameters['sigma']
  beta_parameter = x_pulse.parameters['beta']
  pulse1 = Drag(duration=duration_parameter, sigma=sigma_parameter, beta=beta_parameter, amp=amp, angle=angle)
  return pulse1
def new_experiment_result_function(
    experiment_name,
    solver_result,
    measurement_subsystems,
    memory_slot_indices,
    num_memory_slots,
    backend,
    seed,
    metadata,
):
    result = default_experiment_result_function(
        experiment_name,
        solver_result,
        measurement_subsystems,
        memory_slot_indices,
        num_memory_slots,
        backend,
        seed,
        metadata,
    )
    
    result.statevector = solver_result.y[-1]
    return result
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
def block_pulse2q(backend, amp, angle, width):
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

      with pulse.build(backend) as sched4:
          uchan = pulse.control_channels(0, 1)[0]
          pulse.play(cr_pulse(backend, amp[5], angle[5], width[1]), uchan)
      sched_list.append(sched4)

      with pulse.build(backend) as sched5:
          qubits = (0,1)
          for i in qubits:
              pulse.play(drag_pulse(backend, amp[6+i], angle[6+i]), DriveChannel(i))
      sched_list.append(sched5)

      with pulse.build(backend) as my_program:
        with pulse.transpiler_settings(initial_layout= [0,1]):
          with pulse.align_sequential():
              for sched in sched_list:
                  pulse.call(sched)
              pulse.measure(qubits)

    return my_program

def block_pulse2qdressed(backend, amp, angle, width):
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
              pulse.measure(qubits)

    return my_program
def get_parameterzied_circuit(params_list):
    """
    Creates a parameterized quantum circuit with two qubits and two parameters.
    """
    params = params_list[:int(len(params_list)-2)]
    amp = params[:int(len(params)/2)]
    params_arr = np.array(params)
    angle = params_arr[int(len(params_arr)/2):] * 2*np.pi
    width = params_list[len(params_list)-2:]
    width = [round(w * 100*16) for w in width]
    sched1 = block_pulse2q(backend, amp, angle, width)
    return sched1

def get_statevector(params):
    """
    Computes the statevector of a parameterized quantum circuit.
    """
    backend = FakeQuito()

    backend.configuration().hamiltonian['qub']

    backend.configuration().hamiltonian['qub'] = {'0': 2,'1': 2,'2': 2,'3': 2,'4': 2 }
    backend_model = PulseSystemModel.from_backend(backend)
    backend_sim = PulseSimulator(system_model=backend_model)
    backend_config = backend.configuration().to_dict()
    num_qubits = int(backend_config['n_qubits'])
    f = backend.properties().frequency
    freq = [f(i) for i in range(num_qubits)]
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")

    Array.set_default_backend("jax")
    backend_run = DynamicsBackend.from_backend(backend, evaluation_mode="sparse")
    solver_options = {"method": "jax_odeint", "atol": 1e-6, "rtol": 1e-8}
    backend_run.set_options(solver_options=solver_options,experiment_result_function=new_experiment_result_function)
    backend_run.configuration = lambda: backend.configuration()
    # Returen the density matrix of q0
    #vec = results.get_statevector()
    # print(len(vec))
    #rho = qi.partial_trace(vec, [2,3,4])
    #Returen the density matrix of q0
    sched = get_parameterzied_circuit(params)
    results = backend_run.run(sched).result()
    statevector = results.results[0].statevector
    return statevector

def get_gradient(params):
    """
    Computes the gradient of the statevector of a parameterized quantum circuit with respect
    to its parameters using finite differences.
    """
    statevector0 = get_statevector(params)
    gradient = [statevector0]
    for i in range(len(params)):
        params_perturbed = copy.deepcopy(params)
        params_perturbed[i] += epsilon
        statevector_perturbed = get_statevector(params_perturbed)
        gradient.append((statevector_perturbed - statevector0)/epsilon)
    return gradient

def get_fisher(params):
    """
    Computes the Fisher information matrix for a parameterized quantum circuit.
    """
    gradient = get_gradient(params)
    fisher = np.zeros((len(params), len(params)))
    for i in range(len(params)):
        for j in range(i, len(params)):
            fisher[i,j] = np.real(gradient[i+1].inner(gradient[j+1]) - gradient[i+1].inner(gradient[0])*gradient[j+1].inner(gradient[0]))
            fisher[j,i] = fisher[i,j]
    return fisher

def get_epd(params):
    """
    Computes the effective parameter dimension (EPD) for a parameterized quantum circuit.
    """
    fisher = get_fisher(params)
    epd = matrix_rank(fisher, tol=1e-6)  # compute the rank of the Fisher information matrix
    return epd

# Example usage 
if __name__ == '__main__':
  #random.seed(42)
  pdb.set_trace()
  random_params = [random.uniform(0, 1000)/1000 for i in range(16)]
  width = [random.randrange(16, 64)/100 for _ in range(2)]
  params = random_params + width
  result = get_epd(params)
  print(result)