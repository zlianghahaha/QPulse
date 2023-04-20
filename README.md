# Parameterized Quantum Pulse
This tool is designed to give a detailed guidance for how to design pulse-level quantum circuit.
At this early version we provide: measure the expressivity, entanglement capability, and effective parameter dimension of parameterized quantum pulses, and also provide 6 different suggested pulse-level design spaces for application benchmarking. For details about the design space, please check our paper [Towards Advantages of Parameterized Quantum Pulses](https://arxiv.org/pdf/2304.09253.pdf).

To begin, please install all the required packages on your virtual machine or conda environment by running:

```python
pip install -r requirements.txt
```

## 1. Metric Measurement
This section focuses on measuring the expressivity, entanglement capability, and effective parameter dimension of the parameterized quantum pulses. We provide metric measurement results for 6 proposed pulse-level design spaces, which are stored in `.pickle` format.

If you want to reproduce the result from the paper. Please go to `/MetricsMeasurement/data/`.

You'll find a Jupyter Notebook file [Pulse_Expressibility&Entanglement](https://github.com/zlianghahaha/ParameterizedQuantumPulse/blob/main/MetricsMeasurement/data/Pulse_Expressibility%26Entanglement.ipynb), Open the file and import data from `/MetricsMeasurement/data/metricsmeasurementdata`

As for expressivity, please go to `/MetricsMeasurement/Expressivity`, and check [Expressivity Example](https://github.com/zlianghahaha/ParameterizedQuantumPulse/blob/main/MetricsMeasurement/Expressivity/JakartaPulseVQA_Expressibility.ipynb).This example demonstrates how to calculate the expressivity of a single-qubit pulse, including methods to compute the measured result from the target pulse, theoretical result, and K-L divergence.

For entanglement capability, navigate to `/MetricsMeasurement/EntCapability` check [Entangement Capability Example](https://github.com/zlianghahaha/ParameterizedQuantumPulse/blob/main/MetricsMeasurement/EntCapability/Pulse_Ent_Example.ipynb).

As for effective parameter dimension, please go to `/MetricsMeasurement/EPD` check [Effective Parameter Dimension Example](https://github.com/zlianghahaha/ParameterizedQuantumPulse/blob/main/MetricsMeasurement/EPD/EPD_Example.ipynb).


## 2. Application Benchmark

Ready for the application? In this early version, we provide two applications: VQE for ground state energy of quantum chemistry and portfolio optimization for quantum finance.

We have the following arguments:

```python
'--backend',  type=str,   default='ibmq_quito',help='name of the backend(Or a simulator like FakeManila)')
'--optimizer',type=str,   default='COBYLA',    help='name of the non-gradient optimizer')
'--policy',   type=str,   default='cxrx',      help='name of the pulse growth policy, related to NAPA and have not added in this version.')
'--application',  type=str,   default='chemistry',      help='name of the benchmark application')
'--pulse_id', type=int,   default=1,           help='indicate the design space at pulse level.')
'--molecule', type=str,   default='H2',        help='name of the molecules')
'--n_assets', type=int,   default=2,           help='number of assets')
'--tune_freq',type=bool,  default=False,       help='specify if frequencies are tuned')
'--n_iter',   type=int,   default=100,         help='number of training iterations')
'--n_shot',   type=int,   default=1024,        help='number of shots for measurement')
'--n_step',   type=int,   default=1,           help='number of pulse_layers')
'--max_jobs', type=int,   default=8,           help='number of max_jobs for multiprocessing')
'--rhobeg',   type=float, default=0.1 ,        help='rhobeg for non-gradient optimizer')
```
Please note that if you want to run an application in chemistry, you should define the molecule. If you don't, the molecule will default to H2. And if you want to run an application in finance, please specify the `n_assets`, otherwise, it will be set to 2 automatically.
Also, be sure to select the desired pulse-level design space.

Pulse IDs 1, 2, 3, 4, 5, 6 correspond to Hardware-efficient (HE) pulse, fixed CR amp HE pulse, Decay-layer pulse, fixed CR amp Decay-layer pulse, Dressed pulse, and fixed CR dressed pulse. For detailed structures of these pulses, please refer to our paper.

[Example Script for Application Benchmark](https://github.com/zlianghahaha/ParameterizedQuantumPulse/blob/main/example_script.sh) provides two examples for quantum chemistry and quantum finance.

```python
python -W ignore -u main.py --backend=FakeManila --application=chemistry --pulse_id=1 --molecule=H2 > testchemistyH2HE2q.txt&

```

This example runs Qiskit-Dynamics with the system model from FakeManila, and the application is ground state energy for H2 in quantum chemistry. The pulse-level design space is hardware-efficient pulse with 7 parameters.
