# Parameterized Quantum Pulse
This is the tool for measure the expressivity, entanglement capability, effective parameter dimention of the parameterized quantum pulse. And 6 different pulse-level design space available for application benchmark. 

## 1. Application Benchmark
please install all the requirement on your VM or conda.

```python
pip -r requirements.txt
```

Ready for the application?
At this very early version, we provide two applications: VQE for Ground state energy of quantum chemistry, and portfolio optimization of quantum finance.

We have args:

```python
'--backend',  type=str,   default='ibmq_quito',help='name of the backend(not a simulator)')
'--optimizer',type=str,   default='COBYLA',    help='name of the non-gradient optimizer')
'--policy',   type=str,   default='cxrx',      help='name of the pulse growth policy, deleted in this version')
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
'--n_parameters',   type=int, default=7,       help='number of parameters in pulse ansatz')
```
Please noticed if you want to do application as chemistry, you should to also define the molecule, if you don't, then the molecule will be default to H2.
And if you want to do application as finance, please indicate the n_assets, otherwise, it is automatically set to 2.
And be sure to realize what kind of pulse-level design space that you want. 
Pulse ID 1,2,3,4,5,6 are corresponding to Hardware-efficient(HE) pulse, fixed CR amp HE pulse, Decay-layer pulse, fixed CR amp Decay-layer pulse, Dressed pulse, fixed CR dressed pulse. For the detail structure of these pulses, please refer to our paper.

