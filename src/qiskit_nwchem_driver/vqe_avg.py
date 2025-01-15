import numpy as np
import pylab
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers.qubit_mapper import QubitMapper
from qiskit.primitives import Estimator

from quantum_gates.simulators import MrAndersonSimulator
from quantum_gates.gates import standard_gates, noise_free_gates, numerical_gates, almost_noise_free_gates 
from quantum_gates.circuits import EfficientCircuit, BinaryCircuit
from quantum_gates.utilities import DeviceParameters
from quantum_gates.utilities import setup_backend
from quantum_gates.utilities import fix_counts
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeValenciaV2, FakeKyoto, FakeVigoV2


from qiskit_aer import Aer, AerSimulator
from qiskit import transpile
from qiskit.circuit import Parameter
from qiskit.circuit import QuantumCircuit

from collections import defaultdict

backend_fake = FakeKyiv()
sim_noise_ander = MrAndersonSimulator(gates=standard_gates, CircuitClass=EfficientCircuit, parallel= False)
sim_free_ander = MrAndersonSimulator(gates=noise_free_gates, CircuitClass=EfficientCircuit, parallel= False)
sim_noise_aer = AerSimulator.from_backend(backend_fake)
sim_free_aer =AerSimulator()

n_qubit = 4
n_classic_bit = 4
linear_qubit_layout = [0,1,2,3] # linear layout
psi0 = [1] + [0] * (2**n_qubit-1) # starting state
shots = 100 # shots
device_param = DeviceParameters(linear_qubit_layout)
device_param.load_from_backend(backend_fake) # get parameters from the backend
device_param_lookup = device_param.__dict__() # get dict representation.

def run_vqe_with_averaging(ansatz, estimator, mapper, es_problem, n_repeats, maxiter, is_tUPS):
    """
    Run VQE multiple times, average the energy results and plot convergence.

    Parameters:
    - ansatz: The variational ansatz used in VQE
    - estimator: The estimator used to calculate the expectation value
    - mapper: The mapping used for the qubit operators
    - es_problem: The electronic structure problem
    - n_repeats: The number of VQE runs to average (default is 100)

    Returns:
    - avergare_values: Average values for each VQE step'
    - max_evaluations: number of iterations 
    - vqe_solver: VQE object
    - result: minimum eigenenergy
    """
    
    # Initialize variables to store all results
    all_counts = []  # List to store evaluation counts for each run
    all_values = []  # List to store energy values for each run
    all_probs_noise_ander = []
    all_probs_free_ander = []
    all_probs_noise_aer = []
    all_probs_free_aer = []
    max_evaluations = 0  # Reset counts for each repetition


    # Map the Hamiltonian
    qubit_op = mapper.map(es_problem.second_q_ops()[0])  # Map second quantized Hamiltonian
    n_qubits = qubit_op.num_qubits

    # Perform VQE multiple times
    for run in range(n_repeats):

        def store_intermediate_result(eval_count, parameters, mean, std):
            counts.append(eval_count)
            values.append(mean)
            
        counts = []  # Reset counts for each repetition
        values = []  # Reset values for each repetition  
        ground_param = [] #Reset parameters for each repetition
        
        # Set VQE solver with the callback function
        vqe_solver = VQE(estimator, ansatz, SLSQP(maxiter=maxiter), callback=store_intermediate_result)
        vqe_solver.initial_point = np.random.rand(ansatz.num_parameters)

        # Compute the ground state energy
        result = vqe_solver.compute_minimum_eigenvalue(qubit_op)
      
        vqe_result = vqe_solver._build_vqe_result(
        ansatz = vqe_solver.ansatz,
        optimizer_result = result.optimizer_result,
        aux_operators_evaluated = result.aux_operators_evaluated,
        optimizer_time = result.optimizer_time
        )

        optimal_circuit = vqe_result.optimal_circuit
        ansatz_circuit = optimal_circuit
        ansatz_circuit = ansatz_circuit.decompose()  
        ansatz_circuit.measure_all()

        t_circ = transpile(
            ansatz_circuit,
            backend_fake,
            initial_layout=linear_qubit_layout,
            seed_transpiler=10,
            scheduling_method='asap'
        )

        t_circ = t_circ.assign_parameters(vqe_result.optimal_parameters)


        probs_noise_ander = sim_noise_ander.run(
            t_qiskit_circ=t_circ, 
            qubits_layout=linear_qubit_layout, 
            psi0=np.array(psi0), 
            shots=shots, 
            device_param=device_param_lookup,
            nqubit=n_qubit) 
        probs_noise_ander = fix_counts(probs_noise_ander, n_classic_bit)
            
        probs_free_ander = sim_free_ander.run(
            t_qiskit_circ=t_circ, 
            qubits_layout=linear_qubit_layout, 
            psi0=np.array(psi0), 
            shots=shots, 
            device_param=device_param_lookup,
            nqubit=n_qubit) 
        probs_free_ander = fix_counts(probs_free_ander, n_classic_bit)

        if is_tUPS:
            
            probs_noise_aer = {'state': 0} 

            print("Can't perform noisy Qiskit Aer simulation with tUPS ansatz")
            
        else:    
           
            probs_noise_aer = sim_noise_aer.run(t_circ, shots=10000).result()

        probs_free_aer = sim_free_aer.run(t_circ, shots=10000).result()
      
        # Store the results of this run and check maximum number of evaluations
        all_counts.append(counts)
        all_values.append(values)
        all_probs_noise_ander.append(probs_noise_ander)
        all_probs_free_ander.append(probs_free_ander)
        all_probs_noise_aer.append(probs_noise_aer)
        all_probs_free_aer.append(probs_free_aer)
        max_evaluations = max(max_evaluations, len(values))  

    average_energy = np.zeros(max_evaluations)

    for values in all_values:
        padded_values = np.pad(values, (0, max_evaluations - len(values)), constant_values=values[-1])
        average_energy += padded_values

    average_energy /= n_repeats  # Average over all runs


    average_probs_noise_ander = defaultdict(float)
    average_probs_free_ander = defaultdict(float)
    average_probs_noise_aer = defaultdict(float)
    average_probs_free_aer = defaultdict(float)


    for realization in all_probs_noise_ander:
        for state, prob in realization.items():
            average_probs_noise_ander[state] += prob

    for state in average_probs_noise_ander:
        average_probs_noise_ander[state] /= n_repeats

    average_probs_noise_ander = dict(average_probs_noise_ander)

    for realization in all_probs_free_ander:
        for state, prob in realization.items():
            average_probs_free_ander[state] += prob

    for state in average_probs_free_ander:
        average_probs_free_ander[state] /= n_repeats

    average_probs_free_ander = dict(average_probs_free_ander)
    
    if is_tUPS:
        pass
    else:    
        for realization in all_probs_noise_aer:
            counts = realization.get_counts()
            for state, prob in counts.items():
                average_probs_noise_aer[state] += prob

        for state in average_probs_noise_aer:
            average_probs_noise_aer[state] /= n_repeats

        average_probs_noise_aer = dict(average_probs_noise_aer)

    for realization in all_probs_free_aer:
        counts = realization.get_counts()
        for state, prob in counts.items():
            average_probs_free_aer[state] += prob

    for state in average_probs_free_aer:
        average_probs_free_aer[state] /= n_repeats

    average_probs_free_aer = dict(average_probs_free_aer)
    
    return average_energy, average_probs_noise_ander, average_probs_free_ander, average_probs_noise_aer, average_probs_free_aer, max_evaluations

