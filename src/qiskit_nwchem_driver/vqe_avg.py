import numpy as np
import pylab
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers.qubit_mapper import QubitMapper
from qiskit.primitives import Estimator
from collections import defaultdict



def run_vqe_with_averaging(ansatz, estimator, mapper, es_problem, n_repeats):
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
    all_param = []   # List to store optimized parameters for each run
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
        vqe_solver = VQE(estimator, ansatz, SLSQP(), callback=store_intermediate_result)
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
        
        ground_param = vqe_result.optimal_parameters

        # Store the results of this run and check maximum number of evaluations
        all_counts.append(counts)
        all_values.append(values)
        all_param.append(ground_param)
        max_evaluations = max(max_evaluations, len(values))  

    average_values = np.zeros(max_evaluations)
    #average_param = np.zeros(len(ground_param))
    average_param = defaultdict(float)

    for values in all_values:
        padded_values = np.pad(values, (0, max_evaluations - len(values)), constant_values=values[-1])
        average_values += padded_values

    average_values /= n_repeats  # Average over all runs

    for param_dict in all_param:
        for key, value in param_dict.items():
            average_param[key] += value

    for key in average_param:
        average_param[key] /= n_repeats

    average_param = dict(average_param)
    
    return average_values, max_evaluations, vqe_solver, result, average_param, optimal_circuit

