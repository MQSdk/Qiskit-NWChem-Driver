import os
from qiskit_nature_qe import nwchem_driver
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit.exceptions import QiskitError

if __name__ == "__main__":
    # hdf5_up_file = os.path.join(
    #     "qe_files", "out_H2_spin", "H2_spin.save", "wfcup1.hdf5"
    # )
    # hdf5_dw_file = os.path.join(
    #     "qe_files", "out_H2_spin", "H2_spin.save", "wfcdw1.hdf5"
    # )
    # xml_file = os.path.join(
    #     "qe_files", "out_H2_spin", "H2_spin.save", "data-file-schema.xml"
    # )
    nwchem_output_file = os.path.join(
        "qe_files", "LiH_YAML", "H1Li1-7.0.yaml"
    )
    driver = nwchem_driver.NWchem_Driver(nwchem_output_file)

    problem = driver.run()

    # ------------------ Solve with NumPyMinimumEigensolver ------------------
    mapper = JordanWignerMapper()
    algo = NumPyMinimumEigensolver()
    algo.filter_criterion = problem.get_default_filter_criterion()

    solver = GroundStateEigensolver(mapper, algo)
    # FIXME: Get numpy ground state solver running
    try:
        
        result = solver.solve(problem)
        print(result)
    except QiskitError as e:
        print("Could not find ground-state!")
