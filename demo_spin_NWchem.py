import os 
import sys
import yaml
import os
from qiskit_nature_qe import nwchem_driver
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit.exceptions import QiskitError
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD, UCC
import numpy as np

preamble="""
"$schema": https://raw.githubusercontent.com/Microsoft/Quantum/master/Chemistry/Schema/broombridge-0.1.schema.json
"""

def is_integer(str):
    try:
        int(str)
        return True
    except ValueError:
        return False

def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False
def extract_fields(file):
    data = {} #yaml.load(initial_input)
    data['format'] = {'version' : '0.1'}
    data['bibliography'] = [{'url' : 'https://nwchemgit.github.io'}]
    data['generator'] = {'source' : 'nwchem',
                         'version' : '7.0.2'}
    skip_input_geometry = False
    geometry = None
    coulomb_repulsion = None
    scf_energy = None
    scf_energy_offset = None
    energy_offset = None
    one_electron_integrals = None
    two_electron_integrals = None
    n_electrons_alpha = None
    basis_set = None
    n_electrons_beta = None
    n_orbitals = None
    initial_state = None
    ccsd_energy = None
    reader_mode = ""
    excited_state_count = 1
    excitation_energy = 0.0
    fci_energy = None
    f = open(file)
    for line in f.readlines():
        ln = line.strip()
        ln_segments = ln.split()
        if len(ln) == 0 or ln[0]=="#": #blank or comment line
            continue
        if reader_mode=="":
            if ln == "============================== echo of input deck ==============================":
                reader_mode = "input_deck"
            elif ln_segments[:2] == ["enrep_tce", "="]:
                coulomb_repulsion = {
                    'units' : 'hartree',
                    'value' : float(ln_segments[2])
                }
            elif ln_segments[:2] == ["EHF(total)", "="]:
                scf_energy = {
                    'units' : 'hartree',
                    'value' : float(ln_segments[2])
                }
            elif ln_segments[:3] == ["Shift", "(HFtot-HFA)", "="]:
                scf_energy_offset = {
                    'units' : 'hartree',
                    'value' : float(ln_segments[3])
                }
                energy_offset = {
                    'units' : 'hartree',
                    'value' : float(ln_segments[3])
                }
            elif ln == "begin_one_electron_integrals":
                reader_mode = "one_electron_integrals"
                one_electron_integrals = {
                    'units' : 'hartree',
                    'format' : 'sparse',
                    'values' : []
                }
            elif ln == "begin_two_electron_integrals":
                reader_mode = "two_electron_integrals"
                two_electron_integrals = {
                    'units' : 'hartree',
                    'format' : 'sparse',
                    'index_convention' : 'mulliken',
                    'values' : []
                }
            elif ln.find("number of electrons") != -1 and ln.find("Fourier space") != -1:
                try:
                    n_electrons = int(ln_segments[5]) + int(ln_segments[11])
                except:
                    continue
            elif ln.find("number of orbitals") != -1 and ln.find("Fourier space") != -1:
                n_orbitals = int(ln_segments[6]) + int(ln_segments[12])
            elif ln.find("total     energy") != -1:
                total_energy = float(ln_segments[3])
            elif ln.find("Number of active alpha electrons") != -1:
                n_electrons_alpha = int(ln_segments[-1])
            elif ln.find("Number of active beta electrons") != -1:
                n_electrons_beta = int(ln_segments[-1])
            elif ln.find("Number of active orbitals") != -1:
                n_orbitals = int(ln_segments[-1])
            elif ln.find("ion-ion   energy") != -1:
                coulomb_repulsion = {
                    'units' : 'hartree',
                    'value' : float(ln_segments[3])
                }
            elif ln.find("Total MCSCF energy") != -1:
                fci_val = float(ln.split("=")[1].strip())
                fci_energy = {"units" : "hartree", "value" : fci_val, "upper": fci_val+0.1, "lower":fci_val-0.1}
            #elif ln.split("=")[0].strip() == "CCSD total energy / hartree":
            #    if fci_energy is None:
            #        fci_val = float(ln.split("=")[1].strip())
            #        fci_energy = {"units" : "hartree", "value" : fci_val, "upper": fci_val*0.99, "lower":fci_val*1.01}
            # elif ln == "Ground state specification:":
            #     reader_mode = "initial_state"
            #     assert ccsd_energy is not None, "CCSD energy should be available before the groud state specification."
            #     if initial_state is None:
            #         initial_state = []
            #     initial_state += [ {'state':{
            #             'label' : '|G>',
            #             'energy' : { 'units' : 'hartree', 
            #                          'value' : ccsd_energy} ,
            #             'superposition' : []
            #             }}]
            # elif ln == "Excited state specification:":
            #     reader_mode = "initial_state"
            #     assert ccsd_energy is not None, "CCSD energy should be available before the excited state specification."
            #     if initial_state is None:
            #         initial_state = []
            #     initial_state += [ {'state':{
            #             'label' : '|E%d>'%(excited_state_count),
            #             'energy' : { 'units' : 'hartree', 
            #                          'value' : ccsd_energy+excitation_energy} ,
            #             'superposition' : []
            #             }}]
            #     excited_state_count += 1
            # elif ln.split("=")[0].strip() == "Excitation energy / hartree":
            #     excitation_energy = float(ln.split("=")[1].strip())
            # elif ln.split("=")[0].strip() == "CCSD total energy / hartree":
            #     ccsd_energy = float(ln.split("=")[1].strip())
            elif ln == '''Geometry "geometry" -> ""''':
                reader_mode = "cartesian_geometry"
                if geometry is None:
                    geometry = {'coordinate_system': 'cartesian'}
                geometry['atoms'] = []
        elif reader_mode == "cartesian_geometry":
            if ln_segments[0] == "Output":
                if 'angstroms' in ln_segments:
                    geometry['units'] = 'angstrom'
                elif 'a.u.' in ln_segments:
                    geometry['units'] = 'bohr'
            elif ln_segments[0] == 'Atomic':
                reader_mode = ""
            elif is_integer(ln_segments[0]):
                assert 'atoms' in geometry
                #if not 'atoms' in geometry:
                #    geometry['atoms'] = []
                geometry['atoms'] += [{"name":ln_segments[1],
                                       "coords":
                                       [float(ln_segments[3]), float(ln_segments[4]), float(ln_segments[5])]}]
        elif reader_mode == "initial_state":
            if ln == "-------------------------------------":
                reader_mode = ""
            elif 'norm' in ln or 'string' in ln or 'exp' in ln:
                continue
            else:
                #vals = ln.split(":")
                if len(initial_state[-1]['state']['superposition']) > 0 and initial_state[-1]['state']['superposition'][-1][-1] != '|vacuum>':
                    initial_state[-1]['state']['superposition'][-1] += ln.replace('|0>', '|vacuum>').split()
                else:
                    vals = ln.replace('|0>', '|vacuum>').split(":")
                    initial_state[-1]['state']['superposition'] += [
                        [float(vals[0].strip())] +
                        vals[1].split()[:]
                    ]
                    '''initial_state[-1]['state']['superposition'] += [
                        [float(vals[0].strip())] +
                        vals[1].split()[:-1] +
                        ['|vacuum>']
                    ]'''
        elif reader_mode == "input_deck":
            if ln == "================================================================================":
                reader_mode = ""
            elif ln_segments[0:2]== ["geometry", "units"]:
                reader_mode = "input_geometry"
                assert len(ln_segments) >= 3
                geometry = {'coordinate_system': 'cartesian'}
                geometry['units'] = ln_segments[2]
            # elif ln_segments[0].lower() == "basis":
            #     reader_mode = "input_basis"
            #     basis_set = {}
            #     basis_set['name'] = 'unknown'
            #     basis_set['type'] = 'gaussian'
        elif reader_mode=="input_geometry":
            if ln.lower()=="end":
                reader_mode = "input_deck"
            elif ln_segments[0] == "symmetry":
                assert len(ln_segments) == 2
                geometry['symmetry'] = ln_segments[1]
            elif skip_input_geometry == False: #atom description line
                if len(ln_segments) != 4 or not is_float(ln_segments[1]) or not is_float(ln_segments[2]) or not is_float(ln_segments[3]):
                    skip_input_geometry = True
                else:
                    if not 'atoms' in geometry:
                        geometry['atoms'] = []
                        geometry['atoms'] += [{"name":ln_segments[0],
                                               "coords":
                                                   [float(ln_segments[1]), float(ln_segments[2]), float(ln_segments[3])]}]
        elif reader_mode == "input_basis":
            if ln.lower() == "end":
                reader_mode = "input_deck"
            else:
                if ln.find('library') != -1:
                    assert len(ln_segments) == 3
                    assert ln_segments[1] == "library"
                    basis_set['name'] = ln_segments[2]
                    basis_set['type'] = 'gaussian'
        elif reader_mode == "one_electron_integrals":
            if ln == "end_one_electron_integrals":
                reader_mode = ""
            else:
                assert len(ln_segments) == 3
                if int(ln_segments[0]) >= int(ln_segments[1]):
                    one_electron_integrals['values'] += [[
                            int(ln_segments[0]),
                            int(ln_segments[1]),
                            float(ln_segments[2])
                            ]]
        elif reader_mode == "two_electron_integrals":
            if ln == "end_two_electron_integrals":
                reader_mode = ""
            else:
                assert len(ln_segments) == 5
                two_electron_integrals['values'] += [[
                     int(ln_segments[0]),
                      int(ln_segments[1]),
                      int(ln_segments[2]),
                      int(ln_segments[3]),
                      float(ln_segments[4])
                    ]]

    # if fci_energy is None:
    #     assert ccsd_energy is not None
    #     fci_energy = {"units" : "hartree", "value" : ccsd_energy, "upper": ccsd_energy+0.1, "lower": ccsd_energy-0.1}

    assert one_electron_integrals is not None, "one_electron_integrals is missing from NWChem output. Required to extract YAML"
    assert two_electron_integrals is not None, "two_electron_integrals is missing from NWChem output. Required to extract YAML"
    assert geometry is not None, "geometry information is missing from NWChem output. Required to extract YAML"
    # assert basis_set  is not None, "basis_set is missing from NWChem output. Required to extract YAML"
    # assert coulomb_repulsion  is not None, "coulomb_repulsion is missing from NWChem output. Required to extract YAML"
    # assert scf_energy  is not None, "scf_energy is missing from NWChem output. Required to extract YAML"
    # assert scf_energy_offset  is not None, "scf_energy_offset is missing from NWChem output. Required to extract YAML"
    # assert energy_offset  is not None, "energy_offset is missing from NWChem output. Required to extract YAML"
    # assert fci_energy  is not None, "fci_energy is missing from NWChem output. Required to extract YAML"
    assert n_orbitals  is not None, "n_orbitals is missing from NWChem output. Required to extract YAML"
    # assert n_electrons_alpha  is not None, "n_electrons_alpha is missing from NWChem output. Required to extract YAML"
    # assert n_electrons_beta  is not None, "n_electrons_beta is missing from NWChem output. Required to extract YAML"
    hamiltonian = {'one_electron_integrals' : one_electron_integrals,
                   'two_electron_integrals' : two_electron_integrals}
    integral_sets =  [{"metadata": { 'molecule_name' : 'unknown'},
                       "geometry":geometry,
                    #    "basis_set":basis_set,
                        "total_energy":total_energy,
                       "coulomb_repulsion" : coulomb_repulsion,
                    #    "scf_energy" : scf_energy,
                    #    "scf_energy_offset" : scf_energy_offset,
                    #    "energy_offset" : energy_offset,
                    #    "fci_energy" : fci_energy,
                       "hamiltonian" : hamiltonian,
                       "n_orbitals" : n_orbitals,
                       "n_electrons" : n_electrons }]
    if initial_state is not None:
        integral_sets[-1]["initial_state_suggestions"] = initial_state
    data['integral_sets'] = integral_sets
    f.close()
    return data

def main():
    data = extract_fields('qe_files/n2/output/demo.out')
    with open('qe_files/n2/output/demo.yaml', 'w') as f:
        f.write(yaml.dump(data, default_flow_style=False)) 
    
    driver = nwchem_driver.NWchem_Driver('qe_files/n2/output/demo.yaml')
    es_problem = driver.run()
    hamiltonian  = es_problem.hamiltonian
    print(hamiltonian.second_q_op())
    # # ------------------ Solve with NumPyMinimumEigensolver ------------------
    mapper = JordanWignerMapper()
    # algo = NumPyMinimumEigensolver()
    # algo.filter_criterion = problem.get_default_filter_criterion()

    # solver = GroundStateEigensolver(mapper, algo)
    # # FIXME: Get numpy ground state solver running
    # try:
        
    #     result = solver.solve(problem)
    #     print(result)
    # except QiskitError as e:
    #     print("Could not find ground-state!")
    
    ansatz = UCCSD(
        es_problem.num_spatial_orbitals,
        es_problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            es_problem.num_spatial_orbitals,
            es_problem.num_particles,
            mapper,
        ),
    )

    vqe_solver = VQE(Estimator(), ansatz, SLSQP())
    
    # vqe_solver.initial_point = [0.0] * ansatz.num_parameters
    vqe_solver.initial_point = np.random.rand(ansatz.num_parameters)
    calc = GroundStateEigensolver(mapper, vqe_solver)
    res = calc.solve(es_problem)
    print(res)
if __name__ == "__main__":
    main()