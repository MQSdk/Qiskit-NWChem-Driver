import sys
import math
import yaml
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
    virtual_orb = None
    # scf_energy = None
    # scf_energy_offset = None
    # energy_offset = None
    one_electron_integrals = None
    two_electron_integrals = None
    # n_electrons_alpha = None
    basis_set = None
    # n_electrons_beta = None
    n_orbitals = None
    initial_state = None
    # ccsd_energy = None
    reader_mode = ""
    # excited_state_count = 1
    # excitation_energy = 0.0
    # fci_energy = None
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
                    n_orbitals = math.ceil(int(virtual_orb) + n_electrons*0.5)
                except:
                    continue
            #elif ln.find("number of orbitals") != -1 and ln.find("Fourier space") != -1:
                #n_orbitals = int(ln_segments[6]) + int(ln_segments[12])
            elif ln.find("total     energy") != -1:
                total_energy = float(ln_segments[3])
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
            if ln_segments[0] == 'virtual' and ln_segments[1] != 'orbital':
                virtual_orb = int(ln_segments[1])
            elif ln == "================================================================================":
                reader_mode = ""
            elif ln_segments[0:2]== ["geometry", "units"]:
                reader_mode = "input_geometry"
                assert len(ln_segments) >= 3
                geometry = {'coordinate_system': 'cartesian'}
                geometry['units'] = ln_segments[2]
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
    assert one_electron_integrals is not None, "one_electron_integrals is missing from NWChem output. Required to extract YAML"
    assert two_electron_integrals is not None, "two_electron_integrals is missing from NWChem output. Required to extract YAML"
    assert geometry is not None, "geometry information is missing from NWChem output. Required to extract YAML"
    assert n_orbitals  is not None, "n_orbitals is missing from NWChem output. Required to extract YAML"
    hamiltonian = {'one_electron_integrals' : one_electron_integrals,
                   'two_electron_integrals' : two_electron_integrals}
    integral_sets =  [{"metadata": { 'molecule_name' : 'unknown'},
                       "geometry":geometry,
                       "total_energy":total_energy,
                       "coulomb_repulsion" : coulomb_repulsion,
                       "hamiltonian" : hamiltonian,
                       "virtual_orbitals" : virtual_orb,
                       "n_orbitals" : n_orbitals,
                       "n_electrons" : n_electrons }]
    if initial_state is not None:
        integral_sets[-1]["initial_state_suggestions"] = initial_state
    data['integral_sets'] = integral_sets
    f.close()
    return data


if __name__ == "__main__":

    out_file_path = sys.argv[1]
    yaml_file_path = sys.argv[2]
    
    data = extract_fields(out_file_path)
    with open(yaml_file_path, 'w') as f:
        f.write(yaml.dump(data, default_flow_style=False))
