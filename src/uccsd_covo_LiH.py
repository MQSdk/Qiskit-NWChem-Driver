import os
import numpy as np
import yaml
from pyscf import fci
from joblib import Parallel, delayed
import json
import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
from multiprocessing import Pool
from nwchem_utils import load_from_yaml

# Definir el solver FCI globalmente
fcisolver = fci.direct_spin1.FCI()
NUM_PROCESSES = 17
# Función que procesa un solo archivo YAML
def process_yaml_file(yaml_data):
    covo, bond_distance, n_electrons, n_spatial_orbitals, nuclear_repulsion_energy, h1, h2 = yaml_data
    print(f"Procesando COVO: {covo}, Bond distance: {bond_distance}")
    

    # Inicializar WaveFunctionUCC
    WF = WaveFunctionUCC(
        num_elec=n_electrons,
        cas=(2, n_spatial_orbitals),
        mo_coeffs=np.identity(n_spatial_orbitals),
        h_ao=h1,
        g_ao=h2,
        excitations="SD",
        include_active_kappa=True,
    )

    # Optimizar la función de onda
    WF.run_wf_optimization_1step(
        optimizer_name="SLSQP",
        tol=1e-7,
        orbital_optimization=False
    )

    # Calcular energía FCI
    energy_fci, ci_vec = fcisolver.kernel(h1e=h1, eri=h2, norb=n_spatial_orbitals, nelec=n_electrons)      

    energy_fci = energy_fci + nuclear_repulsion_energy
    energy_uccsd = WF.energy_elec + nuclear_repulsion_energy
    
    return covo, bond_distance, energy_uccsd, energy_fci

# Lista de COVOs
covos = [1,4,8,12]

# Recolectar todos los archivos YAML
yaml_data = []
for covo in covos:
    data_dir_yaml = os.path.join("..", "data", "PW_LiH_data", "3x3_aperiodic",'NWChem', f'{covo}covo_yaml')
    if not os.path.exists(data_dir_yaml):
        print(f"Directorio no encontrado: {data_dir_yaml}")
        continue
    data_files = os.listdir(data_dir_yaml)
    for data_file in data_files:
        if not data_file.endswith('.yaml'):
            continue
        temp = data_file.split('-')
        temp1 = temp[1].split('.')
        bond_distance = float(temp1[0] + '.' + temp1[1])
        yaml_file = os.path.join(data_dir_yaml, data_file)
        n_electrons, n_spatial_orbitals, nuclear_repulsion_energy, h1, h2 = load_from_yaml(yaml_file, include_spin=False)
        yaml_data.append((covo, bond_distance, n_electrons, n_spatial_orbitals, nuclear_repulsion_energy, h1, h2))

# Paralelizar el procesamiento de archivos YAML
if __name__ == '__main__':
    data_covos = {}
 #   results = Parallel(n_jobs=NUM_PROCESSES)(delayed(process_yaml_file)(data) for data in yaml_data)
    with Pool(processes = NUM_PROCESSES) as pool:
        results = pool.map(process_yaml_file, yaml_data)

    # Agrupar resultados por covo
    results_by_covo = {}
    for covo, bond_distance, energy_uccsd, energy_fci in results:
        if covo not in results_by_covo:
            results_by_covo[covo] = {'bond_distances': [], 'uccsd': [], 'fci': []}
        results_by_covo[covo]['bond_distances'].append(bond_distance)
        results_by_covo[covo]['uccsd'].append(energy_uccsd)
        results_by_covo[covo]['fci'].append(energy_fci)

    # Ordenar y estructurar los resultados
    for covo in results_by_covo:
        bond_distances = np.array(results_by_covo[covo]['bond_distances'])
        total_energies_uccsd = np.array(results_by_covo[covo]['uccsd'])
        total_energies_fci = np.array(results_by_covo[covo]['fci'])

        # Ordenar por bond_distance
        sorted_indices = np.argsort(bond_distances)
        bond_distances = bond_distances[sorted_indices]
        total_energies_uccsd = total_energies_uccsd[sorted_indices]
        total_energies_fci = total_energies_fci[sorted_indices]

        data_covos[covo] = [bond_distances.tolist(), total_energies_uccsd.tolist(), total_energies_fci.tolist()]

    # Guardar resultados
    with open('results_covo_LiH.json', 'w') as f:
        json.dump(data_covos, f, indent=4)
