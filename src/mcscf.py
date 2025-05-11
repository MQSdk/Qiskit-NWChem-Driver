import numpy as np
import json
import pyscf
from pyscf import gto, scf, mcscf, fci
from multiprocessing import Pool
import os

# Definir geometrías, distancias, base y espacios activos
geometries = [
    """H 0.0 0.0 -0.3; H 0.0 0.0 0.3;""",
    """H 0.0 0.0 -0.35; H 0.0 0.0 0.35;""",
    """H 0.0 0.0 -0.4; H 0.0 0.0 0.4;""",
    """H 0.0 0.0 -0.45; H 0.0 0.0 0.45;""",
    """H 0.0 0.0 -0.5; H 0.0 0.0 0.5;""",
    """H 0.0 0.0 -0.75; H 0.0 0.0 0.75;""",
    """H 0.0 0.0 -1.0; H 0.0 0.0 1.0;""",
    """H 0.0 0.0 -1.25; H 0.0 0.0 1.25;""",
    """H 0.0 0.0 -1.5; H 0.0 0.0 1.5;""",
    """H 0.0 0.0 -1.75; H 0.0 0.0 1.75;""",
    """H 0.0 0.0 -2.0; H 0.0 0.0 2.0;""",
    """H 0.0 0.0 -2.25; H 0.0 0.0 2.25;""",
    """H 0.0 0.0 -2.5; H 0.0 0.0 2.5;""",
    """H 0.0 0.0 -3.0; H 0.0 0.0 3.0;""",
]

distances = [0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
basis = ["cc-pvdz","cc-pvtz","cc-pvqz","cc-pv5z","aug-cc-pvqz","aug-cc-pv5z"]
active_spaces = [2, 3, 5, 9, 13, 19]
unit = "angstrom"
results = {base: {str(d): {} for d in distances} for base in basis}

# Función para procesar una geometría
def process_geometry(args):
    i, geometry, base, distance = args
    print(f"Procesando geometría {i+1}/{len(geometries)} para base {base}, Distance {distance} Å, PID: {os.getpid()}")
 
    # Definir la molécula
    mol = gto.M(atom=geometry, basis=base, unit=unit)
    mol.build()

    # Cálculo Hartree-Fock
    mf = scf.RHF(mol).run()
    energy_HF = mf.e_tot

    # Cálculo FCI
    fcisolver = fci.FCI(mf)
    energy_fci, fci_vector = fcisolver.kernel()
    norb_fci = fcisolver.norb

    # Resultados iniciales
    result = {
        "HF": energy_HF,
        "FCI": energy_fci,
        "N_orb": norb_fci
    }
    print(f"Base: {base}, Distance: {distance} Å, HF: {energy_HF:.8f}, FCI: {energy_fci:.8f}, N_orb: {norb_fci:.8f}")

    # Cálculo CASSCF
    for ncas in active_spaces:
        if ncas <= norb_fci:
            try:
                mycas = mf.CASSCF(ncas, 2)
		
                mycas.frozen = [0]
		
                mycas.run()
                
                energy_casscf = mycas.e_tot
                
                result[f"CASSCF_{ncas}"] = energy_casscf
                
                print(f"Base: {base}, Distance: {distance} Å, CASSCF(2,{ncas}): {energy_casscf:.8f}")
            except Exception as e:
                print(f"Error en CASSCF(2,{ncas}) para Distance {distance}: {e}")
                result[f"CASSCF_{ncas}"] = None

    return (base, distance, result)

# Paralelizar el procesamiento de geometrías
if __name__ == '__main__':
    # Preparar argumentos para cada geometría
    for base in basis:
        print(f"Procesando base: {base}")

        tasks = [(i, geometry, base, str(distances[i])) for i, geometry in enumerate(geometries)]

        # Usar Pool para paralelizar
        with Pool() as pool:
            # Mapear las tareas a los procesadores
            results_list = pool.map(process_geometry, tasks)

        # Combinar resultados
        for base_result, distance, result in results_list:
            results[base_result][str(distance)] = result

    # Guardar resultados en JSON
    output_file = "energies_h2_atomic_frozen.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Resultados guardados exitosamente en {output_file}")
    except Exception as e:
        print(f"Error al guardar el archivo JSON: {e}")

