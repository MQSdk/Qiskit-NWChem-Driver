import numpy as np
import json
import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
import matplotlib.pyplot as plt
import pyscf
from pyscf import mcscf, ao2mo, fci

geometries = ["""H 0.0 0.0 -0.3; H 0.0 0.0 0.3;""", 
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
             """H 0.0 0.0 -3.0; H 0.0 0.0 3.0;""",]

distances = [0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]  

basis = ["aug-cc-pv5z"]
results = {base: {} for base in basis}
active_spaces = [1, 2, 4, 8, 12, 18]
unit= "angstroms"
for base in (basis):
    for i, geometry in enumerate(geometries):
       
        mol = pyscf.M(atom = geometry, basis = base, unit = unit)
        mf = mol.RHF().run()

        energy_HF = mf.e_tot
        
        fcisolver = fci.FCI(mf)
    
        energy_fci, fci_vector = fcisolver.kernel()
        norb_fci = fcisolver.norb

        distance = str(distances[i])  
        results[base][distance] = {
            "HF": energy_HF,
            "FCI": energy_fci,
            "N_orb": norb_fci
        }
        print(f"Base: {base}, Distance: {distance} Å, HF: {energy_HF:.8f}, FCI: {energy_fci:.8f}, N_orb: {norb_fci:.8f}")
        active_spaces = [1, 2, 4, 8, 12, 18]

        for ncas in active_spaces:
            if ncas <= norb_fci:
                mycas = mf.CASSCF(ncas, 2).run()

                energy_casscf = mycas.e_tot

                results[base][distance][f"CASSCF_{ncas}"] = energy_casscf
            
                print(f"Base: {base}, Distance: {distance} Å, CASSCF({2},{ncas}): {energy_casscf:.8f}")  


output_file = "energies_h2_atomic_5.json"
try:
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Resultados guardados exitosamente en {output_file}")
except Exception as e:
    print(f"Error al guardar el archivo JSON: {e}")