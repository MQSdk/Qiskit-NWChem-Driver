import numpy as np
import json
import slowquant.SlowQuant as sq
from slowquant.unitary_coupled_cluster.ucc_wavefunction import WaveFunctionUCC
import pyscf
from pyscf import mcscf, ao2mo, gto, fci
from multiprocessing import Pool


NUM_PROCESSES = 17


geometries = [
    """Li 0.0 0.0 -0.65; H 0.0 0.0 0.65;""",
    """Li 0.0 0.0 -0.70; H 0.0 0.0 0.70;""",
    """Li 0.0 0.0 -0.75; H 0.0 0.0 0.75;""",
    """Li 0.0 0.0 -0.80; H 0.0 0.0 0.80;""",
    """Li 0.0 0.0 -0.85; H 0.0 0.0 0.85;""",
    """Li 0.0 0.0 -0.90; H 0.0 0.0 0.90;""",
    """Li 0.0 0.0 -0.95; H 0.0 0.0 0.95;""",
    """Li 0.0 0.0 -1.00; H 0.0 0.0 1.00;""",
    """Li 0.0 0.0 -1.25; H 0.0 0.0 1.25;""",
    """Li 0.0 0.0 -1.50; H 0.0 0.0 1.50;""",
    """Li 0.0 0.0 -1.75; H 0.0 0.0 1.75;""",
    """Li 0.0 0.0 -2.00; H 0.0 0.0 2.00;""",
    """Li 0.0 0.0 -2.25; H 0.0 0.0 2.25;""",
    """Li 0.0 0.0 -2.50; H 0.0 0.0 2.50;""",
    """Li 0.0 0.0 -3.00; H 0.0 0.0 3.00;""",
    """Li 0.0 0.0 -3.50; H 0.0 0.0 3.50;"""
]

basis = "cc-pvdz"
cas = (4, 10)
unit = "angstrom"

def process_geometry(args):
    idx, geometry = args
    print(f"\nProcessing geometry {idx+1}:")
    print(geometry)

    # SlowQuant
    SQobj = sq.SlowQuant()
    SQobj.set_molecule(geometry, distance_unit=unit)
    SQobj.set_basis_set(basis)

    # PySCF
    mol = pyscf.M(atom=geometry, basis=basis, unit=unit)
    mf = mol.RHF().run()

    energy_HF = mf.e_tot   

    mol_pseudo = gto.M(atom=geometry, basis=basis, unit=unit, pseudo='ccecp')
    mf_pseudo = mol_pseudo.RHF().run()

    # HF
    mo_coeffs = mf.mo_coeff
    mo_coeffs_pseudo = mf_pseudo.mo_coeff
    # Integrals in AO basis
    hcore_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    eri_4fold_ao = mol.intor('int2e_sph')

    # UCCSD
    WF = WaveFunctionUCC(
        num_elec=SQobj.molecule.number_electrons,
        cas=cas,
        mo_coeffs=mo_coeffs,
        h_ao=hcore_ao,
        g_ao=eri_4fold_ao,
        excitations="SD",
        include_active_kappa=True,
    )

    WF.run_wf_optimization_1step(
        optimizer_name="SLSQP",
        tol=1e-5,
        orbital_optimization=False
    )

    energy_uccsd = WF.energy_elec + mol.energy_nuc()
  
    fcisolver = fci.FCI(mf) 
    pseudosolver = fci.FCI(mf_pseudo) 

    energy_fci, fci_vector = fcisolver.kernel()
    energy_pseudo, pseudo_vector = pseudosolver.kernel()

    norb_fci = fcisolver.norb

    print(f"Number of orbitals in FCI: {norb_fci}")
    print(f"Hartree-Fock energy for geometry {idx+1} = {energy_HF}")
    print(f"UCCSD energy for geometry {idx+1} = {energy_uccsd}")
    print(f"FCI energy for geometry {idx+1} = {energy_fci}")
    print(f"FCI energy with pseudopotential for geometry {idx+1} = {energy_pseudo}")
    return {
        'geometry': geometry,
        'energy Hartree-Fock': energy_HF,
        'energy uccsd': energy_uccsd,
	'energy fci': energy_fci,
        'energy pseudo fci': energy_pseudo,
        'nuclear_repulsion': mol.energy_nuc(),
        'electronic_energy': WF.energy_elec
    }

if __name__ == "__main__":
    # Prepare arguments for parallel processing
    args = [(i, geom) for i, geom in enumerate(geometries)]

    # Use all available CPU cores
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(process_geometry, args)

    # Save results to JSON
    with open('results_pvdz_LiH.json', 'w') as f:
        json.dump(results, f, indent=4)
