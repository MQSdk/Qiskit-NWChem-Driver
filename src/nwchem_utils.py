import numpy as np
import yaml
from yaml.loader import SafeLoader

def get_spatial_integrals(one_electron, two_electron, n_orb):
    one_electron_spatial_integrals = np.zeros((n_orb, n_orb))
    two_electron_spatial_integrals = np.zeros((n_orb, n_orb, n_orb, n_orb))

    for ind, val in enumerate(one_electron):
        # This is because python index starts at 0
        i = int(val[0] - 1)
        j = int(val[1] - 1)
        one_electron_spatial_integrals[i, j] = val[2]
        if i != j:
            one_electron_spatial_integrals[j, i] = val[2]

    for ind, val in enumerate(two_electron):
        i = int(val[0] - 1)
        j = int(val[1] - 1)
        k = int(val[2] - 1)
        l = int(val[3] - 1)
        two_electron_spatial_integrals[i, j, k, l] = val[4]
        if two_electron_spatial_integrals[k, l, i, j] == 0:  # klij
            two_electron_spatial_integrals[k, l, i, j] = val[4]
        if two_electron_spatial_integrals[i, j, l, k] == 0:  # ijlk
            two_electron_spatial_integrals[i, j, l, k] = val[4]
        if two_electron_spatial_integrals[l, k, i, j] == 0:  # lkij
            two_electron_spatial_integrals[l, k, i, j] = val[4]
        if two_electron_spatial_integrals[j, i, k, l] == 0:  # jikl
            two_electron_spatial_integrals[j, i, k, l] = val[4]
        if two_electron_spatial_integrals[k, l, j, i] == 0:  # klji
            two_electron_spatial_integrals[k, l, j, i] = val[4]
        if two_electron_spatial_integrals[j, i, l, k] == 0:  # jilk
            two_electron_spatial_integrals[j, i, l, k] = val[4]
        if two_electron_spatial_integrals[l, k, j, i] == 0:  # lkji
            two_electron_spatial_integrals[l, k, j, i] = val[4]

    return one_electron_spatial_integrals, two_electron_spatial_integrals

def convert_to_spin_index(one_electron, two_electron, n_orb):
    h1 = np.block([[one_electron, np.zeros((int(n_orb), int(n_orb)))],
                  [np.zeros((int(n_orb), int(n_orb))), one_electron]])
    h2 = np.zeros((2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb))

    for i in range(len(two_electron)):
        for j in range(len(two_electron)):
            for k in range(len(two_electron)):
                for l in range(len(two_electron)):
                    h2[i, j, k + n_orb, l + n_orb] = two_electron[i, j, k, l]
                    h2[i + n_orb, j + n_orb, k, l] = two_electron[i, j, k, l]

                    if i != k and j != l:  # Pauli exclusion principle
                        h2[i, j, k, l] = two_electron[i, j, k, l]
                        h2[i + n_orb, j + n_orb, k + n_orb, l + n_orb] = two_electron[i, j, k, l]
    return h1, 0.5 * h2

def load_from_yaml(file_name, include_spin=True):
    data = yaml.load(open(file_name, "r"), SafeLoader)
    n_electrons = data['integral_sets'][0]['n_electrons']
    n_spatial_orbitals = data['integral_sets'][0]['n_orbitals']
    nuclear_repulsion_energy = data['integral_sets'][0]['coulomb_repulsion']['value']
    one_electron_import = data['integral_sets'][0]['hamiltonian']['one_electron_integrals']['values']
    two_electron_import = data['integral_sets'][0]['hamiltonian']['two_electron_integrals']['values']

    one_electron_spatial_integrals, two_electron_spatial_integrals = get_spatial_integrals(
        one_electron_import, two_electron_import, n_spatial_orbitals
    )
    if include_spin:
        h1, h2 = convert_to_spin_index(
            one_electron_spatial_integrals, two_electron_spatial_integrals, n_spatial_orbitals
        )
    else:
        h1 = one_electron_spatial_integrals
        h2 = two_electron_spatial_integrals

    return n_electrons, n_spatial_orbitals, nuclear_repulsion_energy, h1, h2
