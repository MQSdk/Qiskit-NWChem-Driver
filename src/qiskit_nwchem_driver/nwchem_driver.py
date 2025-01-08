from warnings import warn
import xmltodict
import numpy as np
import pyscf
import pyscf.fci
from qiskit_nature.second_q.drivers import ElectronicStructureDriver
from qiskit_nature.second_q.drivers.electronic_structure_driver import _QCSchemaData
from qiskit_nature.second_q.formats.qcschema import QCSchema
from qiskit_nature.second_q.formats.qcschema_translator import qcschema_to_problem
from qiskit_nature.second_q.problems import ElectronicBasis, ElectronicStructureProblem
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.operators.symmetric_two_body import S1Integrals
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.operators.tensor_ordering import to_physicist_ordering, to_chemist_ordering
from . import calc_matrix_elements
from . import eri_pair_densities
from . import wfc
import yaml
from yaml import SafeLoader

class NWchem_Driver(ElectronicStructureDriver):
    def __init__(self, nwchem_output: str) -> None:
        """NWchem driver class

        Args:
            nwchem_output (str): NWchem output file
        """
        super().__init__()
        self.nwchem_output = nwchem_output
        self.total_energy = None

    def run(self=True) -> ElectronicStructureProblem:
        return self.to_problem()

    def to_problem(
        self,
        basis: ElectronicBasis = ElectronicBasis.MO,
        include_dipole: bool = False,
    ) -> ElectronicStructureProblem:
        problem = self.to_qiskit_problem_old()
        return problem

    # this function creates the matrix (tensor) with the values of the hamiltonian we import, which are the one-electron and two-electron integrals
    def get_spatial_integrals(self, one_electron,two_electron,n_orb):
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
            i = int(val[0]-1)
            j = int(val[1]-1)
            k = int(val[2]-1)
            l = int(val[3]-1)
            two_electron_spatial_integrals[i, j, k, l] = val[4]
            if two_electron_spatial_integrals[k, l, i, j] == 0:     #klij
                two_electron_spatial_integrals[k, l, i, j] = val[4]
            if two_electron_spatial_integrals[i, j, l, k] == 0:     #ijlk
                two_electron_spatial_integrals[i, j, l, k] = val[4]
            if two_electron_spatial_integrals[l, k, i, j] == 0:     #lkij
                two_electron_spatial_integrals[l, k, i, j] = val[4]
            if two_electron_spatial_integrals[j, i, k, l] == 0:     #jikl
                two_electron_spatial_integrals[j, i, k, l] = val[4]
            if two_electron_spatial_integrals[k, l, j, i] == 0:     #klji
                two_electron_spatial_integrals[k, l, j, i] = val[4]
            if two_electron_spatial_integrals[j, i, l, k] == 0:     #jilk
                two_electron_spatial_integrals[j, i, l, k] = val[4]
            if two_electron_spatial_integrals[l, k, j, i] == 0:     #lkji
                two_electron_spatial_integrals[l, k, j, i] = val[4]

        return one_electron_spatial_integrals, two_electron_spatial_integrals
    
    # this function expands (double dimension) the previous integrals by considering spin up/down possibilities
    def convert_to_spin_index(self, one_electron, two_electron,n_orb):
        h1 = np.block([[one_electron, np.zeros((int(n_orb), int(n_orb)))],
                    [np.zeros((int(n_orb), int(n_orb))), one_electron]])
        h2 = np.zeros((2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb))

        for i in range(len(two_electron)):
            for j in range(len(two_electron)):
                for k in range(len(two_electron)):
                    for l in range(len(two_electron)):

                        h2[i,j, k + n_orb, l + n_orb] = two_electron[i, j, k, l]
                        h2[i + n_orb, j + n_orb,k, l] = two_electron[i, j, k, l]
                        # h2[i, j+n_orb, k+n_orb, l] = two_electron[i, j, k, l]
                        # h2[]

                        if i!=k and j!=l:   # Pauli exclusion priciple
                            h2[i,j,k,l] = two_electron[i,j,k,l]
                            h2[i + n_orb, j + n_orb, k + n_orb, l + n_orb] = two_electron[i, j, k, l]
        return h1, 0.5*h2

    def load_from_yaml(self,file_name):
        
        data = yaml.load(open(file_name,"r"),SafeLoader)
        n_electrons = data['integral_sets'][0]['n_electrons']
        #n_electrons = (data['integral_sets'][0]['n_electrons_up'], data['integral_sets'][0]['n_electrons_down'])
        #n_spatial_orbitals = n_electrons
        n_spatial_orbitals = data['integral_sets'][0]['n_orbitals']
        nuclear_repulsion_energy = data['integral_sets'][0]['coulomb_repulsion']['value']
        # nuclear_repulsion_energy = 0
        self.total_energy = data['integral_sets'][0]['total_energy']
        one_electron_import = data['integral_sets'][0]['hamiltonian']['one_electron_integrals']['values']
        two_electron_import = data['integral_sets'][0]['hamiltonian']['two_electron_integrals']['values']
        #n_spatial_orbitals = max(val for sublist in one_electron_import for val in sublist)    
        
        one_electron_spatial_integrals, two_electron_spatial_integrals = self.get_spatial_integrals(one_electron_import,two_electron_import,n_spatial_orbitals)
        h1, h2 = self.convert_to_spin_index(one_electron_spatial_integrals,two_electron_spatial_integrals,n_spatial_orbitals)
        # reference_energy = data['integral_sets'][0]['initial_state_suggestions'][0]['state']['energy']['value']
        symbols = [i['name'] for i in data['integral_sets'][0]['geometry']['atoms']]
        coords = []
        for i in data['integral_sets'][0]['geometry']['atoms']:
            coords = coords + i['coords']
        coords = np.array(coords)
        # return n_electrons, n_spatial_orbitals, nuclear_repulsion_energy, h1, h2, reference_energy, symbols, coords
        return n_electrons, n_spatial_orbitals, nuclear_repulsion_energy, h1, h2
        # return n_electrons, n_spatial_orbitals, nuclear_repulsion_energy, one_electron_spatial_integrals, two_electron_spatial_integrals
    def to_qcschema(self, *, include_dipole: bool = True) -> QCSchema:
        return super().to_qcschema(include_dipole=include_dipole)



    
    
    
    def to_qiskit_problem_old(self):
        num_particles, n_spatial_orbitals, nucl_repulsion, h1, h2 = self.load_from_yaml(self.nwchem_output)
        # num_particles = (
        #     1,
        #     1
        #  )   # number of electron 

        # split matrices and tensors by spin up or down
        h_ij_up = h1[:n_spatial_orbitals,:n_spatial_orbitals]
        h_ij_dw = h1[n_spatial_orbitals:, n_spatial_orbitals:]
        eri_up = h2[:n_spatial_orbitals,:n_spatial_orbitals,:n_spatial_orbitals,:n_spatial_orbitals]
        eri_dw = h2[n_spatial_orbitals:, n_spatial_orbitals:, n_spatial_orbitals:, n_spatial_orbitals:]
        eri_up_dw = h2[:n_spatial_orbitals, :n_spatial_orbitals, n_spatial_orbitals:, n_spatial_orbitals:]
        eri_dw_up = h2[n_spatial_orbitals:, n_spatial_orbitals:, :n_spatial_orbitals, :n_spatial_orbitals]
        print(f"h_ij up-down equal: {np.allclose(h_ij_dw, h_ij_up)}")
        print(f"eri up-down equal: {np.allclose(eri_dw, eri_up)}")
        print(f"eri up-(down-up) equal: {np.allclose(eri_dw_up, eri_up)}")
        print(f"eri (up-down)-(down-up) equal: {np.allclose(eri_up_dw, eri_dw_up)}")
        
        # Transform ERIs to chemist's index order #ijkl ->ikjl -> iklj
        eri_up = eri_up.swapaxes(1, 2).swapaxes(2, 3)
        eri_dw = eri_dw.swapaxes(1, 2).swapaxes(2, 3)
        eri_dw_up = eri_dw_up.swapaxes(1, 2).swapaxes(2, 3)
        eri_up_dw = eri_up_dw.swapaxes(1, 2).swapaxes(2, 3)
        
        eri_aa = S1Integrals(eri_up)
        eri_ab = S1Integrals(eri_up_dw)
        eri_bb = S1Integrals(eri_dw)
        eri_ba = S1Integrals(eri_dw_up)
        
        
        
        
        # Qiskit calculation
        # Convert matrices and tensors to a valid object for Qiskit
        integrals = ElectronicIntegrals.from_raw_integrals(
            h_ij_up,
            eri_aa,
            h_ij_dw,
            eri_bb,
            eri_ba,
            # auto_index_order=True,
            validate=True, 
        )
        qiskit_energy = ElectronicEnergy(integrals) # electronic energy, including nuclear repulsion
        qiskit_energy.nuclear_repulsion_energy = nucl_repulsion
        qiskit_problem = ElectronicStructureProblem(qiskit_energy) # create a problem for Qiskit

        #basis
        qiskit_problem.basis = ElectronicBasis.MO # molecular orbital basis
        
        # number of particles for spin-up, spin-down
        qiskit_problem.num_particles = num_particles
        qiskit_problem.num_spatial_orbitals = n_spatial_orbitals

        # qiskit_problem.reference_energy = reference_energy

        return qiskit_problem

    def solve_fci(self, n_energies=1):
        # Calculate matrix elements
        h_ij_up, h_ij_dw = self.calc_h_ij() # calculate one-electron integrals
        eri_up, eri_dw, eri_dw_up, eri_up_dw = self.calc_eri() # calculate two-electron integrals

        # Transform ERIs to chemist's index order
        eri_up = eri_up.swapaxes(1, 2).swapaxes(1, 3)
        eri_dw = eri_dw.swapaxes(1, 2).swapaxes(1, 3)
        eri_dw_up = eri_dw_up.swapaxes(1, 2).swapaxes(1, 3)
        eri_up_dw = eri_up_dw.swapaxes(1, 2).swapaxes(1, 3)

        # calculate repulsion between nucleus
        nucl_repulsion = calc_matrix_elements.nuclear_repulsion_energy(
            self.wfc_up_obj.atoms, self.wfc_up_obj.cell_volume
        )

        nelec = (int(np.sum(self.occupations_up)), int(np.sum(self.occupations_dw))) # number of spins up and down
        
        # FCI calculation
        nroots = n_energies  # number of excited states to calculate

        norb = self.wfc_up_obj.nbnd # number of molecular orbitals

        self.fcisolver = pyscf.fci.direct_uhf.FCISolver() #initialize FCI solver
        
        # Ordering of parameters from direct_uhf.make_hdiag
        # Calculate eigenvalues (energies) and eigenvectors (CI coefficients) with FCI
        self.fci_evs, self.fci_evcs = self.fcisolver.kernel(
            h1e=(h_ij_up.real, h_ij_dw.real),  # a, b (a=up, b=down)
            eri=(
                eri_up.real,
                eri_up_dw.real,
                eri_dw.real,
            ),  # aa, ab, bb (a=up, b=down)
            norb=norb,
            nelec=nelec,
            nroots=nroots,
        )

        # Save eigenvalues and -vectors in lists
        if n_energies == 1:
            self.fci_evs = np.array([self.fci_evs])
            self.fci_evcs = [self.fci_evcs]

        fci_energy = self.fci_evs + nucl_repulsion # total energy for FCI

        self.fci_energy = fci_energy
        return fci_energy
