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

    def run(self=True) -> ElectronicStructureProblem:
        return self.to_problem()

    def to_problem(
        self,
        basis: ElectronicBasis = ElectronicBasis.MO,
        include_dipole: bool = False,
    ) -> ElectronicStructureProblem:
        # if basis != ElectronicBasis.MO:
        #     warn(
        #         f"In {self.__class__.__name__}.{self.to_problem.__name__}:\n"
        #         + "Using MO basis although AO basis was specified, "
        #         + "since the AO basis is the plane-wave basis and typically "
        #         + "a large number of plane-waves is used which would result "
        #         + "in large matrices!"
        #     )
        # basis: ElectronicBasis = ElectronicBasis.MO
        # include_dipole: bool = False

        # qcschema = self.to_qcschema(include_dipole=include_dipole)

        # problem = qcschema_to_problem(
        #     qcschema, basis=basis, include_dipole=include_dipole
        # )

        # if include_dipole and problem.properties.electronic_dipole_moment is not None:
        #     problem.properties.electronic_dipole_moment.reverse_dipole_sign = True
        problem = self.to_qiskit_problem_old()
        return problem


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
            if two_electron_spatial_integrals[k, l, i, j] == 0:
                two_electron_spatial_integrals[k, l, i, j] = val[4]
            if two_electron_spatial_integrals[i, j, l, k] == 0:
                two_electron_spatial_integrals[i, j, l, k] = val[4]
            if two_electron_spatial_integrals[l, k, i, j] == 0:
                two_electron_spatial_integrals[l, k, i, j] = val[4]
            if two_electron_spatial_integrals[j, i, k, l] == 0:
                two_electron_spatial_integrals[j, i, k, l] = val[4]
            if two_electron_spatial_integrals[k, l, j, i] == 0:
                two_electron_spatial_integrals[k, l, j, i] = val[4]
            if two_electron_spatial_integrals[j, i, l, k] == 0:
                two_electron_spatial_integrals[j, i, l, k] = val[4]
            if two_electron_spatial_integrals[l, k, j, i] == 0:
                two_electron_spatial_integrals[l, k, j, i] = val[4]

        return one_electron_spatial_integrals, two_electron_spatial_integrals

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

                        if i!=k and j!=l:   # Pauli exclusion priciple
                            h2[i,j,k,l] = two_electron[i,j,k,l]
                            h2[i + n_orb, j + n_orb, k + n_orb, l + n_orb] = two_electron[i, j, k, l]
        return h1, 0.5*h2

    def load_from_yaml(self,file_name):
        
        data = yaml.load(open(file_name,"r"),SafeLoader)
        n_electrons = data['integral_sets'][0]['n_electrons']
        n_spatial_orbitals = data['integral_sets'][0]['n_orbitals']
        nuclear_repulsion_energy = data['integral_sets'][0]['coulomb_repulsion']['value']

        one_electron_import = data['integral_sets'][0]['hamiltonian']['one_electron_integrals']['values']
        two_electron_import = data['integral_sets'][0]['hamiltonian']['two_electron_integrals']['values']

        one_electron_spatial_integrals, two_electron_spatial_integrals = self.get_spatial_integrals(one_electron_import,two_electron_import,n_spatial_orbitals)
        h1, h2 = self.convert_to_spin_index(one_electron_spatial_integrals,two_electron_spatial_integrals,n_spatial_orbitals)
        reference_energy = data['integral_sets'][0]['initial_state_suggestions'][0]['state']['energy']['value']
        symbols = [i['name'] for i in data['integral_sets'][0]['geometry']['atoms']]
        coords = []
        for i in data['integral_sets'][0]['geometry']['atoms']:
            coords = coords + i['coords']
        coords = np.array(coords)
        # return n_electrons, n_spatial_orbitals, nuclear_repulsion_energy, h1, h2, reference_energy, symbols, coords
        return n_electrons, n_spatial_orbitals, nuclear_repulsion_energy, h1, h2, reference_energy
    def to_qcschema(self, *, include_dipole: bool = True) -> QCSchema:
        include_dipole: bool = False
        num_particles, n_spatial_orbitals, nucl_repulsion, h1, h2, reference_energy, symbols, coords = self.load_from_yaml(self.nwchem_output)
        h_ij_up = h1[:n_spatial_orbitals,:n_spatial_orbitals]
        h_ij_dw = h1[n_spatial_orbitals:, n_spatial_orbitals:]
        eri_up = h2[:n_spatial_orbitals,:n_spatial_orbitals,:n_spatial_orbitals,:n_spatial_orbitals]
        eri_dw = h2[n_spatial_orbitals:, n_spatial_orbitals:, n_spatial_orbitals:, n_spatial_orbitals:]
        eri_up_dw = h2[:n_spatial_orbitals, :n_spatial_orbitals, n_spatial_orbitals:, n_spatial_orbitals:]
        eri_dw_up = h2[n_spatial_orbitals:, n_spatial_orbitals:, :n_spatial_orbitals, :n_spatial_orbitals]
        # # Calculate matrix elements
        # h_ij_up, h_ij_dw = self.calc_h_ij()
        # # All ERIs are given in physicists' order
        # eri_up, eri_dw, eri_dw_up, eri_up_dw = self.calc_eri()
        
        # Transform ERIs to chemist's index order to define S1Integrals object
        # Do not use qiskit_naute function to_chemist_ordering since
        # it tries to determine the index order of the ERIs
        # which fails for eri_up_dw and eri_dw_up because ERIs connecting different
        # spins do not satisfy all symmetries and therefore a index order
        # cannot be detemined based on symmetries
        eri_up = eri_up.swapaxes(1, 2).swapaxes(1, 3)
        eri_dw = eri_dw.swapaxes(1, 2).swapaxes(1, 3)
        eri_dw_up = eri_dw_up.swapaxes(1, 2).swapaxes(1, 3)
        eri_up_dw = eri_up_dw.swapaxes(1, 2).swapaxes(1, 3)

        print(f"h_ij up-down equal: {np.allclose(h_ij_dw, h_ij_up)}")
        print(f"eri up-down equal: {np.allclose(eri_dw, eri_up)}")
        print(f"eri up-(down-up) equal: {np.allclose(eri_dw_up, eri_up)}")
        print(f"eri (up-down)-(down-up) equal: {np.allclose(eri_up_dw, eri_dw_up)}")

        # nucl_repulsion = calc_matrix_elements.nuclear_repulsion_energy(
        #     self.wfc_up_obj.atoms, self.wfc_up_obj.cell_volume
        # )

        # ??? Is the following correct for calculating the overlap matrix?
        #     We search for the overlap matrix between two AOs, i.e. plane-waves
        #     which should be the identity, right?
        #     For spin-polarized QE calculations (self.c_ip_up.conj() @ self.c_ip_up.T)
        #     and (self.c_ip_dw.conj() @ self.c_ip_dw.T) are the overlap matrices between the
        #     Kohn-Sham orbitals of up- and down-spin, respectively. Both of them are
        #     identity matrices. But the overlap between up- and down-spin
        #     (self.c_ip_up.conj() @ self.c_ip_dw.T) is used to calculate the
        #     AngularMomentum operator in qcschema_to_problem.
        #     See qiskit_nature.second_q.formats.qcschema_translator.get_overlap_ab_from_qcschema
        #     where (self.c_ip_up @ self.c_ip_dw.T) instead of (self.c_ip_up.conj() @ self.c_ip_dw.T)
        #     is calculated if the overlap matrix is the identity (as is the case here) and note
        #     that coeff_a is equal to self.c_ip_up.T which is the same as data.mo_coeff below.
        #     We suspect that there is a bug in
        #     qiskit_nature.second_q.formats.qcschema_translator.get_overlap_ab_from_qcschema
        #     and coeff_a.T.conj() @ overlap @ coeff_b would be the correct formula.
        #     Note that the ground state needs to have a zero angular momentum.
        #     For our example of H_2 with the identity matrix as the overlap matrix the
        #     angular momentum expectation value of the ground state is ~1e-8 for which
        #     np.isclose(~1e-8, 0.0) is False. Therefore, the ground-state is not identified
        #     as the ground state by the numpy ground-state eigensolver.
        #     The numpy ground-state eigensolver would find the correct ground-state if
        #     the overlap calculated with the get_overlap_ab_from_qcschema function would
        #     return the identity matrix which can be forced by setting data.overlap
        #     below to None.
        #     For our example of H_2 (self.c_ip_up.conj() @ self.c_ip_dw.T) is not equal
        #     to the identity matrix. Note that (self.c_ip_up.conj() @ self.c_ip_dw.T) =
        #     (self.c_ip_up.conj() @ overlap @ self.c_ip_dw.T) if overlap is the identity matrix
        #     which is the case for our plane-wave AO basis. We think that
        #     (self.c_ip_up.conj() @ self.c_ip_dw.T) does not have to be the identity matrix
        #     but the angular momentum has to be zero for the ground state.
        #     We have to further investigate different spin-polarized DFT calculation and
        #     the angular momentum of their many-body ground states to check if
        #     there is a bug in qiskit_nature or in our understand of the overlap matrix
        # overlap = np.eye(self.c_ip_up.shape[1], dtype=np.float64)
        overlap = None

        # Molecular orbitals (MOs) are the Kohn-Sham orbitals
        # Atomic orbitals (AOs) are the plane-waves
        # Up=a, down=b
        data = _QCSchemaData()
        # data.hij # h_ij in atomic orbital basis, i.e. plane-waves in our case
        # data.hij_b # h_ij_b in atomic orbital basis, i.e. plane-waves in our case
        # data.eri # eri in atomic orbital basis, i.e. plane-waves in our case
        data.hij_mo = h_ij_up
        data.hij_mo_b = h_ij_dw

        data.eri_mo = S1Integrals(eri_up)
        data.eri_mo_ba = S1Integrals(eri_dw_up)
        data.eri_mo_bb = S1Integrals(eri_dw)

        data.e_nuc = nucl_repulsion
        data.e_ref = reference_energy
        
        data.overlap = overlap

        # haven't decide yet 
        # data.mo_coeff = (
        #     self.c_ip_up.T
        # )  # shape: (nao, nmo) = (#plane-waves, #kohn-sham orbitals)
        # data.mo_coeff_b = (
        #     self.c_ip_dw.T
        # )  # shape: (nao, nmo) = (#plane-waves, #kohn-sham orbitals)

        
        # data.mo_energy = self.wfc_up_obj.ks_energies
        # data.mo_energy_b = self.wfc_dw_obj.ks_energies
        # data.mo_occ = self.occupations_up
        # data.mo_occ_b = self.occupations_dw
        
        
        
        # data.dip_x
        # data.dip_y
        # data.dip_z
        # data.dip_mo_x_a
        # data.dip_mo_y_a
        # data.dip_mo_z_a
        # data.dip_mo_x_b
        # data.dip_mo_y_b
        # data.dip_mo_z_b
        # data.dip_nuc
        # data.dip_ref
        data.symbols = symbols
        data.coords = coords
        
        
        # data.multiplicity = self.nspin + 1  # Spin + 1
        data.multiplicity = 2
        data.charge = 0
        # data.masses
        # data.method
        # data.basis = self.basis
        
        # data.creator = self.creator
        # data.version = self.version
        # # data.routine
        # data.nbasis = self.p.shape[0]
        # data.nmo = self.c_ip_up.shape[0]
        # data.nalpha = int(np.sum(self.occupations_up))
        # data.nbeta = int(np.sum(self.occupations_dw))
        data.nalpha = int(num_particles/2)
        data.nbeta = int(num_particles/2)
        # data.keywords

        return self._to_qcschema(data, include_dipole=include_dipole)

    def calc_h_ij(self):
        # Kinetic energy
        iTj_up = calc_matrix_elements.iTj(self.p, self.c_ip_up)
        iTj_dw = calc_matrix_elements.iTj(self.p, self.c_ip_dw)

        # Nuclear repulsion
        iUj_up = calc_matrix_elements.iUj(
            self.p,
            self.c_ip_up,
            self.wfc_up_obj.atoms,
            self.wfc_up_obj.cell_volume,
        )
        iUj_dw = calc_matrix_elements.iUj(
            self.p,
            self.c_ip_dw,
            self.wfc_dw_obj.atoms,
            self.wfc_dw_obj.cell_volume,
        )

        h_ij_up = iTj_up - iUj_up
        h_ij_dw = iTj_dw - iUj_dw

        return h_ij_up, h_ij_dw

    def calc_eri(self):
        # Calculate ERIs via pair density
        assert (
            self.wfc_up_obj.gamma_only
        ), "Calculating ERIs via pair densities is only implemented for the gamma-point!"
        assert (
            self.wfc_dw_obj.gamma_only
        ), "Calculating ERIs via pair densities is only implemented for the gamma-point!"

        eri_up: np.ndarray = (
            eri_pair_densities.eri_gamma(p=self.p, c_ip_up=self.c_ip_up)
            / self.wfc_up_obj.cell_volume
        )
        eri_dw: np.ndarray = (
            eri_pair_densities.eri_gamma(p=self.p, c_ip_up=self.c_ip_dw)
            / self.wfc_dw_obj.cell_volume
        )
        eri_dw_up: np.ndarray = (
            eri_pair_densities.eri_gamma(
                p=self.p, c_ip_up=self.c_ip_up, c_ip_dw=self.c_ip_dw
            )
            / self.wfc_dw_obj.cell_volume
        )
        # eri_up_dw: np.ndarray = (
        #     eri_pair_densities.eri_gamma(
        #         p=self.p, c_ip_up=self.c_ip_dw, c_ip_dw=self.c_ip_up
        #     )
        #     / self.wfc_dw_obj.cell_volume
        # )
        eri_up_dw: np.ndarray = eri_dw_up.swapaxes(0, 1).swapaxes(2, 3)

        return eri_up, eri_dw, eri_dw_up, eri_up_dw

    
    
    
    def to_qiskit_problem_old(self):
        num_particles, n_spatial_orbitals, nucl_repulsion, h1, h2, reference_energy = self.load_from_yaml(self.nwchem_output)
        # # Calculate matrix elements
        # h_ij_up, h_ij_dw = self.calc_h_ij()     # one electron integrals
        # eri_up, eri_dw, eri_dw_up, eri_up_dw = self.calc_eri()      # two electron integrals

        # nucl_repulsion = calc_matrix_elements.nuclear_repulsion_energy(     # effective nuclear repulsion energy
        #     self.wfc_up_obj.atoms, self.wfc_up_obj.cell_volume      # volume of supercell 
        # )

        num_particles = (
            1,
            1
         )   # number of electron 
        h_ij_up = h1[:n_spatial_orbitals,:n_spatial_orbitals]
        h_ij_dw = h1[n_spatial_orbitals:, n_spatial_orbitals:]
        eri_up = h2[:n_spatial_orbitals,:n_spatial_orbitals,:n_spatial_orbitals,:n_spatial_orbitals]
        eri_dw = h2[n_spatial_orbitals:, n_spatial_orbitals:, n_spatial_orbitals:, n_spatial_orbitals:]
        eri_up_dw = h2[:n_spatial_orbitals, :n_spatial_orbitals, n_spatial_orbitals:, n_spatial_orbitals:]
        eri_up = eri_up.swapaxes(1, 2).swapaxes(1, 3)
        eri_dw = eri_dw.swapaxes(1, 2).swapaxes(1, 3)
        eri_up_dw = eri_up_dw.swapaxes(1, 2).swapaxes(1, 3)
        eri_aa = S1Integrals(eri_up)
        eri_ab = S1Integrals(eri_up_dw)
        eri_bb = S1Integrals(eri_dw)
        
        
        
        # Qiskit calculation
        integrals = ElectronicIntegrals.from_raw_integrals(
            h_ij_up,
            eri_aa,
            h_ij_dw,
            eri_bb,
            eri_ab,
            auto_index_order=True,
            validate=True,
        )
        qiskit_energy = ElectronicEnergy(integrals)
        qiskit_energy.nuclear_repulsion_energy = nucl_repulsion
        qiskit_problem = ElectronicStructureProblem(qiskit_energy)

        #basis
        qiskit_problem.basis = ElectronicBasis.MO
        
        # number of particles for spin-up, spin-down
        qiskit_problem.num_particles = num_particles
        qiskit_problem.num_spatial_orbitals = n_spatial_orbitals

        qiskit_problem.reference_energy = reference_energy

        return qiskit_problem

    def solve_fci(self, n_energies=1):
        # Calculate matrix elements
        h_ij_up, h_ij_dw = self.calc_h_ij()
        eri_up, eri_dw, eri_dw_up, eri_up_dw = self.calc_eri()

        # Transform ERIs to chemist's index order
        eri_up = eri_up.swapaxes(1, 2).swapaxes(1, 3)
        eri_dw = eri_dw.swapaxes(1, 2).swapaxes(1, 3)
        eri_dw_up = eri_dw_up.swapaxes(1, 2).swapaxes(1, 3)
        eri_up_dw = eri_up_dw.swapaxes(1, 2).swapaxes(1, 3)

        nucl_repulsion = calc_matrix_elements.nuclear_repulsion_energy(
            self.wfc_up_obj.atoms, self.wfc_up_obj.cell_volume
        )

        nelec = (int(np.sum(self.occupations_up)), int(np.sum(self.occupations_dw)))
        # FCI calculation
        nroots = n_energies  # number of states to calculate

        norb = self.wfc_up_obj.nbnd

        self.fcisolver = pyscf.fci.direct_uhf.FCISolver()
        # Ordering of parameters from direct_uhf.make_hdiag
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

        fci_energy = self.fci_evs + nucl_repulsion

        self.fci_energy = fci_energy
        return fci_energy
