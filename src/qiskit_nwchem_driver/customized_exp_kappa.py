import numpy as np
import scipy

def construct_exp_kappa_matrix(
	num_orbs: int, num_inactive_orbs: int, num_active_orbs: int, kappa_elements: list)-> tuple[np.ndarray]:

	kappa = np.zeros((num_orbs, num_orbs)) # Initialize nxn matrix

	assert len(kappa_elements) == num_orbs * (num_orbs - 1) / 2, "The number of kappa elements should be the same as the number of elements of a lower triangle matrix"

	count = 0
	# inactive -> active
	for i in np.arange(num_inactive_orbs):
		for v in np.arange(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
			kappa[i,v]=kappa_elements[count]
			#print("see here",i,j,kappa[i,j])
			count = count + 1
	
	# inactive -> virtual
	for i in np.arange(num_inactive_orbs):
		for a in np.arange(num_inactive_orbs + num_active_orbs, num_orbs):
			kappa[i,a]=kappa_elements[count]
			#print("see here",i,j,kappa[i,j])
			count = count + 1

	# active -> virtual
	for v in np.arange(num_inactive_orbs, num_inactive_orbs + num_active_orbs):
		for a in np.arange(num_inactive_orbs + num_active_orbs, num_orbs):
			if count==len(kappa_elements):
				break
			else:
				kappa[v,a]=kappa_elements[count]
				#print("see here",i,j,kappa[i,j])
			
			count = count + 1

	kappa = kappa + -1.0*kappa.T # kappa = sum_pq kappa_pq (E_pq - E_qp) --> kappa^dagger = - kappa --> exp(kappa)^dagger = exp(-kappa) --> exp(kappa)*exp(kappa)^dagger = 1

	exp_kappa = scipy.linalg.expm(kappa)

	assert np.allclose(np.eye(num_orbs), exp_kappa@exp_kappa.T.conj()), "exp(kappa) is not unitary!"

	return exp_kappa



