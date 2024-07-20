import ase 
from ase.io import nwchem
with open("qe_files/3x3_periodic/NWChem/1covo/H1Li1-1.3.out") as file: 
    data = nwchem.read_nwchem_out(file, index=-1)
    
    