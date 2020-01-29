import numpy as np
from itertools import permutations, product
from time import time
from casimir_functions import *

# Calls computations up to supersymmetrized casimir coefficients
# in basis if necessary. 

def check():
	# Check supertrace vanishing conditions (Killing)
	check = True
	num_check = 1000
	for i in range(1000):
		i1, i2, nu = np.random.randint(0,dim,3);
		cCheck = 0
		for rho in range(dim):
			cCheck += Killing[rho,i2]*SC[nu,i1,rho] + (-1)**(iD[nu]*(iD[i1]))*Killing[i1,rho]*SC[nu,i2,rho]
		if cCheck != 0:
			print("BAD 2Trace:", i1, i2, nu, cCheck)
			check = False
	if check:
		print("2Trace Clear on", num_check, "cases")

	# Check supertrace vanishing conditions (Quartic)
	check = True
	for i in range(1000):
		i1, i2, i3, i4, nu = np.random.randint(0,dim,5);
		cCheck = 0
		for rho in range(dim):
			cCheck += QT[rho,i2,i3,i4]*SC[nu,i1,rho] + (-1)**(iD[nu]*(iD[i1]))*QT[i1,rho,i3,i4]*SC[nu,i2,rho] + (-1)**(iD[nu]*(iD[i1]+iD[i2]))*QT[i1,i2,rho,i4]*SC[nu,i3,rho] + (-1)**(iD[nu]*(iD[i1]+iD[i2]+iD[i3]))*QT[i1,i2,i3,rho]*SC[nu,i4,rho]
		if cCheck != 0:
			print("BAD 4Trace:", i1, i2, i3, i4, nu, cCheck)
			check = False
	if check:
		print("4Trace Clear on", num_check, "cases")

	# Check random superbrackets on Killing
	check = True
	for i in range(1000):
		j1, j2, nu = np.random.randint(0,dim,3);
		cCheck = 0
		for rho in product(range(dim)):
			cCheck += SC[nu,rho,j1]*KI[rho,j2] + (-1)**((iD[j2]+1)*(iD[j1]*iD[nu]+iD[j1]+iD[nu]))*SC[nu,rho,j2]*KI[j1,rho]
		if cCheck != 0:
			print("BAD 2Commute:", j1,j2,nu, cCheck)
			check = False
	if check:
		print("2Commute Clear on", num_check, "cases")

	# Check random superbrackets on coefficients
	num_check = 10000
	check = True
	for i in range(num_check):
		j1, j2, j3, j4, nu = np.random.randint(0,dim,5);
		cCheck = 0
		for rho in range(dim):
			cCheck += QC[rho,j2,j3,j4]*SC[nu,rho,j1] + \
			(-1)**(iD[nu]*(iD[j1]))*QC[j1,rho,j3,j4]*SC[nu,rho,j2] + \
			(-1)**(iD[nu]*(iD[j1]+iD[j2]))*QC[j1,j2,rho,j4]*SC[nu,rho,j3] + \
			(-1)**(iD[nu]*(iD[j1]+iD[j2]+iD[j3]))*QC[j1,j2,j3,rho]*SC[nu,rho,j4]
		if np.abs(cCheck) > 10**(-10): # floating point issues
			print("BAD 4Commute:", j1, j2, j3, j4, nu, cCheck)
			check = False
	if check:
		print("4Commute Clear on", num_check, "cases")

	# Check random superbrackets on supersymmetrized coefficients
	check = True
	for i in range(num_check):
		j1, j2, j3, j4, nu = np.random.randint(0,dim,5);
		cCheck = 0
		for rho in range(dim):
			cCheck += SQC[rho,j2,j3,j4]*SC[nu,rho,j1] + \
			(-1)**(iD[nu]*(iD[j1]))*SQC[j1,rho,j3,j4]*SC[nu,rho,j2] + \
			(-1)**(iD[nu]*(iD[j1]+iD[j2]))*SQC[j1,j2,rho,j4]*SC[nu,rho,j3] + \
			(-1)**(iD[nu]*(iD[j1]+iD[j2]+iD[j3]))*SQC[j1,j2,j3,rho]*SC[nu,rho,j4]
		if np.abs(cCheck) > 10**(-10): # floating point issues
			print("BAD S4Commute:", j1, j2, j3, j4, nu, cCheck)
			check = False
	if check:
		print("S4Commute Clear on", num_check, "cases")

N = 2
mfactors = [5184,2500,1024,324,64,4,1,4,64] # Factors we multiply the QC by for each N to clarify pattern
types, second_range, full_dims, full_dim, basis, dbasis, index_dim, iD, index_type, bosonic_num, dim = make_arrays(N)
start = time()
#compute_traces(N)
#compute_structure_constants(N)
#compute_quartic_coeffs(N)
#compute_supersymmetrization(N)

# Load Computed Values
QT = np.load('CasimirData/QuarticTraceN'+str(N)+'.npy') # k_{i1,i2,i3,i4} = QT[i1,i2,i3,i4] : QuarticTrace
NQT = np.transpose(np.nonzero(QT)) # NQT[i] are the nonzero index sets
Killing = np.load('CasimirData/KillingN'+str(N)+'.npy') # k_{i,j} = Killing[i,j] : Killing Form
KI = np.linalg.inv(Killing) # k^{i,j} = KI[i,j] : Killing Inverse
NKI = np.nonzero(KI)[1] # KI[i,NKI[i]] is the only nonzero KI[i,.]
SC = np.load('CasimirData/StructureConstantsN'+str(N)+'.npy') # f_{u1,u2}^l1 = SC[u1,u2,l1] : Structure Coefficients 
QC = np.load('CasimirData/QuarticCoeffsN'+str(N)+'.npy')*mfactors[N] # k^{j1,j2,j3,j4} = QC[j1,j2,j3,j4] : Quartic Casimir Coefficients
QC0 = np.transpose(np.nonzero(QC))
print("Number of nonzero QC entries:", QC0.shape[0])
SQC = np.load('CasimirData/SQuarticCoeffsN'+str(N)+'.npy')*mfactors[N] 
SQC0 = np.transpose(np.nonzero(SQC))
num0 = SQC0.shape[0]
print("Number of nonzero SQC entries:", num0)

"""
# Checking proper supersymmetrization
ind = SQC0[np.random.randint(num0)]
for i1,i2,i3,i4 in permutations(ind):
	print(index_type[i1], index_type[i2], index_type[i3], index_type[i4],"\t|", i1, i2, i3, i4,"\t|",SQC[i1,i2,i3,i4])
"""
"""
# Conformal Terms
print("Conformal Terms")
for i1 in range(10):
	for i2 in range(i1,10):
		for i3 in range(i2,10):
			for i4 in range(i3,10):
				if SQC[i1,i2,i3,i4] != 0:
					print(index_type[i1], index_type[i2], index_type[i3], index_type[i4],"\t|", i1, i2, i3, i4,"\t|", to_string(basis[i1])[4:], to_string(basis[i2])[4:], to_string(basis[i3])[4:], to_string(basis[i4])[4:],"\t|",SQC[i1,i2,i3,i4])
"""
term_types = []
# All Terms
print("All Terms")
for i1 in range(dim):
	for i2 in range(i1,dim):
		for i3 in range(i2,dim):
			for i4 in range(i3,dim):
				if SQC[i1,i2,i3,i4] != 0:
					print(index_type[i1], index_type[i2], index_type[i3], index_type[i4],"\t|", i1, i2, i3, i4,"\t|", to_string(basis[i1])[4:], to_string(basis[i2])[4:], to_string(basis[i3])[4:], to_string(basis[i4])[4:],"\t|",SQC[i1,i2,i3,i4])
					term_name = index_type[i1] + index_type[i2] + index_type[i3]  + index_type[i4]
					if term_name not in term_types:
						term_types.append(term_name)

print(term_types)
#check()


print("Time: ", time() - start) 