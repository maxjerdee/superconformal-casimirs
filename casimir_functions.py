import numpy as np
from time import time
from itertools import product, permutations

# This file contains the relevant functions that may be used for Casimir Calculations

# Since I am not good enough at Mathematica, 
# here we abstractly perform the relevant superconformal algebra calculations. 
types = ["D","M","K","P","R","Q","S"]
second_range = []
full_dims = []
full_dim = []
basis = []
dbasis = []
index_dim = []
iD = []
index_type = []
bosonic_num = 0
dim = 0

def make_arrays(N=0): # Instaniating useful arrays given N = \mathcal{N} value. Function must be called before computation
	# We take the basis D, M_a^b, P_{ab}, K^{ab}, Q_{ar}, S^a{}_r, R_{rs}
	global second_range, full_dims, full_dim, basis, dbasis, index_dim, iD, index_type, bosonic_num, dim
	second_range = [1,2,2,2,N,N,N]
	full_dims = [1,4,4,4,N*N,2*N,2*N] # Full potential number of elements bracket can map into. 
	full_dim = sum(full_dims);
	# selecting spanning basis for the Killing Form trace:
	basis = [ttE("D",0,0),ttE("M",0,0),ttE("M",0,1),ttE("M",1,0),ttE("K",0,0),ttE("K",1,0),ttE("K",1,1),ttE("P",0,0),ttE("P",1,0),ttE("P",1,1)]
	# Dual basis allows for a projection of the full expression onto the selected basis
	dbasis = [ttE("D",0,0),ttE("M",0,0)-ttE("M",1,1),ttE("M",0,1),ttE("M",1,0),ttE("K",0,0),ttE("K",1,0)+ttE("K",0,1),ttE("K",1,1),ttE("P",0,0),ttE("P",1,0)+ttE("P",0,1),ttE("P",1,1)]
	# list of dimensions for D-grading:
	index_dim = [0,0,0,0,-1,-1,-1,1,1,1] # Dimension of that basis
	iD = [0,0,0,0,0,0,0,0,0,0]
	index_type = ["D","M","M","M","K","K","K","P","P","P"]

	for i in range(N): # Use R elements with i > j
		for j in range(i):
		 basis.append(ttE("R",i,j))
		 dbasis.append(ttE("R",i,j)-ttE("R",j,i))
		 index_dim.append(0)
		 iD.append(0)
		 index_type.append("R")
	bosonic_num = len(basis)
	for i in range(2): # All of the Q's
		for j in range(N):
		 basis.append(ttE("Q",i,j))
		 dbasis.append(ttE("Q",i,j))
		 index_dim.append(0.5)
		 iD.append(1)
		 index_type.append("Q")
	for i in range(2): # All of the S's
		for j in range(N):
		 basis.append(ttE("S",i,j))
		 dbasis.append(ttE("S",i,j))
		 index_dim.append(-0.5)
		 iD.append(1)
		 index_type.append("S")
	dim = len(basis)
	return types, second_range, full_dims, full_dim, basis, dbasis, index_dim, iD, index_type, bosonic_num, dim

def to_string(X, reduce=False): # Express a full array as the corresponding basis elements (perhaps reduced)
	if reduce:
		return 
	res = ""
	c = 0; r = 0; t = 0;
	for i in range(full_dim):
		if X[i] != 0:
			scal = X[i]
			if np.imag(scal) == 0:
				scal = np.real(scal)
			res += str(scal) + " " + str(types[t]) + "("+ str(c)+ "," + str(r) + ") + " 
		r += 1
		if r >= second_range[t]: 
			r = 0
			c += 1
		if c*second_range[t] + r >= full_dims[t]:
			c = 0; r = 0; t += 1
	return res[:-3]

def bra(X1, X2): # Defines the bracket operation between general algebra elements
	res = np.zeros(full_dim,dtype=complex);
	for i,j in product(np.nditer(np.nonzero(X1),flags=["zerosize_ok"]),np.nditer(np.nonzero(X2),flags=["zerosize_ok"])):
		res += X1[i]*X2[j]*basis_bracket(i,j)
	return res

def DK(c1,r1,c2,r2):
	return -ttE("K",c2,r2)
def DP(c1,r1,c2,r2):
	return ttE("P",c2,r2)	
def DQ(c1,r1,c2,r2):
	return 1/2*ttE("Q",c2,r2)
def DS(c1,r1,c2,r2):
	return -1/2*ttE("S",c2,r2)
def MM(c1,r1,c2,r2):
	return -de(c1,r2)*ttE("M",c2,r1)+de(c2,r1)*ttE("M",c1,r2)
def MK(c1,r1,c2,r2):
	return -de(c1,c2)*ttE("K",r1,r2) - de(c1,r2)*ttE("K",r1,c2)+de(c1,r1)*ttE("K",c2,r2)
def MP(c1,r1,c2,r2):
	return de(c2,r1)*ttE("P",c1,r2) + de(r2,r1)*ttE("P",c1,c2) - de(c1,r1)*ttE("P",c2,r2)
def MQ(c1,r1,c2,r2):
	return de(c2,r1)*ttE("Q",c1,r2) - 1/2*de(c1,r1)*ttE("Q",c2,r2)
def MS(c1,r1,c2,r2):
	return -de(c1,c2)*ttE("S",r1,r2) + 1/2*de(c1,r1)*ttE("S",c2,r2)
def KP(c1,r1,c2,r2):
	return de(c2,c1)*ttE("M",r2,r1)+de(c2,r1)*ttE("M",r2,c1)+de(r2,c1)*ttE("M",c2,r1)+de(r2,r1)*ttE("M",c2,c1)+2*(de(c2,c1)*de(r2,r1)+de(r2,c1)*de(c2,r1))*ttE("D",0,0)
def KQ(c1,r1,c2,r2):
	return -1j*(de(c2,c1)*ttE("S",r1,r2)+de(c2,r1)*ttE("S",c1,r2))
def PS(c1,r1,c2,r2):
	return -1j*(de(c1,c2)*ttE("Q",r1,r2)+de(r1,c2)*ttE("Q",c1,r2))
def RR(c1,r1,c2,r2):
	return 1j*(de(c1,c2)*ttE("R",r1,r2)+de(r1,r2)*ttE("R",c1,c2)-de(c1,r2)*ttE("R",r1,c2)-de(r1,c2)*ttE("R",c1,r2))
def RQ(c1,r1,c2,r2):
	return 1j*(de(c1,r2)*ttE("Q",c2,r1)-de(r1,r2)*ttE("Q",c2,c1))
def RS(c1,r1,c2,r2):
	return 1j*(de(c1,r2)*ttE("S",c2,r1)-de(r1,r2)*ttE("S",c2,c1))
def QQ(c1,r1,c2,r2):
	return 2*de(r1,r2)*ttE("P",c1,c2)
def QS(c1,r1,c2,r2):
	return 2j*(de(r1,r2)*(ttE("M",c1,c2)+de(c1,c2)*ttE("D",0,0)) - 1j*de(c1,c2)*ttE("R",r1,r2))
def SS(c1,r1,c2,r2):
	return -2*de(r1,r2)*ttE("K",c1,c2)

relations = {
		(0,2):DK,
		(0,3):DP,
		(0,5):DQ,
		(0,6):DS,
		(1,1):MM,
		(1,2):MK,
		(1,3):MP,
		(1,5):MQ,
		(1,6):MS,
		(2,3):KP,
		(2,5):KQ,
		(3,6):PS,
		(4,4):RR,
		(4,5):RQ,
		(4,6):RS,
		(5,5):QQ,
		(5,6):QS,
		(6,6):SS
	}

def basis_bracket(i1,i2): # Bracket on algebra basis elements of indices i1,i2.
	t1, c1, r1 = index_to_type(i1);
	t2, c2, r2 = index_to_type(i2);
	sign = 1
	if t1 > t2:
		temp = t2,c2,r2
		t2,c2,r2=t1,c1,r1
		t1,c1,r1=temp
		if (t1,t2) != (5,6):
			sign = -1
	
	if (t1,t2) in relations:
		return relations[(t1,t2)](c1,r1,c2,r2)*sign
	return np.zeros(full_dim);

def structure_constant(l1,l2,u1):
	return np.dot(bra(basis[l1],basis[l2]),dbasis[u1])

def de(a,b):
	if a == b:
		return 1
	return 0

def index_to_type(index):
	t = 0
	while full_dims[t] <= index:
		index = index - full_dims[t];
		t += 1
	r = index % second_range[t]
	c = int(index / second_range[t])
	return t, c, r

def type_to_index(ty,c,r):
	t = types.index(ty)
	index = 0
	for i in range(t):
		index += full_dims[i]
	index += second_range[t]*c + r
	return index

def ttE(ty,c,r): #type to Element
	res = np.zeros(full_dim, dtype=complex)
	res[type_to_index(ty,c,r)] = 1
	return res

def killing(X1,X2):
	res = 0
	for i in range(len(basis)):
		X3 = bra(X1,bra(X2,basis[i]));
		#print(toString(basis[i]), np.dot(X3,dbasis[i]))
		if i < bosonic_num:
			res += np.dot(X3,dbasis[i])
		else:
			res -= np.dot(X3,dbasis[i])
	return res

def quartic_invariant(X1,X2,X3,X4):
	res = 0
	for i in range(len(basis)):
		X5 = bra(X1,bra(X2,bra(X3,bra(X4,basis[i]))));
		#print(toString(basis[i]), np.dot(X3,dbasis[i]))
		if i < bosonic_num:
			res += np.dot(X5,dbasis[i])
		else:
			res -= np.dot(X5,dbasis[i])
	return res

def fermionic_order(index_list): # Given the tuple of indices, determine the 
	ferm_indices = [i for i in index_list if i >= bosonic_num]
	sign = 1
	for i in range(len(ferm_indices)):
		for j in range(i+1,len(ferm_indices)):
			if ferm_indices[i] > ferm_indices[j]:
				sign *= -1
			elif ferm_indices[i] == ferm_indices[j]:
				sign = 0
	return sign

# Computation Functions
def compute_traces(N=0): # Compute Killing Form and quartic traces
	types, second_range, full_dims, full_dim, basis, dbasis, index_dim, iD, index_type, bosonic_num, dim = make_arrays(N)
	start = time()
	print("Computing Killing Form:")
	kill = np.zeros((dim,dim),dtype=complex);
	for i in range(dim):
		for j in range(dim):
			if index_dim[i] + index_dim[j] == 0:
				kill[i,j] = killing(basis[i],basis[j])
	np.save(('CasimirData/KillingN'+str(N)+'.npy'),kill)
	killI = np.linalg.inv(kill)
	print("Time: ", time() - start)

	print("Computing Quartic:")
	num_compute = 0;
	quartic_trace = np.zeros((dim,dim,dim,dim),dtype=complex)
	for i1 in range(dim):
		print(i1,"/",dim, (time()-start))
		for i2 in range(i1+1):
			for i3 in range(i1+1):
				for i4 in range(i1+1):
					if index_dim[i1] + index_dim[i2] + index_dim[i3] + index_dim[i4] == 0:
						num_compute += 1
						quartic_trace[i1,i2,i3,i4] = quartic_invariant(basis[i1],basis[i2],basis[i3],basis[i4])
						quartic_trace[i2,i3,i4,i1] = quartic_trace[i1,i2,i3,i4]*(lambda: -1, lambda: 1)[i1 < bosonic_num]()
						quartic_trace[i3,i4,i1,i2] = quartic_trace[i2,i3,i4,i1]*(lambda: -1, lambda: 1)[i2 < bosonic_num]()
						quartic_trace[i4,i1,i2,i3] = quartic_trace[i3,i4,i1,i2]*(lambda: -1, lambda: 1)[i3 < bosonic_num]()
	print(num_compute)
	np.save(('CasimirData/QuarticTraceN'+str(N)+'.npy'),quartic_trace)

def compute_structure_constants(N=0):
	types, second_range, full_dims, full_dim, basis, dbasis, index_dim, iD, index_type, bosonic_num, dim = make_arrays(N)
	structure_constants = np.zeros((dim,dim,dim),dtype=complex)
	for l1, l2, u1 in product(range(dim),range(dim),range(dim)):
		structure_constants[l1,l2,u1] = structure_constant(l1,l2,u1)
	np.save(('CasimirData/StructureConstantsN'+str(N)+'.npy'),structure_constants)

""" # Old version (Fails for superconformal case)
def compute_quartic_coeffs(N=0): # Raise indices using Killing Form (first compute traces)
	Killing = np.load('CasimirData/KillingN'+str(N)+'.npy') # k_{i,j} = Killing[i,j] : Killing Form
	KI = np.linalg.inv(Killing) # k^{i,j} = KI[i,j] : Killing Inverse
	NKI = np.nonzero(KI)[1] # KI[i,NKI[i]] is the only nonzero KI[i,.]
	QT = np.load('CasimirData/QuarticTraceN'+str(N)+'.npy') # k_{i1,i2,i3,i4} = QT[i1,i2,i3,i4] : QuarticTrace
	NQT = np.transpose(np.nonzero(QT)) # NQT[i] are the nonzero index sets
	QC = np.zeros((dim,dim,dim,dim),dtype=complex) # Quartic Casimir Coefficients

	for ind in NQT:
		QC[NKI[ind[0]],NKI[ind[1]],NKI[ind[2]],NKI[ind[3]]] += KI[NKI[ind[0]],ind[0]]*KI[NKI[ind[1]],ind[1]]*KI[NKI[ind[2]],ind[2]]*KI[NKI[ind[3]],ind[3]]*QT[ind[0],ind[1],ind[2],ind[3]]
	np.save(('CasimirData/QuarticCoeffsN'+str(N)+'.npy'),QC)
"""
def compute_quartic_coeffs(N=0): # Raise indices using Killing Form (first compute traces)
	Killing = np.load('CasimirData/KillingN'+str(N)+'.npy') # k_{i,j} = Killing[i,j] : Killing Form
	KI = np.linalg.inv(Killing) # k^{i,j} = KI[i,j] : Killing Inverse
	NKI = np.nonzero(KI)[1] # KI[i,NKI[i]] is the only nonzero KI[i,.]
	QT = np.load('CasimirData/QuarticTraceN'+str(N)+'.npy') # k_{i1,i2,i3,i4} = QT[i1,i2,i3,i4] : QuarticTrace
	NQT = np.transpose(np.nonzero(QT)) # NQT[i] are the nonzero index sets
	QC = np.zeros((dim,dim,dim,dim),dtype=complex) # Quartic Casimir Coefficients

	for ind in NQT:
		i1,i2,i3,i4 = ind[0],ind[1],ind[2],ind[3]
		j1,j2,j3,j4 = NKI[ind[0]],NKI[ind[1]],NKI[ind[2]],NKI[ind[3]]
		QC[j1,j2,j3,j4] += \
		(-1)**(iD[j1]+iD[j2]+iD[j3]+iD[j4]+(iD[i1]+iD[i2]+iD[i3]+iD[i4])/2)*\
		QT[i1,i2,i3,i4]*KI[i1,j1]*KI[i2,j2]*KI[i3,j3]*KI[i4,j4]
	np.save(('CasimirData/QuarticCoeffsN'+str(N)+'.npy'),QC)

def compute_supersymmetrization(N=0): # Super-Symmetrization
	QC = np.load('CasimirData/QuarticCoeffsN'+str(N)+'.npy')	 # k^{j1,j2,j3,j4} = QC[j1,j2,j3,j4] : Quartic Casimir Coefficients
	QC0 = np.transpose(np.nonzero(QC))
	SQC = np.zeros((dim,dim,dim,dim),dtype=complex)
	for ind in QC0:
		i1,i2,i3,i4 = ind[0],ind[1],ind[2],ind[3]
		res = 0
		for j1,j2,j3,j4 in permutations([i1,i2,i3,i4]):
			res += QC[j1,j2,j3,j4]*fermionic_order([j1,j2,j3,j4])
		for j1,j2,j3,j4 in permutations([i1,i2,i3,i4]):
			SQC[j1,j2,j3,j4] = res/24*fermionic_order([j1,j2,j3,j4])

	np.save(('CasimirData/SQuarticCoeffsN'+str(N)+'.npy'),SQC)
