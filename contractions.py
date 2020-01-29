import numpy as np
from itertools import permutations, product
from time import time
from casimir_functions import *
from scipy.sparse.linalg import lsqr
from string import Template
from fractions import Fraction

# "Projects" various supersymmetrized combinations of terms in full basis 

def find_contribution(types, index_ranges, in1, in2):
	contribution = np.zeros((dim,dim,dim,dim),dtype=complex)
	cons = {}
	for a,b,c,d in product(range(index_ranges[0]),range(index_ranges[1]),range(index_ranges[2]),range(index_ranges[3])):
		inds = [a,b,c,d]
		ind1 = [inds[in1[0]],inds[in1[1]],inds[in1[2]],inds[in1[3]]]
		ind2 = [inds[in2[0]],inds[in2[1]],inds[in2[2]],inds[in2[3]]]
		#print(ind1,ind2)
		i1 = type_to_index(types[0],ind1[0],ind2[0])
		i2 = type_to_index(types[1],ind1[1],ind2[1])
		i3 = type_to_index(types[2],ind1[2],ind2[2])
		i4 = type_to_index(types[3],ind1[3],ind2[3])
		bi1, bi2, bi3, bi4 = bindex[i1], bindex[i2], bindex[i3], bindex[i4]
		sign = bsign[i1]*bsign[i2]*bsign[i3]*bsign[i4]*fermionic_order([bi1,bi2,bi3,bi4])
		#print(i1,i2,i3,i4,bi1,bi2,bi3,bi4,sign)
		for j1,j2,j3,j4 in permutations([bi1,bi2,bi3,bi4]):
			contribution[j1,j2,j3,j4] += sign
		cons = {}
		con0 = np.transpose(np.nonzero(contribution))
		for ind in con0:
			if ind[0] <= ind[1] <= ind[2] <= ind[3]:
				cons[str(ind[0]) + " " + str(ind[1]) + " " + str(ind[2]) + " " + str(ind[3])] = \
				contribution[ind[0],ind[1],ind[2],ind[3]]/24
	return cons

def to_latex(string): # Convert the string form (i.e. "M_0^1 M_1^0 Q_23 S^2_3" -> "M_\alpha^\beta M_\beta^\alpha Q_{\gamma r} S^\gamma{}_r")
	greek = ["\\alpha ","\\beta ","\\gamma ","\\delta "]
	roman = ["r","s","t","u"]
	type_types = [["none","none"],["greek","greek"],["greek","greek"],["greek","greek"],["roman","roman"],["greek","roman"],["greek","roman"]]
	type_forms = ["D","M_$a{}^$b","K^{$a$b}","P_{$a$b}","R_{$a$b}","Q_{$a$b}","S^$a{}_$b"]
	parts = string.split()
	info = []
	for p in parts:
		temp = []
		temp.append(types.index(p[:1]))
		for i in range(len(p)):
			if p[i:i+1].isdigit():
				temp.append(int(p[i:i+1]))
		info.append(temp)
	index_types = ["none","none","none","none"]
	for inf in info:
		if len(inf) > 1:
			index_types[inf[1]] = type_types[inf[0]][0]
			index_types[inf[2]] = type_types[inf[0]][1]
	index_names = ["","","",""]
	greek_num = 0
	roman_num = 0
	for i in range(len(index_types)):
		if index_types[i] == "greek":
			index_names[i] = greek[greek_num]
			greek_num += 1
		if index_types[i] == "roman":
			index_names[i] = roman[roman_num]
			roman_num += 1
	res = ""
	for inf in info:
		if inf[0] == 0:
			res += "D"
		else:
			res += Template(type_forms[inf[0]]).substitute(a = index_names[inf[1]],b = index_names[inf[2]])
	return res

def format_number(z):
	real = np.real(z)
	imag = np.imag(z)
	if imag == 0:
		return mini_format(real)
	if real == 0:
		return mini_format(imag) + "i"
	else:
		return mini_format(real) + " + " + mini_format(imag) + "i"

def mini_format(num):
	if np.abs(np.abs(num) - 1) < 10**(-6): # Floating Point
		return str(Fraction(num).limit_denominator(10000))[:-1]
	if np.abs(num - round(num)) < 10**(-3):
		return str(Fraction(num).limit_denominator(10000))
	if num < 0:
		return "-("+str(Fraction(num).limit_denominator(10000))[1:]+")"
	return "("+str(Fraction(num).limit_denominator(10000))+")"
	

N = 2
mfactors = [5184,2500,1024,324,64,4,1,4,64] # Factors we multiply the QC by for each N to clarify pattern
types, second_range, full_dims, full_dim, basis, dbasis, index_dim, iD, index_type, bosonic_num, dim = make_arrays(N)
start = time()
#compute_traces(N)
#compute_structure_constants(N)
#compute_quartic_coeffs(N)
#compute_supersymmetrization(N)

# Load Computed Values
SQC = np.load('CasimirData/SQuarticCoeffsN'+str(N)+'.npy')*mfactors[N] 
SQC0 = np.transpose(np.nonzero(SQC))
num0 = SQC0.shape[0]
print("Number of nonzero SQC entries:", num0)

term_types = []
goal_con = {}
# All Terms
print("All Terms")
for i1 in range(dim):
	for i2 in range(i1,dim):
		for i3 in range(i2,dim):
			for i4 in range(i3,dim):
				if SQC[i1,i2,i3,i4] != 0:
					print(index_type[i1], index_type[i2], index_type[i3], index_type[i4],"\t|", i1, i2, i3, i4,"\t|", to_string(basis[i1])[4:], to_string(basis[i2])[4:], to_string(basis[i3])[4:], to_string(basis[i4])[4:],"\t|",SQC[i1,i2,i3,i4])
					goal_con[str(i1) + " " + str(i2) + " " + str(i3) + " " + str(i4)] = SQC[i1,i2,i3,i4]
					term_name = index_type[i1] + index_type[i2] + index_type[i3] + index_type[i4]
					if term_name not in term_types:
						term_types.append(term_name)

print("Term Types:", term_types)
#print(goal_con)

# Special Arrays for computation
bindex = np.zeros(full_dim,dtype=int) # bsign[i]*basis[bindex[i]] = [0,...,0,1,0..] (1 in the i position)
bsign = np.zeros(full_dim,dtype=int)
for i in range(full_dim):
	temp = np.zeros(full_dim)
	temp[i] = 1
	for j in range(dim):
		dot = np.dot(dbasis[j],temp)
		if dot != 0:
			bindex[i] = j
			bsign[i] = np.real(dot)
print(np.array([i for i in range(25)]))
print(bindex)
print(bsign)

# Finding contributions of all possible index contractions
term_cons = {}

# Conformal:
# DDDD:
term_cons["D D D D"] = find_contribution(["D","D","D","D"],[1,1,1,1],[0,1,2,3],[0,1,2,3])
# DDMM:
term_cons["D D M_0^1 M_1^0"] = find_contribution(["D","D","M","M"],[2,2,1,1],[2,3,0,1],[2,3,1,0])
# DDKP:
term_cons["D D K^01 P_01"] = find_contribution(["D","D","K","P"],[2,2,1,1],[2,3,0,0],[2,3,1,1])
# DMKP:
term_cons["D M_0^1 K^02 P_12"] = find_contribution(["D","M","K","P"],[2,2,2,1],[3,0,0,1],[3,1,2,2])
# MMMM:
term_cons["M_0^1 M_1^2 M_2^3 M_3^0"] = find_contribution(["M","M","M","M"],[2,2,2,2],[0,1,2,3],[1,2,3,0])
# MMKP:
term_cons["M_0^1 M_2^3 K^02 P_13"] = find_contribution(["M","M","K","P"],[2,2,2,2],[0,2,0,1],[1,3,2,3])
#term_cons["M_0^1 M_1^2 K^30 P_23"] = find_contribution(["M","M","K","P"],[2,2,2,2],[0,1,3,2],[1,2,0,3])
term_cons["M_0^1 M_1^0 K^23 P_23"] = find_contribution(["M","M","K","P"],[2,2,2,2],[0,1,2,2],[1,2,3,3])
# KKPP:
term_cons["K^01 K^23 P_01 P_23"] = find_contribution(["K","K","P","P"],[2,2,2,2],[0,2,0,2],[1,3,1,3])
term_cons["K^01 K^23 P_02 P_13"] = find_contribution(["K","K","P","P"],[2,2,2,2],[0,2,0,1],[1,3,2,3])
# Superconformal:
# DDRR:
term_cons["D D R_01 R_10"] = find_contribution(["D","D","R","R"],[N,N,1,1],[2,3,0,1],[2,3,1,0])
# DDQS:
term_cons["D D Q_01 S^0_1"] = find_contribution(["D","D","Q","S"],[2,N,1,1],[2,3,0,0],[2,3,1,1])
# DMQS: 
term_cons["D M_0^1 Q_12 S^0_2"] = find_contribution(["D","M","Q","S"],[2,2,N,1],[3,0,1,0],[3,1,2,2])
# DRQS:
term_cons["D R_01 Q_20 S^2_1"] = find_contribution(["D","R","Q","S"],[N,N,2,1],[3,0,2,2],[3,1,0,1])
# MMRR:
term_cons["M_0^1 M_1^0 R_23 R_32"] = find_contribution(["M","M","R","R"],[2,2,N,N],[0,1,2,3],[1,0,3,2])
# MMQS:
term_cons["M_0^1 M_1^0 Q_23 S^2_3"] = find_contribution(["M","M","Q","S"],[2,2,2,N],[0,1,2,2],[1,0,3,3])
term_cons["M_0^1 M_1^2 Q_23 S^0_3"] = find_contribution(["M","M","Q","S"],[2,2,2,N],[0,1,2,0],[1,2,3,3])
# MKQQ:
term_cons["M_0^1 K^02 Q_13 Q_23"] = find_contribution(["M","K","Q","Q"],[2,2,2,N],[0,0,1,2],[1,2,3,3])
#term_cons["M_0^1 K^02 Q_23 Q_13"] = find_contribution(["M","K","Q","Q"],[2,2,2,N],[0,0,2,1],[1,2,3,3])
# MPSS:
term_cons["M_0^1 P_12 S^0_3 S^2_3"] = find_contribution(["M","P","S","S"],[2,2,2,N],[0,1,0,2],[1,2,3,3])
#term_cons["M_0^1 P_12 S^2_3 S^0_3"] = find_contribution(["M","P","S","S"],[2,2,2,N],[0,1,2,0],[1,2,3,3])
# MRQS:
term_cons["M_0^1 R_23 Q_12 S^0_3"] = find_contribution(["M","R","Q","S"],[2,2,N,N],[0,2,1,0],[1,3,2,3])
# KPRR:
term_cons["K^01 P_01 R_23 R_32"] = find_contribution(["K","P","R","R"],[2,2,N,N],[0,0,2,3],[1,1,3,2])
# KPQS:
term_cons["K^01 P_01 Q_23 S^2_3"] = find_contribution(["K","P","Q","S"],[2,2,2,N],[0,0,2,2],[1,1,3,3])
term_cons["K^01 P_02 Q_13 S^2_3"] = find_contribution(["K","P","Q","S"],[2,2,2,N],[0,0,1,2],[1,2,3,3])
# KRQQ:
term_cons["K^01 R_23 Q_02 Q_13"] = find_contribution(["K","R","Q","Q"],[2,2,N,N],[0,2,0,1],[1,3,2,3])
# PRSS:
term_cons["P_01 R_23 S^0_2 S^1_3"] = find_contribution(["P","R","S","S"],[2,2,N,N],[0,2,0,1],[1,3,2,3])
# RRRR:
term_cons["R_01 R_10 R_23 R_32"] = find_contribution(["R","R","R","R"],[N,N,N,N],[0,1,2,3],[1,0,3,2])
term_cons["R_01 R_12 R_23 R_30"] = find_contribution(["R","R","R","R"],[N,N,N,N],[0,1,2,3],[1,2,3,0])
# RRQS:
term_cons["R_01 R_10 Q_23 S^2_3"] = find_contribution(["R","R","Q","S"],[N,N,2,N],[0,1,2,2],[1,0,3,3])
term_cons["R_01 R_12 Q_32 S^3_0"] = find_contribution(["R","R","Q","S"],[N,N,N,2],[0,1,3,3],[1,2,2,0])
# QQSS:
term_cons["Q_01 Q_23 S^0_1 S^2_3"] = find_contribution(["Q","Q","S","S"],[2,N,2,N],[0,2,0,2],[1,3,1,3])
term_cons["Q_01 Q_23 S^2_1 S^0_3"] = find_contribution(["Q","Q","S","S"],[2,N,2,N],[0,2,2,0],[1,3,1,3])
term_cons["Q_01 Q_21 S^0_3 S^2_3"] = find_contribution(["Q","Q","S","S"],[2,N,2,N],[0,2,0,2],[1,1,3,3])

terms = {}
for v in term_cons.values():
	for k in v.keys():
		terms[k] = 1
lterms = list(terms.keys())

print(lterms)
for k in goal_con.keys():
	if k not in lterms:
		print("Not In",k)

#print("R_01 R_12 Q_32 S^3_0", find_contribution(["R","R","Q","S"],[N,N,N,2],[0,1,3,3],[1,2,2,0]))

t2 = []
for val in term_cons.values():
	t = np.zeros(len(lterms), dtype=complex)
	for k,v in val.items():
		t[lterms.index(k)] = v
	t2.append(t)
A = np.transpose(np.array(t2))

t = np.zeros(len(lterms), dtype=complex)
for k,v in goal_con.items():
	t[lterms.index(k)] = v

#print(A)
#print(np.transpose([t]))
res = lsqr(A,t)
if res[1] == 1:
	print("Solution Found:")
	labels = list(term_cons.keys())
	ans = ""
	for i in range(len(labels)):
		ans += str(res[0][i]) + labels[i] + " + "
	print(ans)

# Print as Latex
latex = "C_{\\mathcal{N} ="+ str(N) + "}^{(4)}&= \\frac{1}{" + str(mfactors[N]) + "}\\text{ssym}\\Big["
inc = 0
for i in range(len(labels)):
	if res[0][i] != 0:
		if format_number(res[0][i])[:1] != "-" and i != 0:
			latex += "+ "
		latex += format_number(res[0][i]) + to_latex(labels[i]) + " "
		if inc % 4 == 3:
			latex += "\\\\ &"
		inc += 1
text_file = open("Output.txt", "w")
text_file.write(latex + "\\Big]")
text_file.close()
print("Time: ", time() - start) 
