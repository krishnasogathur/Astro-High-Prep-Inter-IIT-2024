'''
;====================================================================================================
;                              X2ABUNDANCE
;
; Package for determining elemental weight percentages from XRF line fluxes
;
; Algorithm developed by P. S. Athiray (Athiray et al. 2015)
; Codes in IDL written by Netra S Pillai
; Codes for XSPEC localmodel developed by Ashish Jacob Sam and Netra S Pillai
;
; Developed at Space Astronomy Group, U.R.Rao Satellite Centre, Indian Space Research Organisation
;
;====================================================================================================

This file contains the common functions/methods and class definitions used in the repository

The major library used for this repository is the open-source xraylib at https://github.com/tschoonj/xraylib which is available to be added as dependacy only via conda (Anaconda/miniconda). Thus it is essential to use conda virtual enviornment to execute the code.

The repository uses the following dependacies
	xraylib:    	installed by running conda install xraylib
	numpy: 		installed by running conda install numpy
	astropy: 	installed by running conda install astropy
'''
from typing import Any
import numpy as np
import glob

# @profile
def n_elements(array)->int:
		return np.size(array)

# @profile
def dblarr(*args:int) ->Any:
	return np.zeros(tuple(args))

# @profile
def total(MyList:list) -> Any:
    return np.sum(MyList)

# @profile
def ChangeEveryElement(function,array:list) -> None:
        array[:] = list(map(function, array))

# @profile
def readcol(filename: str, format: str = None) -> tuple:
    rowformat = format.split(',') if format else []
    rowformat = [x.capitalize() for x in rowformat]
    TupleOfLists = [[] for _ in rowformat]

    for filenames in glob.glob(filename):
        with open(filenames, 'r') as f:
            for line in f:
                inputstring = line.split()
                if len(inputstring) == len(rowformat) and inputstring[0]:
                    try:
                        for i, val in enumerate(inputstring):
                            if rowformat[i] in {'B', 'I', 'L', 'Z'}:
                                TupleOfLists[i].append(int(val))
                            elif rowformat[i] in {'D', 'F'}:
                                TupleOfLists[i].append(float(val))
                            elif rowformat[i] != 'X':
                                TupleOfLists[i].append(val)
                    except:
                        continue

    return tuple(np.array(lst) for lst in TupleOfLists)


def strarr(count:int)->list:
	return ['']*count

# def readcol(filename: str, format: str = None) -> tuple:
#     # Prepare row format
#     rowformat = format.split(',') if format else []
#     rowformat = [x.capitalize() for x in rowformat]

#     # Initialize lists to store columns based on format
#     TupleOfLists = [[] for _ in rowformat]

#     # Read and process lines
#     for filenames in glob.glob(filename):
#         with open(filenames, 'r') as f:
#             lines = f.readlines()

#         # Split lines and assign values directly based on rowformat
#         for line in lines:
#             inputstring = line.split()
#             if len(inputstring) == len(rowformat):
#                 for i, (val, fmt) in enumerate(zip(inputstring, rowformat)):
#                     if fmt == 'X':
#                         continue  # Skip "X" columns
#                     elif fmt in {'B', 'I', 'L', 'Z'}:
#                         TupleOfLists[i].append(int(val))
#                     elif fmt in {'D', 'F'}:
#                         TupleOfLists[i].append(float(val))
#                     else:
#                         TupleOfLists[i].append(val)

#     # Convert each list to a numpy array and return as a tuple
#     return tuple(np.array(lst) for lst in TupleOfLists if lst)


# 	TupleOfNpArray=[]
# 	for listitem in TupleOfLists:
# 		TupleOfNpArray.append(np.array(listitem))
# 	return tuple(TupleOfNpArray)


# @profile
def SortVectors(TupleOfArrays: tuple, Reverse: bool = False) -> tuple:
    stacked = np.column_stack(TupleOfArrays)
    sorted_data = stacked[np.lexsort(stacked.T[::-1] if Reverse else stacked.T)]
    return tuple(sorted_data[:, i] for i in range(sorted_data.shape[1]))



# @profile
def PRODUCT(array:list)->float:
    return np.prod(array)

# @profile
def Tuple2String(my_row: tuple) -> str:
    return ' '.join(map(str, my_row))
# def Tuple2String(MyRow:tuple)->str:
# 	output=""
# 	for x in MyRow:
# 		output=output+x.__str__()+" "
# 	return output

# @profile
def file_lines(filename: str) -> int:
    with open(filename) as f:
        return sum(1 for _ in f)
# def file_lines(filename:str)->int:
# 	return sum(1 for _ in open(filename))

# @profile
def fix(stuff):
	typeof = type(stuff)
	if(typeof is float):
		return int(stuff)
	elif typeof is list:
		intlist = []
		for f in stuff:
			intlist.append(int(f))
		return intlist
	else:#numpy array
            return stuff.astype(int)


# @profile
def array_indices_custom(np_array, *location):
    # Flatten and handle list input once
    if isinstance(location[0], list):
        location = location[0]

    # Get the shape of the array
    array_shape = np_array.shape

    # Use NumPy's built-in unravel_index for efficient index calculation
    indices = np.unravel_index(location, array_shape)

    # Return as a list of arrays
    return np.column_stack(indices)
# def array_indices_custom(NpArray, *location)->list:
# 	ReturnValue = np.array([])
# 	if (type(location[0]) is list):
# 		location=list(location[0])
# 	else:
# 		location=list(location)
# 	for CurrentLocation in location:
# 		ArrayShape = np.shape(NpArray)
# 		prod = np.prod(np.shape(NpArray))
# 		for i in range(0,ArrayShape.__len__()):
# 			item=ArrayShape[i]
# 			prod=prod/item
# 			ArrayShape[i]=(CurrentLocation//prod)%item
# 		ReturnValue.append(np.array(ArrayShape))
# 	return ReturnValue

# @profile


class Xrf_Lines:
    def __init__(self, Edgeenergy, FluorYield, jumpfactor, RadRate, LineEnergy,Energy_nist, photoncs_nist, totalcs_nist,scatteringcs_nist,elename_string, aff1 = None, aff2 =None)  -> None:
        self.fluoryield = FluorYield
        self.edgeenergy = Edgeenergy
        self.jumpfactor = jumpfactor
        self.radrate = RadRate
        self.lineenergy = LineEnergy
        self.energy_nist=Energy_nist
        self.photoncs_nist = photoncs_nist
        self.totalcs_nist = totalcs_nist
        self.scatteringcs_nist = scatteringcs_nist
        self.elename_string = elename_string
        self.aff1 = aff1
        self.aff2 = aff2
            


class Const_Xrf:
	def __init__(self,total_attn_at_line, total_attn, photoelec_attn, total_attn_mask = None) -> None:
		self.total_attn_at_line=total_attn_at_line
		self.total_attn = total_attn
		# self.total_attn_rayleigh = total_attn_rayleigh
		self.photoelec_attn=photoelec_attn
		self.total_attn_mask = total_attn_mask
		# self.total_attn_xspec_rayleigh = total_attn_xspec_rayleigh

class Xrf_Struc:
	def __init__(self,primary,secondary,total) -> None:
		self.primary_xrf=primary
		self.secondary_xrf=secondary
		self.total_xrf=total

class Scat_struc:
	def __init__(self,Coh,Incoh,Total) -> None:
		self.i_total=Total
		self.i_coh=Coh
		self.i_incoh=Incoh
