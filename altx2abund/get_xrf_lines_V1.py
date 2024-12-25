from common_modules import *
import xraylib
import os
import numpy as np
from memo import *

def get_nist_cross_sections(at_no, ele_name, script_path):
    filename = f"{script_path}/data_constants/ffast/ffast_{int(at_no)}_{ele_name}.txt"#; Getting the attenuation coefficients from FFAST database
    columns= readcol(filename, format = 'D,F,F,F,F,F,F,F')

    return (np.hstack([columns[0], np.zeros(100-columns[0].shape[0])]),
            np.hstack([columns[3], np.zeros(100-columns[3].shape[0])]),
            np.hstack([columns[5], np.zeros(100-columns[5].shape[0])]),
            np.hstack([columns[4], np.zeros(100-columns[4].shape[0])]),
            np.hstack([columns[1], np.zeros(100-columns[4].shape[0])]),
            np.hstack([columns[2], np.zeros(100-columns[4].shape[0])]),
            )

# Define a function to fetch edge energy
def get_edge_energy(at_no, shells):
    return [xraylib.EdgeEnergy(at_no, shell) for shell in shells]


# Define a function to fetch fluorescent yields
def get_fluorescent_yields(at_no, shells):
    yields = []
    for shell in shells:
        try:
            yields.append(xraylib.FluorYield(at_no, shell))
        except:
            yields.append(0.0)
    return yields

# Define a function to fetch jump factors
def get_jump_factors(at_no, shells):
    factors = []
    for shell in shells:
        try:
            factors.append(xraylib.JumpFactor(at_no, shell))
        except:
            factors.append(0.0)
    return factors

# Define a function to calculate weighted averages for radiative rates and line energies
def calculate_weighted_averages(at_no, lines):
    radrates = np.zeros((len(lines)))
    energies = np.zeros((len(lines)))
    for i, line in enumerate(lines):
        try:
            radrates[i] = (xraylib.RadRate(at_no, line))
            energies[i] = (xraylib.LineEnergy(at_no, line))
        except:
            radrates[i] = (0.0)
            energies[i] = (0.0)

    allowed_indices = np.where(np.array(radrates) > 0)
    if len(allowed_indices[0]) > 0:
        # print((np.array(radrates)[allowed_indices]).any())
        weighted_energy = np.sum(
            np.array(radrates)[allowed_indices] * np.array(energies)[allowed_indices]
        ) / np.sum(np.array(radrates)[allowed_indices])
        total_radrate = np.sum(np.array(radrates)[allowed_indices])
    else:
        weighted_energy, total_radrate = 0.0, 0.0

    return weighted_energy, total_radrate

# Main function to get XRF lines
@memoize_last
def get_xrf_lines(at_no, k_shell, k_lines, l1_shell, l1_lines, l2_shell, l2_lines, l3_shell, l3_lines):
    no_elements = np.size(at_no)
    script_path = os.path.dirname(os.path.abspath(__file__))

    # Initialize arrays
    edgeenergy = np.zeros((no_elements, 5))
    fluoryield = np.zeros((no_elements, 5))
    jumpfactor = np.zeros((no_elements, 5))
    radrate = np.zeros((no_elements, 5))
    lineenergy = np.zeros((no_elements, 5))
    energy_nist = np.zeros((no_elements, 100))
    photoncs_nist = np.zeros((no_elements, 100))
    totalcs_nist = np.zeros((no_elements, 100))
    
    scatteringcs_nist = np.zeros((no_elements, 100))
    aff1 = np.zeros((no_elements, 100))
    aff2 = np.zeros((no_elements, 100))

    elename_string = []

    # (atomic_number_list, kalpha_list, ele_list) = readcol(
    #     f"{script_path}/data_constants/kalpha_be_density_kbeta.txt", format="I,F,A"
    # )

    (atomic_number_list, kalpha_list, ele_list, be_list, density_list, kbeta_list) = readcol('data_constants/kalpha_be_density_kbeta.txt', format='I,F,A,F,F,F')

    # print("BBBBBBBB")
    # print(atomic_number_list)


    for i in range(no_elements):
        element_index = np.where(atomic_number_list == at_no[i])
        element_name = ele_list[element_index]
        elename_string.append(element_name)


        # Fetch NIST cross-sections
        energy_nist[i, :], photoncs_nist[i, :], totalcs_nist[i, :], scatteringcs_nist[i,:], aff1[i,:], aff2[i,:] = get_nist_cross_sections(
            str(int(at_no[i])).strip(), element_name[0], script_path
        )

        # Edge energy, fluorescent yields, jump factors

        edgeenergy[i] = get_edge_energy(at_no[i], [k_shell, k_shell, l1_shell, l2_shell, l3_shell])
        fluoryield[i] = get_fluorescent_yields(at_no[i], [k_shell, k_shell, l1_shell, l2_shell, l3_shell])
        jumpfactor[i] = get_jump_factors(at_no[i], [k_shell, k_shell, l1_shell, l2_shell, l3_shell])

        # Radiative rates and line energies
        lineenergy[i, 0], radrate[i, 0] = calculate_weighted_averages(at_no[i], k_lines[3:8])  # kbeta
        lineenergy[i, 1], radrate[i, 1] = calculate_weighted_averages(at_no[i], k_lines[:3])  # kalpha
        lineenergy[i, 2], radrate[i, 2] = calculate_weighted_averages(at_no[i], l1_lines)  # l1
        lineenergy[i, 3], radrate[i, 3] = calculate_weighted_averages(at_no[i], l2_lines)  # l2
        lineenergy[i, 4], radrate[i, 4] = calculate_weighted_averages(at_no[i], l3_lines)  # l2


    # Create a structure to return the data
    return Xrf_Lines(edgeenergy, fluoryield, jumpfactor, radrate, lineenergy, energy_nist, photoncs_nist, totalcs_nist, scatteringcs_nist, elename_string, aff1, aff2)
