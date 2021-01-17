import os
import sys
import numpy as np
import unyt
from swiftsimio.visualisation.projection import scatter_parallel as scatter
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

# Make the register backend visible to the script
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
        )
    )
)

from read import MacsisDataset
from register import Macsis


ksz_const = - unyt.thompson_cross_section / 1.16 / unyt.speed_of_light / unyt.proton_mass
tsz_const = unyt.thompson_cross_section * unyt.boltzmann_constant / 1.16 / \
            unyt.speed_of_light ** 2 / unyt.proton_mass / unyt.electron_mass

macsis = Macsis()

halo_handle = macsis.get_zoom(0).get_redshift(-1)
data = MacsisDataset(halo_handle)

# Read data
coordinates = data.read_snapshot('PartType0/Coordinates')
densities = data.read_snapshot('PartType0/Density')
masses = data.read_snapshot('PartType0/Mass')
velocities = data.read_snapshot('PartType0/Velocity')
temperatures = data.read_snapshot('PartType0/Temperature')

centre_of_potential = data.read_catalogue_subfindtab('/FOF/GroupCentreOfPotential')
r500_crit = data.read_catalogue_subfindtab('/FOF/Group_R_Crit500')
m500_crit = data.read_catalogue_subfindtab('/FOF/Group_M_Crit500')

# Select ionised hot gas
temperature_cut = np.where(temperatures > 1.e5)[0]

coordinates = coordinates[temperature_cut]
densities = densities[temperature_cut]
masses = masses[temperature_cut]
velocities = velocities[temperature_cut]
temperatures = temperatures[temperature_cut]

# Rescale coordinates to CoP
for i in range(3):
    coordinates[:, i] -= centre_of_potential[i]

# Compute mean velocity inside R500
radial_dist = np.sqrt(
    coordinates[:, 0] ** 2 +
    coordinates[:, 1] ** 2 +
    coordinates[:, 2] ** 2
)

r500_mask = np.where(radial_dist < r500_crit)[0]
mean_velocity_r500 = np.sum(velocities[r500_mask] * masses[r500_mask]) / np.sum(masses[r500_mask])

print(mean_velocity_r500)