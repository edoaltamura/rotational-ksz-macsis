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


def rotate_coordinates(coord: np.ndarray, angular_momentum_hot_gas: np.ndarray, tilt: str = 'z'):

    x, y, z = coord.T

    if tilt == 'y':
        new_coord = np.vstack((x, z, y)).T
    elif tilt == 'z':
        new_coord = np.vstack((x, y, z)).T
    elif tilt == 'x':
        new_coord = np.vstack((z, y, x)).T
    elif tilt == 'faceon':
        face_on_rotation_matrix = rotation_matrix_from_vector(angular_momentum_hot_gas)
        new_coord = np.einsum('ijk,ik->ij', face_on_rotation_matrix, coord)
    elif tilt == 'edgeon':
        edge_on_rotation_matrix = rotation_matrix_from_vector(angular_momentum_hot_gas, axis='y')
        print(edge_on_rotation_matrix)
        new_coord = np.einsum('ijk,ik->ij', edge_on_rotation_matrix, coord)

    return new_coord


def rotate_velocities(velocities: np.ndarray, angular_momentum_hot_gas: np.ndarray, tilt: str = 'z'):

    vx, vy, vz = velocities.T

    if tilt == 'y':
        new_vel = np.vstack((vx, vz, vy)).T
    elif tilt == 'z':
        new_vel = np.vstack((vx, -vy, vz)).T
    elif tilt == 'x':
        new_vel = np.vstack((-vz, -vy, vx)).T
    elif tilt == 'faceon':
        face_on_rotation_matrix = rotation_matrix_from_vector(angular_momentum_hot_gas)
        new_vel = np.einsum('ijk,ik->ij', face_on_rotation_matrix, velocities)
    elif tilt == 'edgeon':
        edge_on_rotation_matrix = rotation_matrix_from_vector(angular_momentum_hot_gas, axis='y')
        new_vel = np.einsum('ijk,ik->ij', edge_on_rotation_matrix, velocities)

    return new_vel


macsis = Macsis()

halo_handle = macsis.get_zoom(0).get_redshift(-1)
data = MacsisDataset(halo_handle)

# Read data
coordinates = data.read_snapshot('PartType0/Coordinates')
densities = data.read_snapshot('PartType0/Density')
masses = data.read_snapshot('PartType0/Mass')
velocities = data.read_snapshot('PartType0/Velocity')
temperatures = data.read_snapshot('PartType0/Temperature')

centre_of_potential = data.read_catalogue_subfindtab('FOF/GroupCentreOfPotential')[0]
r500_crit = data.read_catalogue_subfindtab('FOF/Group_R_Crit500')[0]
m500_crit = data.read_catalogue_subfindtab('FOF/Group_M_Crit500')[0]

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

mean_velocity_r500 = np.sum(velocities[r500_mask] * masses[r500_mask, None], axis=0) / np.sum(masses[r500_mask])
angular_momentum_r500 = np.sum(
    np.cross(coordinates[r500_mask], velocities[r500_mask] * masses[r500_mask, None]), axis=0
) / np.sum(masses[r500_mask])

velocities_rest_frame = velocities.copy()
for i in range(3):
    velocities_rest_frame[:, i] -= mean_velocity_r500[i]

# Rotate coordinates and velocities
coordinates_edgeon = rotate_coordinates(coordinates, angular_momentum_r500, tilt='edgeon')
velocities_rest_frame_edgeon = rotate_coordinates(velocities_rest_frame, angular_momentum_r500, tilt='edgeon')

compton_y = unyt.unyt_array(
    masses * velocities_rest_frame_edgeon[:, 2], 1.e10 * unyt.Solar_Mass * 1.e3 * unyt.km / unyt.s
) * ksz_const / unyt.unyt_quantity(1., unyt.Mpc) ** 2


print(angular_momentum_r500)