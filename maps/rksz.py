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


def rotate_coordinates(coord: np.ndarray, angular_momentum_hot_gas: np.ndarray, tilt: str = 'y'):
    x, y, z = coord.T

    if tilt == 'y':
        new_coord = np.vstack((x, z, y)).T
    elif tilt == 'z':
        new_coord = np.vstack((x, y, z)).T
    elif tilt == 'x':
        new_coord = np.vstack((z, y, x)).T
    elif tilt == 'faceon':
        face_on_rotation_matrix = rotation_matrix_from_vector(angular_momentum_hot_gas)
        new_coord = np.matmul(face_on_rotation_matrix, coord.T).T
    elif tilt == 'edgeon':
        edge_on_rotation_matrix = rotation_matrix_from_vector(angular_momentum_hot_gas, axis='y')
        new_coord = np.matmul(edge_on_rotation_matrix, coord.T).T
    return new_coord


def rotate_velocities(vel: np.ndarray, angular_momentum_hot_gas: np.ndarray, tilt: str = 'z'):
    vx, vy, vz = vel.T

    if tilt == 'z':
        new_vel = np.vstack((vx, vz, -vy)).T
    elif tilt == 'y':
        new_vel = np.vstack((vx, -vy, vz)).T
    elif tilt == 'x':
        new_vel = np.vstack((-vz, -vy, vx)).T
    elif tilt == 'faceon':
        face_on_rotation_matrix = rotation_matrix_from_vector(angular_momentum_hot_gas)
        new_vel = np.matmul(face_on_rotation_matrix, vel.T).T
    elif tilt == 'edgeon':
        edge_on_rotation_matrix = rotation_matrix_from_vector(angular_momentum_hot_gas, axis='x')
        new_vel = np.matmul(edge_on_rotation_matrix, vel.T).T
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
smoothing_lengths = data.read_snapshot('PartType0/SmoothingLength')

# Remember that the largest FOF has index 1
centre_of_potential = data.read_catalogue_subfindtab('FOF/GroupCentreOfPotential')[1]
r500_crit = data.read_catalogue_subfindtab('FOF/Group_R_Crit500')[1]
m500_crit = data.read_catalogue_subfindtab('FOF/Group_M_Crit500')[1]

# Select ionised hot gas
temperature_cut = np.where(temperatures > 1.e5)[0]

coordinates = coordinates[temperature_cut]
densities = densities[temperature_cut]
masses = masses[temperature_cut]
velocities = velocities[temperature_cut]
temperatures = temperatures[temperature_cut]
smoothing_lengths = smoothing_lengths[temperature_cut]

# Rescale coordinates to CoP
coordinates[:, 0] -= centre_of_potential[0]
coordinates[:, 1] -= centre_of_potential[1]
coordinates[:, 2] -= centre_of_potential[2]

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
velocities_rest_frame[:, 0] -= mean_velocity_r500[0]
velocities_rest_frame[:, 1] -= mean_velocity_r500[1]
velocities_rest_frame[:, 2] -= mean_velocity_r500[2]

# Rotate coordinates and velocities
coordinates_edgeon = rotate_coordinates(coordinates, angular_momentum_r500, tilt='edgeon')
velocities_rest_frame_edgeon = rotate_velocities(velocities_rest_frame, angular_momentum_r500, tilt='edgeon')

# Rotate angular momentum vector for cross check
angular_momentum_r500_rotated = rotate_coordinates(
    angular_momentum_r500 / np.linalg.norm(angular_momentum_r500), angular_momentum_r500, tilt='edgeon'
) * r500_crit / 2

compton_y = unyt.unyt_array(
    masses * velocities_rest_frame_edgeon[:, 2], 1.e10 * unyt.Solar_Mass * 1.e3 * unyt.km / unyt.s
) * ksz_const / unyt.unyt_quantity(1., unyt.Mpc) ** 2
compton_y = compton_y.value

# Restrict map to 2*R500
spatial_filter = np.where(
    (np.abs(coordinates_edgeon[:, 0]) < r500_crit / 2) &
    (np.abs(coordinates_edgeon[:, 1]) < r500_crit / 2) &
    (np.abs(coordinates_edgeon[:, 2]) < r500_crit / 2)
)[0]

coordinates_edgeon = coordinates_edgeon[spatial_filter]
velocities_rest_frame_edgeon = velocities_rest_frame_edgeon[spatial_filter]
smoothing_lengths = smoothing_lengths[spatial_filter]
compton_y = compton_y[spatial_filter]

# Make map using swiftsimio
x = (coordinates_edgeon[:, 0] - coordinates_edgeon[:, 0].min()) / (
        coordinates_edgeon[:, 0].max() - coordinates_edgeon[:, 0].min())
y = (coordinates_edgeon[:, 1] - coordinates_edgeon[:, 1].min()) / (
        coordinates_edgeon[:, 1].max() - coordinates_edgeon[:, 1].min())
h = smoothing_lengths / (coordinates_edgeon.max() - coordinates_edgeon.min()) ** 2

# Gather and handle coordinates to be processed
x = np.asarray(x, dtype=np.float64)
y = np.asarray(y, dtype=np.float64)
m = np.asarray(compton_y, dtype=np.float32)
h = np.asarray(h, dtype=np.float32)
smoothed_map = scatter(x=x, y=y, m=m, h=h, res=128).T
print(compton_y.max(), compton_y.min(), smoothing_lengths.max(), smoothed_map.min())
# smoothed_map = np.ma.masked_where(np.log10(np.abs(smoothed_map)) < -20, smoothed_map)
vlim = np.abs(smoothed_map).max()
plt.imshow(
    smoothed_map,
    norm=SymLogNorm(linthresh=0.01, linscale=1, vmin=-vlim, vmax=vlim),
    cmap="PRGn",
    origin="lower",
    extent=(
        coordinates_edgeon[:, 0].min(),
        coordinates_edgeon[:, 0].max(),
        coordinates_edgeon[:, 1].min(),
        coordinates_edgeon[:, 1].max(),
    )
)
plt.plot([0, angular_momentum_r500_rotated[0]], [0, angular_momentum_r500_rotated[1]], marker='o')
plt.axis('off')
plt.show()

compton_y_DC = smoothed_map - np.mean(smoothed_map)
spectrum = np.fft.fftshift(np.fft.fft2(smoothed_map))
spectrum_wo_DC = np.fft.fftshift(np.fft.fft2(compton_y_DC))

freqx = np.fft.fftshift(np.fft.fftfreq(128, 1))  # q(n, d=1.0)
freqy = np.fft.fftshift(np.fft.fftfreq(128, 1))
fX, fY = np.meshgrid(freqx, freqy)

plt.pcolormesh(fX, fY, np.abs(spectrum))
plt.show()
plt.pcolormesh(fX, fY, np.abs(spectrum_wo_DC))
plt.show()
