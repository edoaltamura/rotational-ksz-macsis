import os
import sys
import unyt
import h5py
import numpy as np
import pandas as pd
from mpi4py import MPI
from swiftsimio.visualisation.projection import scatter_parallel as scatter
from swiftsimio.visualisation.rotation import rotation_matrix_from_vector
from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths


# Make the register backend visible to the script
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
        )
    )
)

comm = MPI.COMM_WORLD
num_processes = comm.size
rank = comm.rank

from read import MacsisDataset
from register import Macsis

ksz_const = - unyt.thompson_cross_section / 1.16 / unyt.speed_of_light / unyt.proton_mass
tsz_const = unyt.thompson_cross_section * unyt.boltzmann_constant / 1.16 / \
            unyt.speed_of_light ** 2 / unyt.proton_mass / unyt.electron_mass


def rotate(coord: np.ndarray, angular_momentum_hot_gas: np.ndarray, tilt: str = 'x'):
    if tilt == 'x':
        rotation_matrix = rotation_matrix_from_vector(np.array([1, 0, 0], dtype=np.float))
    elif tilt == 'y':
        rotation_matrix = rotation_matrix_from_vector(np.array([0, 1, 0], dtype=np.float))
    elif tilt == 'z':
        rotation_matrix = rotation_matrix_from_vector(np.array([0, 0, 1], dtype=np.float))
    elif tilt == 'faceon':
        rotation_matrix = rotation_matrix_from_vector(angular_momentum_hot_gas)
    elif tilt == 'edgeon':
        rotation_matrix = rotation_matrix_from_vector(angular_momentum_hot_gas, axis='y')

    new_coord = np.matmul(rotation_matrix, coord.T).T
    return new_coord


def angular_momentum(halo, particle_type: str = 'gas'):
    # Switch the type of map between gas and DM
    if particle_type == 'gas':
        pt_number = '0'
    elif particle_type == 'dm':
        pt_number = '1'

    data = MacsisDataset(halo)

    # Read data
    coordinates = data.read_snapshot(f'PartType{pt_number}/Coordinates')
    masses = data.read_snapshot(f'PartType{pt_number}/Mass')
    velocities = data.read_snapshot(f'PartType{pt_number}/Velocity')

    # Remember that the largest FOF has index 1
    centre_of_potential = data.read_catalogue_subfindtab('FOF/GroupCentreOfPotential')[1]
    r500_crit = data.read_catalogue_subfindtab('FOF/Group_R_Crit500')[1]

    # Select ionised hot gas
    if particle_type == 'gas':
        temperatures = data.read_snapshot(f'PartType{pt_number}/Temperature')
        temperature_cut = np.where(temperatures > 1.e5)[0]
        coordinates = coordinates[temperature_cut]
        masses = masses[temperature_cut]
        velocities = velocities[temperature_cut]

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

    return angular_momentum_r500


def dump_to_hdf5_parallel():
    macsis = Macsis()
    with h5py.File(f'{macsis.output_dir}/properties.hdf5', 'w', driver='mpio', comm=comm) as f:

        # Retrieve all zoom handles in parallel (slow otherwise)
        data_handles = np.empty(0, dtype=np.object)
        for zoom_id in range(macsis.num_zooms):
            if zoom_id % num_processes == rank:
                print(f"Collecting metadata for process ({zoom_id:03d}/{macsis.num_zooms - 1})...")
                data_handles = np.append(data_handles, macsis.get_zoom(zoom_id).get_redshift(-1))

        zoom_handles = comm.allgather(data_handles)
        zoom_handles = np.concatenate(zoom_handles).ravel()
        zoom_handles = zoom_handles[~pd.isnull(zoom_handles)]
        sort_keys = np.argsort(np.array([int(z.run_name[:-4]) for z in zoom_handles]))
        zoom_handles = sort_keys[sort_keys]

        if rank == 0:
            print([data_handle.run_name for data_handle in data_handles])

        # Editing the structure of the file MUST be done collectively
        if rank == 0:
            print("Preparing structure of the file (collective operations)...")

        names = f.create_dataset("names", (macsis.num_zooms,), dtype=np.str)
        m_500crit = f.create_dataset("m_500crit", (macsis.num_zooms,), dtype=np.float)
        r_500crit = f.create_dataset("r_500crit", (macsis.num_zooms,), dtype=np.float)
        angular_momentum_hotgas_r500 = f.create_dataset(
            "angular_momentum_hotgas_r500", (macsis.num_zooms,), dtype=np.float
        )

        # Data assignment can be done through independent operations
        for zoom_id, data_handle in enumerate(zoom_handles):
            if zoom_id % num_processes == rank:
                print((
                    f"Rank {rank:03d} processing halo ({zoom_id:03d}/{macsis.num_zooms - 1}) | "
                    f"MACSIS name: {data_handle.run_name}"
                ))

                names[zoom_id] = data_handle.run_name
                m_500crit[zoom_id] = data_handle.read_catalogue_subfindtab('FOF/Group_M_Crit500')[1]
                r_500crit[zoom_id] = data_handle.read_catalogue_subfindtab('FOF/Group_R_Crit500')[1]
                angular_momentum_hotgas_r500[zoom_id] = angular_momentum(data_handle, particle_type='gas')


if __name__ == "__main__":

    dump_to_hdf5_parallel()