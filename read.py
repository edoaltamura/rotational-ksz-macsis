import h5py
import inspect

from register import Redshift


class MacsisDataset(object):
    part_type = {
        'gas': '/PartType0',
        'dark_matter': '/PartType1',
        'stars': '/PartType4',
        'black_holes': '/PartType5',
    }

    snapshot_file: h5py.File
    catalogue_grouptab_file: h5py.File
    catalogue_subfindtab_file: h5py.File
    catalogue_particles_file: h5py.File

    def __init__(self, redshift_obj: Redshift):

        self.run_name = redshift_obj.run_name
        self.scale_factor = redshift_obj.scale_factor
        self.a = redshift_obj.a
        self.redshift = redshift_obj.redshift
        self.z = redshift_obj.z
        self.snapshot_path = redshift_obj.snapshot_path
        self.catalogue_grouptab_path = redshift_obj.catalogue_grouptab_path
        self.catalogue_subfindtab_path = redshift_obj.catalogue_subfindtab_path
        self.catalogue_particles_path = redshift_obj.catalogue_particles_path

        self.snapshot_file = h5py.File(self.snapshot_path, 'r')
        self.catalogue_grouptab_file = h5py.File(self.catalogue_grouptab_path, 'r')
        self.catalogue_subfindtab_file = h5py.File(self.catalogue_subfindtab_path, 'r')
        self.catalogue_particles_file = h5py.File(self.catalogue_particles_path, 'r')

    def read_dataset(self, file_type: str, dataset_path: str):
        data = getattr(self, file_type)[dataset_path][:]
        data *= self.a ** data.attrs['aexp-scale-exponent']
        data *= self.snapshot_file['Header'].attrs['HubbleParam'] ** data.attrs['h-scale-exponent']
        return data

    def read_snapshot(self, dataset_path: str):
        return self.read_dataset('snapshot_file', dataset_path)

    def read_catalogue_grouptab(self, dataset_path: str):
        return self.read_dataset('catalogue_grouptab_file', dataset_path)

    def read_catalogue_subfindtab(self, dataset_path: str):
        return self.read_dataset('scatalogue_subfindtab_file', dataset_path)

    def read_catalogue_particles(self, dataset_path: str):
        return self.read_dataset('catalogue_particles_file', dataset_path)


if __name__ == "__main__":
    from register import Macsis
    macsis = Macsis()
    for i in range(3):
        halo_handle = macsis.get_zoom(i).get_redshift(-1)
        data = MacsisDataset(halo_handle)
        print(data.read_snapshot('PartType1/Temperature'))