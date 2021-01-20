import os
import sys
import unyt
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

try:
    plt.style.use("mnras.mplstyle")
except:
    pass

Tcmb0 = 2.7255
projections = ['x', 'y', 'z', 'faceon', 'edgeon']
display_maps = dict()

with h5py.File(f'{Macsis().output_dir}/rksz_gas.hdf5', 'r') as f:
    for projection in projections:
        for i, halo in enumerate(f.keys()):
            dataset = f[f"{halo}/map_{projection}"][:]
            print((
                f"Merging map_{projection} from {halo} | "
                f"shape: {dataset.shape} | "
                f"size: {dataset.nbytes / 1024 / 1024} MB"
            ))
            if i == 0:
                display_maps[projection] = dataset
            else:
                display_maps[projection] += dataset

# smoothed_map = np.ma.masked_where(np.log10(np.abs(smoothed_map)) < -20, smoothed_map)
# Get maximum limits
max_list = []
for p in display_maps:
    max_list.append(np.abs(display_maps[p]).max())
vlim = max(max_list)

fig, axes = plt.subplots(nrows=1, ncols=len(display_maps))

for ax, projection, smoothed_map in zip(axes, projections, display_maps.values()):

    im = ax.imshow(
        smoothed_map,
        norm=SymLogNorm(linthresh=0.01, linscale=1, vmin=-vlim, vmax=vlim),
        cmap="PRGn",
        origin="lower",
        extent=(-1, 1, -1, 1),
    )
    # plt.plot([0, angular_momentum_r500_rotated[0]], [0, angular_momentum_r500_rotated[1]], marker='o')
    ax.set_axis_off()
    ax.set_title(f"Projection {projection}")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax, label=r'$\sum y_{ksz}$')
# fig.tight_layout()
plt.show()
plt.close()
