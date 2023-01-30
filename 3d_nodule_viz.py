import os

try:
    from skimage.measure import marching_cubes
except ImportError:
    # Old version compatible since marching_cubes replaced with marchin_cubes_lewiner in skimage 0.19.0
    from skimage.measure import marching_cubes_lewiner as marching_cubes

import numpy as np
# import pylidc as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

viz3dbackends = ['matplotlib', 'mayavi']


def find_nodules(scan_obj):
    anns = scan_obj.annotations
    masks = [ann.boolean_mask(pad=[(1, 1), (1, 1), (1, 1)]) for ann in anns if ann.calcification > 4]
    return masks


def nodule_3d_viz(nodule_mask, title, edgecolor='0.2', cmap='cool', step=1, figsize=(5, 5), backend='matplotlib'):
    # Pad to cap the ends for masks that hit the edge.
    if backend not in viz3dbackends:
        raise ValueError("backend should be in %s." % viz3dbackends)

    if backend == 'matplotlib':
        if cmap not in plt.cm.cmap_d.keys():
            raise ValueError("Invalid `cmap`. See `plt.cm.cmap_d.keys()`.")

    mask = nodule_mask  # boolean_mask(pad=[(1,1), (1,1), (1,1)])
    rij = 0.625  # self.scan.pixel_spacing
    rk = 1  # self.scan.slice_thickness

    if backend == 'matplotlib':
        verts, faces, _, _ = marching_cubes(mask.astype(np.float), 0.5,
                                            spacing=(rij, rij, rk),
                                            step_size=step)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        t = np.linspace(0, 1, faces.shape[0])
        mesh = Poly3DCollection(verts[faces],
                                edgecolor=edgecolor,
                                facecolors=plt.cm.cmap_d[cmap](t))
        ax.add_collection3d(mesh)

        ceil = max(mask.shape) * rij
        ceil = int(np.round(ceil))

        ax.set_xlim(0, ceil)
        ax.set_xlabel('length (mm)')

        ax.set_ylim(0, ceil)
        ax.set_ylabel('length (mm)')

        ax.set_zlim(0, ceil)
        ax.set_zlabel('length (mm)')

        plt.title(title)
        plt.tight_layout()
        plt.show()
    elif backend == 'mayavi':
        try:
            from mayavi import mlab
            sf = mlab.pipeline.scalar_field(mask.astype(np.float))
            sf.spacing = [rij, rij, rk]
            mlab.pipeline.iso_surface(sf, contours=[0.5])
            mlab.show()
        except ImportError:
            print("Mayavi could not be imported. Is it installed?")


if "__main__" == __name__:
    nodule_path = r"3d_contours/"
    for nodule_name in os.listdir(nodule_path):
        title = "Patient ID: "+nodule_name[:-6]
        nodule = np.load(os.path.join(nodule_path, nodule_name))

        nodule_3d_viz(nodule, title, figsize=(10, 10))
