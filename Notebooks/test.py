import open3d as o3d
import sys
from pathlib import Path
import laspy

sys.path.append(str(Path().resolve().parent))
#from src import data_loader

if __name__ == "__main__":
    o3d.visualization.ExternalVisualizer()
    # This is from the docstring of the draw function of the ExternalVisualizer
    torus = o3d.geometry.TriangleMesh.create_torus()
    sphere = o3d.geometry.TriangleMesh.create_sphere()
    # create shortcut for draw
    draw = o3d.visualization.EV.draw
    draw([ {'geometry': sphere, 'name': 'sphere'},
           {'geometry': torus, 'name': 'torus'} ])