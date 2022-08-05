import pybullet as p
import pybullet_data
import os

class Plane:
    def __init__(self, client):
        # f_name = os.path.dirname(__file__) + "/../../_asset/_general/simpleplane.urdf"  # file in the '_asset' folder
        p.loadURDF(fileName=pybullet_data.getDataPath() + "/plane.urdf",
                   basePosition=[0, 0, 0],
                   physicsClientId=client)