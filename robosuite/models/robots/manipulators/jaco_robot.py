import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion
DEFAULT_INIT_QPOS = np.array([3.192, 3.680, -0.000, 1.170, 0.050, 3.760, 3.142])
# Jaco Real home pose
REAL_INIT_QPOS = np.array([4.982, 2.841, 6.24, 0.758, 4.632, 4.512, 5.0244])
DOWN_INIT_QPOS = np.array([4.992, 3.680, -0.000, 1.170, 0.050, 3.760, 3.142])


class Jaco(ManipulatorModel):
    """
    Jaco is a kind and assistive robot created by Kinova

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0, init_qpos=REAL_INIT_QPOS):
        super().__init__(xml_path_completion("robots/jaco/robot.xml"), idn=idn)
        self.set_init_qpos(init_qpos)

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "JacoThreeFingerGripper"

    @property
    def default_controller_config(self):
        return "default_jaco"

    def set_init_qpos(self, init_qpos):
        self.init_qpos = init_qpos

    def init_qpos(self):
        ## default position
        #return np.array([3.192, 3.680, -0.000, 1.170, 0.050, 3.760, 3.142])
        # Jaco Real home pose
        #return np.array([4.942, 2.842, 0.0011, 0.758, 4.6368, 4.492, 5.0244])
        #return np.array([4.708, 2.619, 0.000, 0.521, 6.279, 3.714, 3.14])
        #np.array([4.872, 3.055, 0.5, 1.294, 4.497, 4.343, 5.0])
        return self.init_qpos


    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
