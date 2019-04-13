import numpy as np
from urdf_parser_py.urdf import URDF


class RobotParam:
    """
    unique params associated with a specific robot.
    """
    def __init__(self, urdf_path, mu=0.7):

        # parse urdf
        urdf_model = URDF.from_xml_file(urdf_path)

        lfoot_link_name = "LFoot"
        rfoot_link_name = "RFoot"
        lfoot_link = [link for link in urdf_model.links if link.name == lfoot_link_name][0]
        rfoot_link = [link for link in urdf_model.links if link.name == rfoot_link_name][0]

        self.foot_size = lfoot_link.collision.geometry.size
        self.foot_sole_position = np.array([lfoot_link.collision.origin.xyz[0],
                                            lfoot_link.collision.origin.xyz[1],
                                            lfoot_link.collision.origin.xyz[2]-self.foot_size[2]/2.0])
        print("foot_size: ", self.foot_size)
        print("foot_sole_position: ", self.foot_sole_position)

        # zmp constraints
        dx_min, dx_max = -self.foot_size[0] / 2.0, self.foot_size[0] / 2.0
        dy_min, dy_max = -self.foot_size[1] / 2.0, self.foot_size[1] / 2.0
        self.zmpConsSpatialForce = np.array([[-1, 0, 0, 0, 0, dy_min],
                                             [1, 0, 0, 0, 0, -dy_max],
                                             [0, 1, 0, 0, 0, dx_min],
                                             [0, -1, 0, 0, 0, -dx_max]])

        # friction constraints
        self.mu = mu
        self.frictionConsSpatialForce = np.array([[0, 0, 0, 1, 0, -self.mu],
                                                  [0, 0, 0, -1, 0, -self.mu],
                                                  [0, 0, 0, 0, 1, -self.mu],
                                                  [0, 0, 0, 0, -1, -self.mu]])

        # unilateral constraints
        self.unilateralConsSpatialForce = np.array([[0, 0, 0, 0, 0, -1]])

        # GRF constraints: zmp constraints + friction constraints + unilateral constraints
        self.SpatialForceCons = np.vstack((self.zmpConsSpatialForce,
                                           self.frictionConsSpatialForce,
                                           self.unilateralConsSpatialForce))
