import numpy as np 
import math as m
from scipy.spatial.transform import Rotation as R
import scipy.linalg 
from scipy.optimize import fsolve

# ----------------------------
# Global Constants
# ----------------------------

# Conversion: inches → millimeters
Inch2mm = 25.4

# Conversion: degrees → radians
Deg2Rad = m.pi / 180


# ----------------------------
# Basic Trigonometric Wrappers
# ----------------------------
# These wrappers make code easier to read when writing DH transformations.

def cos(theta):
    """Return cosine of an angle (input in radians)."""
    return m.cos(theta)

def sin(theta):
    """Return sine of an angle (input in radians)."""
    return m.sin(theta)


# ----------------------------
# Rotation Matrix to Euler Angles
# ----------------------------

def Rmat_to_theta(rot_matrix):
    """
    Convert a 3×3 rotation matrix into Euler angles (φ, θ, ψ) in degrees.
    
    The computation uses standard atan2 formulas to avoid singularities.
    The assumed convention is ZYX or equivalent (depending on usage context).
    
    Parameters
    ----------
    rot_matrix : np.ndarray (3×3)
        Rotation matrix to be converted.
    
    Returns
    -------
    tuple of float
        (psi, theta, phi) in degrees.
    """
    
    # Compute Euler angles using atan2 to handle quadrants properly
    psi = m.atan2(rot_matrix[1][2], rot_matrix[0][2])
    theta = m.atan2(
        m.cos(psi) * rot_matrix[0][2] + m.sin(psi) * rot_matrix[1][2],
        rot_matrix[2][2]
    )
    phi = m.atan2(
        -m.sin(psi) * rot_matrix[0][0] + m.cos(psi) * rot_matrix[1][0],
        -m.sin(psi) * rot_matrix[0][1] + m.cos(psi) * rot_matrix[1][1]
    )
    
    # Convert radians → degrees
    psi = m.degrees(psi)
    theta = m.degrees(theta)
    phi = m.degrees(phi)
    
    # Return Euler angles in (φ, θ, ψ) order (all in degrees)
    return psi, theta, phi


# ----------------------------
# Denavit-Hartenberg Transformation
# ----------------------------

def DHmodel_to_transformation(joint_angle, alpha, depth, link_length,
                              angle_unit_rad=True, length_unit_mm=True):
    """
    Construct a 4×4 homogeneous transformation matrix using
    Denavit–Hartenberg (DH) parameters.
    
    Parameters
    ----------
    joint_angle : float
        Joint angle θ (rotation about z-axis).
    alpha : float
        Link twist α (rotation about x-axis).
    depth : float
        Link offset d (translation along z-axis).
    link_length : float
        Link length a (translation along x-axis).
    angle_unit_rad : bool, optional (default=True)
        If False, convert from degrees → radians.
    length_unit_mm : bool, optional (default=True)
        If False, convert input lengths from inches → millimeters.
    
    Returns
    -------
    transformation : np.ndarray (4×4)
        Homogeneous transformation matrix corresponding to given DH parameters.
    """
    
    # Convert inches → millimeters if needed
    if not length_unit_mm:
        depth *= Inch2mm
        link_length *= Inch2mm
    
    # Convert degrees → radians if needed
    if not angle_unit_rad:
        joint_angle *= Deg2Rad
        alpha *= Deg2Rad
    
    # Build the homogeneous transformation matrix according to DH convention:
    #     Rot(z, θ) * Trans(z, d) * Trans(x, a) * Rot(x, α)
    transformation = np.array([
        [cos(joint_angle), -sin(joint_angle) * cos(alpha),  sin(joint_angle) * sin(alpha), link_length * cos(joint_angle)],
        [sin(joint_angle),  cos(joint_angle) * cos(alpha), -cos(joint_angle) * sin(alpha), link_length * sin(joint_angle)],
        [0,                 sin(alpha),                    cos(alpha),                     depth],
        [0,                 0,                             0,                              1]
    ])
    
    return transformation    


# ----------------------------
# Example usage 
# ----------------------------
# The following examples demonstrate how to use the function
# to build a transformation chain for multiple joints.
#
# print(DHmodel_to_transformation(0, -1.571, 169.770, 64.2))
# J1 = DHmodel_to_transformation(0, -1.571, 169.770, 64.2)
# J2 = DHmodel_to_transformation(-1.222, 0, 0, 305)
# J3 = DHmodel_to_transformation(1.588, 1.571, 0, 0)
# print(J1 @ J2 @ J3)
