import numpy as np 
import math as m
from scipy.spatial.transform import Rotation as R
import scipy.linalg 
from scipy.optimize import fsolve

Inch2mm = 25.4
Deg2Rad = m.pi/180

def cos(theta):
    return m.cos(theta)
def sin(theta):
    return m.sin(theta)

def rotation_martrix(rx, ry, rz, angle_unit_rad = True):
    
    # degree to radian
    if not angle_unit_rad :
        rx *= Deg2Rad
        ry *= Deg2Rad
        rz *= Deg2Rad
    
    # calculate individual rotation matrix 
    rot_x = np.array([[1, 0, 0],
                      [0, cos(rx), -sin(rx)],
                      [0, sin(rx), cos(rx)]])
    
    rot_y = np.array([[cos(ry), 0, sin(ry)],
                      [0, 1, 0],
                      [-sin(ry), 0, cos(ry)]])

    rot_z = np.array([[cos(rz), -sin(rz), 0],
                      [sin(rz), cos(rz), 0],
                      [0, 0, 1]])

    # return z-y-x rotation matrix
    return rot_z @ rot_y @ rot_x

def Rmat_to_theta(rot_matrix):
    
    # calculate Euler angle using atan2 and tranform unit from radian to degree
    psi = m.atan2(rot_matrix[1][2],rot_matrix[0][2])
    cta = m.atan2( m.cos(psi)*rot_matrix[0][2] + m.sin(psi)*rot_matrix[1][2]  , rot_matrix[2][2] ) 
    phi = m.atan2( -m.sin(psi)*rot_matrix[0][0] + m.cos(psi)*rot_matrix[1][0]  , -m.sin(psi)*rot_matrix[0][1] + m.cos(psi)*rot_matrix[1][1] ) 
    psi = m.degrees(psi)
    cta = m.degrees(cta)
    phi = m.degrees(phi)
    
    # return Euler angle in φ, θ, ψ order and in degree
    return psi, cta, phi

def DHmodel_to_transformation(joint_angle, alpha, depth, link_length, angle_unit_rad = True, length_unit_mm = True):
    
    # if length unit is inch, convert inch to mm
    if not length_unit_mm :
        depth *= Inch2mm
        link_length *= Inch2mm
    
    # if angle unit is degree, convert degree to radian
    if not angle_unit_rad : 
        joint_angle *= Deg2Rad
        alpha *= Deg2Rad
    
    # create a transformation matrix using DH model parameters 
    transformation = np.array([[cos(joint_angle), -sin(joint_angle)*cos(alpha), sin(joint_angle)*sin(alpha), link_length*cos(joint_angle)], 
                               [sin(joint_angle), cos(joint_angle)*cos(alpha), -cos(joint_angle)*sin(alpha), link_length*sin(joint_angle)], 
                               [0, sin(alpha), cos(alpha), depth], 
                               [0, 0, 0 ,1]])
    
    return transformation    

#print(DHmodel_to_transformation(0, -1.571, 169.770, 64.2))
#J1 = DHmodel_to_transformation(0, -1.571, 169.770, 64.2)
#J2 = DHmodel_to_transformation(-1.222, 0, 0, 305)
#J3 = DHmodel_to_transformation(1.588, 1.571, 0, 0)

#print(J1@J2@J3)

#print(rotation_martrix(-51.1183, 96.7177, 91.1183, angle_unit_rad=False))
