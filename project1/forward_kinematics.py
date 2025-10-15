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

def Rmat_to_theta(rot_matrix):
    
    # calculate Euler angle using atan2 and tranform unit from radian to degree
    psi = m.atan2(rot_matrix[1][2],rot_matrix[0][2])
    theta = m.atan2( m.cos(psi)*rot_matrix[0][2] + m.sin(psi)*rot_matrix[1][2]  , rot_matrix[2][2] ) 
    phi = m.atan2( -m.sin(psi)*rot_matrix[0][0] + m.cos(psi)*rot_matrix[1][0]  , -m.sin(psi)*rot_matrix[0][1] + m.cos(psi)*rot_matrix[1][1] ) 
    psi = m.degrees(psi)
    theta = m.degrees(theta)
    phi = m.degrees(phi)
    
    # return Euler angle in φ, θ, ψ order and in degree
    return psi, theta, phi

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
