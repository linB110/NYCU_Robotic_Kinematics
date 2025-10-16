import numpy as np
import math as m
import pandas as pd
from forward_kinematics import DHmodel_to_transformation, Rmat_to_theta
from inverse_kinematics import inverse_kinematics

# test data (https://www.youtube.com/watch?v=FNuiNmoqaZM)
# kinematics_table = pd.DataFrame({
#     'joint': [1, 2, 3, 4, 5, 6],
#     'd'    : [169.770, 0, 0, 222.63, 0, 36.25],         
#     'a'    : [64.2, 305, 0, 0, 0, 0],
#     'alpha': [-1.571, 0, 1.571, -1.571, 1.571, 0],
#     'theta': [0, -1.222, 1.588, 0, 0.785, 0],                 
# })

# limit of theta and length
variable_seq = ["θ1", "θ2", "d3", "θ4", "θ5", "θ6"]
limits = [160, 125, 30, 140, 100, 260]

theta1, theta2, length3, theta4, theta5, theta6 = map(float, input(
    "please enter theta(-160~160), theta2(-125~125), length3(-30~30), "
    "theta4(-140~140), theta5(-100~100), theta6(-260~260): ").split())

input_valid = [abs(theta1) <= limits[0], abs(theta2) <= limits[1], abs(length3) <= limits[2], abs(theta4) <= limits[3], abs(theta5) <= limits[4], abs(theta6) <= limits[5]]
names = ["theta1", "theta2", "length3", "theta4", "theta5", "theta6"]
values = [theta1, theta2, length3, theta4, theta5, theta6]

for name, value, valid in zip(names, values, input_valid):
    if valid:
        print(f"{name} = {value}")
    else:
        print(f"{name}  out of range!")
        
# test data (stanford arm)
kinematics_table = pd.DataFrame({
    'joint': [1, 2, 3, 4, 5, 6],
    'd'    : [0, 6.375, length3, 0, 0, 0],         
    'a'    : [0, 0, 0, 0, 0, 0],
    'alpha': [-90, 90, 0, -90, 90, 0],
    'theta': [theta1, theta2, 0, theta4, theta5, theta6],                 
})

depth = kinematics_table['d']
link_length = kinematics_table['a']
alpha = kinematics_table['alpha']
theta = kinematics_table['theta']

# create a identity for operation
T_forward = np.identity(4)

# get parameters from kinematics table and perform post-multiplication
for i in range(6):
    J_i = DHmodel_to_transformation(theta[i], alpha[i], depth[i], link_length[i], angle_unit_rad=False)
    T_forward = T_forward @ J_i

rx, ry, rz = Rmat_to_theta(T_forward[0:3, 0:3])
x, y, z = (T_forward[0:3, 3])

# forward kinematics result
print("[n, o, a, p] : ")
print(T_forward)
print("\n")    

print("Cartesian point (x, y, z, φ, θ, ψ) : ")
print(x, y, z, rx, ry, rz)
print("\n")    

# inverse kinematics result
vals = list(map(float, input(
    "please enter the cartesian point (Euler angle and x,y,z) : ").split()))
if len(vals) != 16:
    raise ValueError("You must enter exactly 16 numbers for a 4x4 matrix.")

T = np.array(vals).reshape((4, 4))

# make sure input of rotation matrix is orthogonal matrix
R06 = T[:3, :3]
U, _, Vt = np.linalg.svd(R06)
R_orthonormal = U @ Vt
T[:3, :3] = R_orthonormal

result = inverse_kinematics(kinematics_table, T)
for i in range(len(result)):
    print("solution " + str(i))
    print("corresponding variables (θ1, θ2, d3, θ4, θ5, θ6) : ")

    for j in range(6):
        if ( abs(result[i][j]) ) >= limits[j]:
            print(variable_seq[j] + " is out of range !")
    print(result[i])
    print("\n")

# forward and inverse kinematics can use for validate each other 
# test = np.array(T_forward)
# print("inverse solution")
# print(inverse_kinematics(kinematics_table, pos_matrix=test))
