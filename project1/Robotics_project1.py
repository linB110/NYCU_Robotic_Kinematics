import numpy as np
import math
import pandas as pd
from forward_kinematics import DHmodel_to_transformation, Rmat_to_theta
import math as m

# test data (https://www.youtube.com/watch?v=FNuiNmoqaZM)
# kinematics_table = pd.DataFrame({
#     'joint': [1, 2, 3, 4, 5, 6],
#     'd'    : [169.770, 0, 0, 222.63, 0, 36.25],         
#     'a'    : [64.2, 305, 0, 0, 0, 0],
#     'alpha': [-1.571, 0, 1.571, -1.571, 1.571, 0],
#     'theta': [0, -1.222, 1.588, 0, 0.785, 0],                 
# })

theta1, theta2, length3, theta4, theta5, theta6 = map(float, input(
    "please enter theta(-160~160), theta2(-125~125), length3(-30~30), "
    "theta4(-140~140), theta5(-100~100), theta6(-260~260): ").split())

input_valid = [abs(theta1) <= 160, abs(theta2) <= 125, abs(length3) <= 30, abs(theta4) <= 140, abs(theta5) <= 100, abs(theta6) <= 260]
names = ["theta1", "theta2", "length3", "theta4", "theta5", "theta6"]
values = [theta1, theta2, length3, theta4, theta5, theta6]

# 輸出結果
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

#print(T_forward[0:3, 0:3])

print("[n, o, a, p] : \n")
print(T_forward)    

print("output : \n")
print(rx, ry, rz)

