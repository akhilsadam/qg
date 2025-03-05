import torch
import numpy as np

def backward_euler_t(non_linear_term,dt):
    return non_linear_term*dt


def CN2(linear_operator,input_field,dt):
    # Linear_operator is the object
    return 0.5*dt*linear_operator.apply(input_field),0.5*dt*Lc

def AB2_t(q_sol_1,q_sol_2,dt):
    return (3/2)*(dt)*jacobian_pq(q_sol_1)  - (1/2)*(dt)*jacobian_pq(q_sol_2)


def AB2_brinkman_t(q_sol_1,q_sol_2,dt):
    return (3/2)*(dt)*brinkman_penalty(q_sol_1,xi,penalty_coeff=5*dt)  - (1/2)*(dt)*brinkman_penalty(q_sol_2,xi,penalty_coeff=5*dt)