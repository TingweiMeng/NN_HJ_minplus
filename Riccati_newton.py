# This is the third example in the paper (see section 4.3)

import random
import numpy as np
import tensorflow as tf
import time
import scipy.io
import matplotlib.pyplot as plt 
from scipy.linalg import expm
import scipy.integrate as integrate
from scipy.integrate import ode

start = time.time()

Lx = 400
LP = 2*Lx
m = 2
ldim = 8
n = 2* ldim
t0 = 5.0
Ndata1d = 50
xleft = -6.0

plotSflag = False
plotxuflag = not plotSflag
if plotSflag:
    filenameS = 'newton_S{}d_t{:.3f}.png'.format(n, t0)
if plotxuflag:
    filenamex = 'newton_x{}d_t{}.png'.format(n, int(t0))
    filenamev = 'newton_v{}d_t{}.png'.format(n, int(t0))
    filenameu = 'newton_u{}d_t{}.png'.format(n, int(t0))

np.random.seed(1)
 
tf.reset_default_graph()

################# def M, A, B, Cx start ############################################################
alpinv = 1000 # 1/alp

# M = (O,O;O,I)/alp (Cpp in the paper)
M_const = np.zeros((n,n))
M_const[ldim:, ldim:] = np.eye(ldim) * alpinv

# A = (O,I;O,O)
A_const = np.zeros((n,n))
A_const[0:ldim, ldim:] = np.eye(ldim)

# Bt = (O,I)
Bt_const = np.zeros((ldim, n))
Bt_const[:,ldim:] = np.eye(ldim)

# Cx is identity
Cx_const = np.eye(n)

# const of zr
Czr = 5.0
################# def M, A, B, Cx end   ###########################################################

################# def initial cond start #################################################
D_data = np.tile(np.eye(n), (1,m,1,1)) * 0.05 / ldim # I/160
a_data = np.zeros((1,m,n,1))
b_data = np.zeros((1,m,1,1))

a_data[0,0,0,0] = -0.1/ldim
b_data[0,0,0,0] = 0.1/ldim
a_data[0,1,0,0] = 0.1/ldim
b_data[0,1,0,0] = 0.1/ldim

################# def initial cond end ###################################################

################# def input x,t start #################
# construct x, t
xright = -xleft
#x_grid = np.random.rand(Ndata, n) / np.sqrt(n)
if n==1:
    Ndata = Ndata1d
    x_grid = np.linspace(xleft, xright, Ndata)
    x_data = np.reshape(x_grid, (Ndata,1,1,1))
else:
    if plotxuflag:
        Ndata = Ndata1d
        x_grid = np.reshape(np.linspace(xleft, xright, Ndata), (Ndata,1,1,1))
        x_data1 = np.concatenate((x_grid, np.zeros((Ndata,1,ldim-1,1))), axis = 2)
        x_data2 = np.concatenate((np.zeros((Ndata,1,1,1))+Czr, np.zeros((Ndata,1,ldim-1,1))), axis = 2)
        x_data = np.concatenate((x_data1, x_data2), axis=2)
    else:
        nx = Ndata1d
        ny = Ndata1d
        Ndata = nx * ny
        x = np.linspace(xleft, xright, nx)
        y = np.linspace(xleft, xright, ny)
        x_data = np.zeros((Ndata, 1,n,1))
        xv, yv = np.meshgrid(x, y)
        xv_arr = np.reshape(xv, (Ndata))
        yv_arr = np.reshape(yv, (Ndata))
        x_data[:,0,0,0] = xv_arr
        x_data[:,0,ldim,0] = yv_arr
t_data = np.reshape(t0, (1,1,1,1))
################# def input x,t end #####################



# for HJ PDE
A_param = tf.Variable(np.tile(np.reshape(A_const, (1,1,n,n)), (1,m,1,1)), dtype=tf.float64)
M_param = tf.Variable(np.tile(np.reshape(M_const, (1,1,n,n)), (1,m,1,1)), dtype=tf.float64) 
Cx_param = tf.Variable(np.tile(np.reshape(Cx_const, (1,1,n,n)), (1,m,1,1)), dtype=tf.float64)

# for trajectory
Bt_param = tf.Variable(np.tile(np.reshape(Bt_const, (1,1,ldim,n)),(Ndata,1,1,1)), dtype=tf.float64)
Atraj_param = tf.Variable(np.tile(np.reshape(A_const, (1,1,n,n)),(Ndata,1,1,1)), dtype=tf.float64)
Mtraj_param = tf.Variable(np.tile(np.reshape(M_const, (1,1,n,n)),(Ndata,1,1,1)), dtype=tf.float64) 

def ode_sourceterm(P, q, r, dt, t):
    sint = tf.py_function(func=np.sin, inp=[t], Tout = tf.float64)
    cost = tf.py_function(func=np.cos, inp=[t], Tout = tf.float64)
    zr1 = tf.tile(sint, tf.constant([1,1,ldim,1], tf.int32))
    zr2 = tf.tile(cost, tf.constant([1,1,ldim,1], tf.int32))
    zr = tf.concat([zr1, zr2], 2) * Czr # zr is xr in the paper, which is 5(sint, cost)
    # dr = -dt (q'Mq/2 -25l/2) Note that Czr=5
    mat_tmp = tf.matmul(tf.transpose(q,perm=[0,1,3,2]), M_param)
    dr = -tf.matmul(mat_tmp, q)/2 + Czr*Czr*ldim/2
    dr = tf.multiply(dt, dr)
    # dq = dt((A-MP)'q - xr)
    MP = tf.matmul(M_param, P)
    mat_tmp = tf.transpose(tf.subtract(A_param, MP), perm=[0,1,3,2])
    dq = tf.multiply(dt, tf.subtract(tf.matmul(mat_tmp, q), zr))
    # dP = dt (A'P + P'A - P'MP + Cx)
    P_tmp = tf.matmul(tf.transpose(A_param, perm=[0,1,3,2]), P)
    P_tmp = tf.add(P_tmp, tf.transpose(P_tmp, perm=[0,1,3,2]))
    P_tmp = tf.subtract(P_tmp, tf.matmul(tf.transpose(P,perm=[0,1,3,2]), MP))
    dP = tf.multiply(dt, tf.add(P_tmp, Cx_param))
    return dP, dq, dr

def ode_solver_RK4(P_in, q_in, r_in, t_in):
    dt = t_in / LP
    P = P_in
    q = q_in
    r = r_in
    Pall = P_in
    qall = q_in
    t0 = t_in
    for l in range(0, LP):
        dP1, dq1, dr1 = ode_sourceterm(P,q,r,dt, t0)
        P1 = tf.add(P, dP1/2)
        q1 = tf.add(q, dq1/2)
        r1 = tf.add(r, dr1/2)
        dP2, dq2, dr2 = ode_sourceterm(P1, q1, r1, dt, t0-dt/2)
        P2 = tf.add(P, dP2/2)
        q2 = tf.add(q, dq2/2)
        r2 = tf.add(r, dr2/2)
        dP3, dq3, dr3 = ode_sourceterm(P2, q2, r2, dt, t0-dt/2)
        P3 = tf.add(P, dP3)
        q3 = tf.add(q, dq3)
        r3 = tf.add(r, dr3)
        dP4, dq4, dr4 = ode_sourceterm(P3, q3, r3, dt, t0-dt)
        t0 = t0-dt
        P = tf.add(P, tf.add(dP1/6, tf.add(dP2/3, tf.add(dP3/3, dP4/6))))
        q = tf.add(q, tf.add(dq1/6, tf.add(dq2/3, tf.add(dq3/3, dq4/6))))
        r = tf.add(r, tf.add(dr1/6, tf.add(dr2/3, tf.add(dr3/3, dr4/6))))
        Pall = tf.concat([Pall, P], 0)
        qall = tf.concat([qall, q], 0)
    return P, q, r, Pall, qall

# x_in is Ndata x1 xn x1
# the output S is the solution at t=0.
def S_nn(x_in, P_out, q_out, r_out):
    # broadcast
    x_in = tf.tile(x_in, tf.constant([1,m,1,1], tf.int32))
    P_out = tf.tile(P_out, tf.constant([Ndata,1,1,1], tf.int32))
    q_out = tf.tile(q_out, tf.constant([Ndata,1,1,1], tf.int32))
    xtranspose = tf.transpose(x_in, perm=[0,1,3,2])
    linear_term = tf.matmul(xtranspose, P_out)
    linear_term = tf.add(tf.matmul(linear_term, x_in)/2, r_out)
    linear_term = tf.add(linear_term, tf.matmul(xtranspose, q_out))
    linear_term = tf.reshape(linear_term, [Ndata, m]) 
    S = tf.reduce_min(linear_term, axis=1)
    k = tf.argmin(linear_term, axis=1)
    return S,k

# x_in is Ndata x1xnx1, P_in is Ndata xn xn, q_in is Ndata xn x1
def dtAxpBu(x_in, dt, P_in, q_in):
    P_in = tf.reshape(P_in, [Ndata,1,n,n])
    q_in = tf.reshape(q_in, [Ndata,1,n,1])
    dx = tf.add(tf.matmul(P_in, x_in), q_in)
    dx = tf.add(-tf.matmul(Mtraj_param, dx), tf.matmul(Atraj_param, x_in))
    dx = tf.multiply(dt, dx)
    return dx

# x_in is Ndata x1x n x1, Pall is Ndata x(2Lx+1) xn xn, qall is Ndata x(2Lx+1) xn x1, t_in is 1x1x1x1
# xall is Ndata x(Lx+1) xnx1
def x_nn(x_in, t_in, Pall, qall):
    dt = t_in / Lx
    xall = x_in
    x0 = x_in
    for l in range(0, Lx):
        dx1 = dtAxpBu(x0, dt, Pall[:,-2*l-1,:,:], qall[:,-2*l-1,:,:])
        dx2 = dtAxpBu(tf.add(x0, dx1/2), dt, Pall[:,-2*l-2,:,:], qall[:,-2*l-2,:,:])
        dx3 = dtAxpBu(tf.add(x0, dx2/2), dt, Pall[:,-2*l-2,:,:], qall[:,-2*l-2,:,:])
        dx4 = dtAxpBu(tf.add(x0, dx3), dt, Pall[:,-2*l-3,:,:], qall[:,-2*l-3,:,:])
        x0 = tf.add(x0, tf.add(dx1/6, tf.add(dx2/3, tf.add(dx3/3, dx4/6))))
        xall = tf.concat([xall, x0], 1)
    return xall

# Pall is Ndata x(Lx+1)xnxn, qall is Ndata x(Lx+1)xnx1, xall is Ndata x(Lx+1) xnx1
# note: in this input, Pall and qall are reversed
# uall is Ndata x(Lx+1) xlparam x1 (for now, lparam = n)
def u_nn(Pall, qall, xall):
    Bt = tf.tile(Bt_param, tf.constant([1,Lx+1,1,1], tf.int32))
    uall = tf.add(tf.matmul(Pall, xall), qall)
    uall = -tf.matmul(Bt, uall) * alpinv
    return uall 

P_resnet, q_resnet, r_resnet, Pall_resnet, qall_resnet = ode_solver_RK4(D_data, a_data, b_data, t_data)
S_resnet, k_resnet = S_nn(x_data, P_resnet, q_resnet, r_resnet)

if plotxuflag:
    Pall_k = tf.transpose(tf.gather(Pall_resnet, k_resnet, axis=1), perm=[1,0,2,3]) # Pall_k is Ndata x(2Lx+1) xnxn
    qall_k = tf.transpose(tf.gather(qall_resnet, k_resnet, axis=1), perm=[1,0,2,3]) # qall_k is Ndata x(2Lx+1) xnx1
    xall_resnet = x_nn(x_data, t_data, Pall_k, qall_k)
    uall_resnet = u_nn(Pall_k[:,::-2,:,:], qall_k[:,::-2,:,:], xall_resnet)
    tall_data = np.linspace(0.0, t0, Lx+1)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
# resnet solver
P_resnet_val = sess.run(P_resnet)
q_resnet_val = sess.run(q_resnet)
r_resnet_val = sess.run(r_resnet)
Pall_resnet_val = sess.run(Pall_resnet)
qall_resnet_val = sess.run(qall_resnet)
S_resnet_val = sess.run(S_resnet)
if plotxuflag:
    uall_resnet_val = sess.run(uall_resnet)
    xall_resnet_val = sess.run(xall_resnet)

print(S_resnet_val)

###################################### plot ###############################################

end = time.time()
print('Time: ', end - start)

if plotSflag:
    plt.figure()
    if n>1:
        a = plt.contourf(xv, yv, np.reshape(S_resnet_val, (nx, ny)), 20)
        plt.colorbar(a)
    else:
        plt.plot(x_grid, np.reshape(S_resnet_val, (Ndata)))
    plt.savefig(filenameS)

if plotxuflag:
    plt.figure()
    plt.plot(np.linspace(0, t0, Lx+1), np.transpose(np.reshape(xall_resnet_val[:,:,0,0], (Ndata, Lx+1))))
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.savefig(filenamex)
    
    plt.figure()
    plt.plot(np.linspace(0, t0, Lx+1), np.transpose(np.reshape(xall_resnet_val[:,:,ldim,0], (Ndata, Lx+1))))
    plt.xlabel('t')
    plt.ylabel('v(t)')
    plt.savefig(filenamev)
    
    plt.figure()
    plt.plot(np.linspace(0, t0, Lx+1), np.transpose(np.reshape(uall_resnet_val[:,:,0,0], (Ndata, Lx+1))))
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.savefig(filenameu)

