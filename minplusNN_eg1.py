# This is the first example in the paper (see section 4.1)

import random
import numpy as np
import tensorflow as tf
import time
import scipy.io
import matplotlib.pyplot as plt 
from scipy.linalg import expm
import argparse

parser = argparse.ArgumentParser(description='min plus NN for high dimensional optimal control problems')
parser.add_argument('--t0', type = float, default = 0.0, help ='initial time')
parser.add_argument('--T', type = float, default = 1.0, help ='terminal time')
parser.add_argument('--plot', type = str, default = "S", help ='which quantity to plot: S, err or xu')
parser.add_argument('--Lx', type = int, default = "20", help ='number of grid points for x in RK')
args = parser.parse_args()

start = time.time()

Lx = args.Lx
LP = 2*Lx
m = 2
n = 16
ldim = n
T = args.T
t0 = args.t0
Ndata1d = 100
xleft = -2.0 # range of x is [-2,2]
batchsize = 125

plotSflag = False
ploterrflag = False
plotxuflag = False

if args.plot == "S":
    plotSflag = True
    filenameS = 'const_S{}d_t{:.2f}.png'.format(n, t0)
if args.plot == "err":
    ploterrflag = True
    filename_err = 'const_err{}d_t{:.2f}'.format(n, t0)
if args.plot == "xu":
    plotxuflag = True
    filenamex = 'const_x{}d_T{}.png'.format(n, int(T))
    filenameu = 'const_u{}d_T{}.png'.format(n, int(T))
    Ndata1d = 25 # change the number of data points in order to plot trajs

np.random.seed(1)
tf.reset_default_graph()

################# def random R, B, S, A, Cx start #########################
# Rinv is ldim*ldim symmetric positive definite, which is inverse of R
Rinv_const = np.eye(n)
R_const = np.eye(n)

# B,S is rand(n,l), A is rand(n,n) (A here is Cxp in the paper)
Bt_const = np.eye(n)
St_const = np.zeros((n,n))
A_const = np.eye(n)

# Cx is nxn symmetric positive def (is Cxx in the paper)
Cx_const = np.eye(n)
################# def random R, B, S, A, Cx end #############################


################# def M,Q start ############################################################
# M = Bt'*Rinv*Bt (Cpp in the paper)
M_const = np.matmul(np.transpose(Bt_const), np.matmul(Rinv_const, Bt_const)) #np.eye(n)

# Q = Cx+ St'*Rinv*St
Q_const = Cx_const + np.matmul(np.transpose(St_const), np.matmul(Rinv_const, St_const))
################# def M,Q end   ###########################################################

################# def initial cond start #################################################
# D(0,i,:,:) is nxn symmetric nonnegative def matrix, which is G_i in the paper
D_data = np.tile(np.eye(n), (1,m,1,1))

# if use shift_x to def a,b, use this
shift_x = np.array([[[[-1.0],[-1.0]], [[1.0],[1.0]]]])
shift_x = np.concatenate((shift_x, np.zeros((1,m,n-2,1))), axis=2)
a_data = -np.matmul(D_data, shift_x)
b_data = -np.matmul(np.transpose(shift_x, (0,1,3,2)), a_data)/2
################# def initial cond end ###################################################


################# def input x,t start #################
# construct x, t
xright = -xleft
if n==1:
    Ndata = Ndata1d
    x_grid = np.linspace(xleft, xright, Ndata)
    x_data = np.reshape(x_grid, (Ndata,1,1,1))
else:
    if plotxuflag:
        Ndata = Ndata1d
        x_grid = np.reshape(np.linspace(xleft, xright, Ndata), (Ndata,1,1,1))
        x_data = np.concatenate((x_grid, np.zeros((Ndata,1,n-1,1))), axis = 2)
    else:
        nx = Ndata1d
        ny = Ndata1d
        Ndata = nx * ny
        x = np.linspace(xleft, xright, nx)
        y = np.linspace(xleft, xright, ny)
        xv, yv = np.meshgrid(x, y)
        xv_arr = np.reshape(xv, (Ndata,1,1,1))
        yv_arr = np.reshape(yv, (Ndata,1,1,1))
        zero_arr = np.zeros((Ndata, 1, n-2,1))
        x_data = np.concatenate((xv_arr, yv_arr, zero_arr), axis = 2)
t_data = np.zeros((Ndata,1,1,1)) + t0

if ploterrflag:
    # batch size: note (Ndata % bs) need to be zero
    bs = batchsize
else:
    bs = Ndata

x_ph = tf.placeholder(tf.float64, shape=(bs, 1,n,1))
t_ph = tf.placeholder(tf.float64, shape=(bs, 1,1,1))
################# def input x,t end #####################


# for HJ PDE
A_param = tf.Variable(np.tile(np.reshape(A_const, (1,1,n,n)), (1,m,1,1)), dtype=tf.float64)
M_param = tf.Variable(np.tile(np.reshape(M_const, (1,1,n,n)), (1,m,1,1)), dtype=tf.float64) 
Cx_param = tf.Variable(np.tile(np.reshape(Cx_const, (1,1,n,n)), (1,m,1,1)), dtype=tf.float64)

# for trajectory
Bt_param = tf.Variable(np.tile(np.reshape(Bt_const, (1,1,n,n)),(bs,1,1,1)), dtype=tf.float64)
St_param = tf.Variable(np.tile(np.reshape(St_const, (1,1,n,n)),(bs,1,1,1)), dtype=tf.float64)
Rinv_param = tf.Variable(np.tile(np.reshape(Rinv_const, (1,1,n,n)),(bs,1,1,1)), dtype=tf.float64)
Atraj_param = tf.Variable(np.tile(np.reshape(A_const, (1,1,n,n)),(bs,1,1,1)), dtype=tf.float64)
Mtraj_param = tf.Variable(np.tile(np.reshape(M_const, (1,1,n,n)),(bs,1,1,1)), dtype=tf.float64) 

def ode_sourceterm(P, q, r, dt):
    # dr = -dt q'Mq/2
    mat_tmp = tf.matmul(tf.transpose(q,perm=[0,1,3,2]), M_param)
    dr = -tf.multiply(dt, tf.matmul(mat_tmp, q))/2
    # dq = dt (A-MP)'q
    MP = tf.matmul(M_param, P)
    mat_tmp = tf.transpose(tf.subtract(A_param, MP), perm=[0,1,3,2])
    dq = tf.multiply(dt, tf.matmul(mat_tmp, q))
    # dP = dt(A'P+P'A - P'MP + Cx) In this case, Cxp=A
    P_tmp = tf.matmul(tf.transpose(A_param, perm=[0,1,3,2]), P)
    P_tmp = tf.add(P_tmp, tf.transpose(P_tmp, perm=[0,1,3,2]))
    P_tmp = tf.subtract(P_tmp, tf.matmul(tf.transpose(P,perm=[0,1,3,2]), MP))
    dP = tf.multiply(dt, tf.add(P_tmp, Cx_param))
    return dP, dq, dr

def ode_solver_RK4(P_in, q_in, r_in, t_in):
    dt = (T-t_in) / LP
    P = P_in
    q = q_in
    r = r_in
    Pall = tf.tile(tf.expand_dims(P_in, axis = 2), tf.constant([bs,1,1,1,1], tf.int32))
    qall = tf.tile(tf.expand_dims(q_in, axis = 2), tf.constant([bs,1,1,1,1], tf.int32))
    for l in range(0, LP):
        dP1, dq1, dr1 = ode_sourceterm(P,q,r,dt)
        P1 = tf.add(P, dP1/2)
        q1 = tf.add(q, dq1/2)
        r1 = tf.add(r, dr1/2)
        dP2, dq2, dr2 = ode_sourceterm(P1, q1, r1, dt)
        P2 = tf.add(P, dP2/2)
        q2 = tf.add(q, dq2/2)
        r2 = tf.add(r, dr2/2)
        dP3, dq3, dr3 = ode_sourceterm(P2, q2, r2, dt)
        P3 = tf.add(P, dP3)
        q3 = tf.add(q, dq3)
        r3 = tf.add(r, dr3)
        dP4, dq4, dr4 = ode_sourceterm(P3, q3, r3, dt)
        P = tf.add(P, tf.add(dP1/6, tf.add(dP2/3, tf.add(dP3/3, dP4/6))))
        q = tf.add(q, tf.add(dq1/6, tf.add(dq2/3, tf.add(dq3/3, dq4/6))))
        r = tf.add(r, tf.add(dr1/6, tf.add(dr2/3, tf.add(dr3/3, dr4/6))))
        Pall = tf.concat([Pall, tf.expand_dims(P, axis = 2)], 2)
        qall = tf.concat([qall, tf.expand_dims(q, axis = 2)], 2)
    return P, q, r, Pall, qall

# x_in is bs*1*n*1, t_in is bs*1*1*1
def S_nn(x_in, P_out, q_out, r_out):
    x_in = tf.tile(x_in, tf.constant([1,m,1,1], tf.int32))
    xtranspose = tf.transpose(x_in, perm=[0,1,3,2])
    linear_term = tf.matmul(xtranspose, P_out)
    linear_term = tf.add(tf.matmul(linear_term, x_in)/2, r_out)
    linear_term = tf.add(linear_term, tf.matmul(xtranspose, q_out))
    # linear_term = x'Px/2 + x'q + r
    linear_term = tf.reshape(linear_term, [bs, m]) 
    S = tf.reduce_min(linear_term, axis=1)
    k = tf.argmin(linear_term, axis=1)
    return S,k, linear_term

# x_in is bs*1*n*1, P_in is bs*n*n, q_in is bs*n*1
def dtAxpBu(x_in, dt, P_in, q_in):
    P_in = tf.reshape(P_in, [bs,1,n,n])
    q_in = tf.reshape(q_in, [bs,1,n,1])
    dx = tf.add(tf.matmul(P_in, x_in), q_in)
    dx = tf.add(-tf.matmul(Mtraj_param, dx), tf.matmul(Atraj_param, x_in))
    dx = tf.multiply(dt, dx)
    return dx

# x_in is bs*1*n*1, Pall is bs*(2Lx+1)*n*n, qall is bs*(2Lx+1)*n*1, t_in is bs*1*1*1
# xall is bs*(Lx+1)*n*1
def x_nn(x_in, t_in, Pall, qall):
    dt = (T-t_in) / Lx
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

# Pall is bs*(Lx+1)*n*n, qall is bs*(Lx+1)*n*1, xall is bs*(Lx+1)*n*1
# note: in this input, Pall and qall are reversed
# uall is bs*(Lx+1)*lparam*1 (for now, lparam = n)
def u_nn(Pall, qall, xall):
    Bt = tf.tile(Bt_param, tf.constant([1,Lx+1,1,1], tf.int32))
    St = tf.tile(St_param, tf.constant([1,Lx+1,1,1], tf.int32))
    Rinv = tf.tile(Rinv_param, tf.constant([1,Lx+1,1,1], tf.int32))
    uall = tf.add(tf.matmul(Pall, xall), qall)
    uall = -tf.matmul(Rinv, tf.add(tf.matmul(Bt, uall), tf.matmul(St, xall)))
    return uall 


P_resnet, q_resnet, r_resnet, Pall_resnet, qall_resnet = ode_solver_RK4(D_data, a_data, b_data, t_ph)
S_resnet, k_resnet, Sk_resnet = S_nn(x_ph, P_resnet, q_resnet, r_resnet)

if ploterrflag:
    dSkdt = tf.concat([tf.expand_dims(tf.gradients(Sk_resnet[...,i], t_ph)[0], axis=0) for i in range(m)], axis = 0)
    dSkdx = tf.concat([tf.expand_dims(tf.gradients(Sk_resnet[...,i], x_ph)[0], axis=0) for i in range(m)], axis = 0)
    pde_err = -dSkdt + tf.reduce_sum(dSkdx*dSkdx/2 - x_ph * x_ph /2 - dSkdx*x_ph, axis=-2, keepdims=True)

if plotxuflag:
    k_expand = tf.expand_dims(k_resnet, axis = -1)
    Pall_k = tf.gather(Pall_resnet, k_expand, axis=1, batch_dims=1)[:,0] # Pall_k is bs*(2Lx+1)*n*n
    qall_k = tf.gather(qall_resnet, k_expand, axis=1, batch_dims=1)[:,0] # qall_k is bs*(2Lx+1)*n*1
    xall_resnet = x_nn(x_ph, t_ph, Pall_k, qall_k)
    uall_resnet = u_nn(Pall_k[:,::-2,:,:], qall_k[:,::-2,:,:], xall_resnet)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if plotSflag:
    S_resnet_val = sess.run(S_resnet, feed_dict={x_ph: x_data, t_ph: t_data})

if ploterrflag:
    pde_err_val = np.zeros((m,Ndata,1,1,1))
    for i in range(int(Ndata/bs)):
        pde_err_val[:,i*bs:(i+1)*bs,...] = sess.run(pde_err, feed_dict={x_ph: x_data[i*bs:(i+1)*bs,:,:,:], t_ph: t_data[i*bs:(i+1)*bs,:,:,:]})
    print('max err: {}\n'.format(np.amax(np.abs(pde_err_val))))

if plotxuflag:
    uall_resnet_val = sess.run(uall_resnet, feed_dict={x_ph: x_data, t_ph: t_data})
    xall_resnet_val = sess.run(xall_resnet, feed_dict={x_ph: x_data, t_ph: t_data})
print('End')

end = time.time()
print('Time: ', end - start)


if plotSflag:
    if n>1:
        plt.figure()
        a = plt.contourf(xv, yv, np.reshape(S_resnet_val, (nx, ny)), 20)
        plt.colorbar(a)
        plt.savefig(filenameS)
    else:
        plt.figure()
        plt.plot(x_grid, np.reshape(S_resnet_val, (Ndata)))
        plt.savefig(filenameS)

if ploterrflag:
    if n>1:
        pde_err_val_reshape = np.reshape(pde_err_val, (m, nx, ny))
        for i in range(m):
            plt.figure()
            a = plt.contourf(xv, yv, pde_err_val_reshape[i,:,:], 20)
            plt.colorbar(a)
            plt.savefig(filename_err + "_{}.png".format(i))
    else:
        pde_err_val_reshape = np.reshape(pde_err_val, (m, Ndata))
        for i in range(m):
            plt.figure()
            plt.plot(x_grid, pde_err_val_reshape[i,:])
            plt.savefig(filename_err + "_{}.png".format(i))

if plotxuflag:
    plt.figure()
    plt.plot(np.linspace(t0, T, Lx+1), np.transpose(np.reshape(xall_resnet_val[:,:,0,0], (Ndata, Lx+1))))
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.savefig(filenamex)
    plt.figure()
    plt.plot(np.linspace(t0, T, Lx+1), np.transpose(np.reshape(uall_resnet_val[:,:,0,0], (Ndata, Lx+1))))
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.savefig(filenameu)
