# This is the second example in the paper (see section 4.2)

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
parser.add_argument('--case', type = int, default = 1, help ='which case in example 2: case 1 or 2')
parser.add_argument('--Lx', type = int, default = "20", help ='number of grid points for x in RK')
args = parser.parse_args()

start = time.time()

case_num = args.case

Lx = args.Lx
LP = 2*Lx
if case_num == 1:
    n = 1
    m = 2
else:
    n = 16
    m = 4
ldim = n
T = args.T
t0 = args.t0
Ndata1d = 100
batchsize = 625
xleft = -2.0 # range of x is [-2,2]

plotSflag = False
ploterrflag = False
plotxuflag = False

if args.plot == "S":
    plotSflag = True
    filenameS = 'tdep_S{}d_t{:.2f}.png'.format(n, t0)
if args.plot == "err":
    ploterrflag = True
    filename_err = 'tdep_err{}d_t{:.2f}'.format(n, t0)
if args.plot == "xu":
    plotxuflag = True
    filenamex = 'tdep_x{}d_T{}.png'.format(n, int(T))
    filenameu = 'tdep_u{}d_T{}.png'.format(n, int(T))
    Ndata1d = 25 # change the number of data points in order to plot trajs

np.random.seed(1)
tf.reset_default_graph()

################# def initial cond start #################################################
if case_num == 1:
    nonzero_index = np.array([0,0])
    D_scalar = np.array([1.0, 1.0])
    a_scalar = np.array([-0.9, 0.9])
    b_data = np.reshape(np.array([0.405, 0.405]), (1,m,1,1))
else:
    # nonzero_index[i] gives the nonzero index for a_i
    nonzero_index = np.array([0,0,1,1])
    # D_scalar[i], a_scalar[i] give the nonzero value
    D_scalar = np.array([1.0, 1.0, 0.5, 0.5])
    a_scalar = np.array([-0.9, 0.9, -0.9, 0.9])    
    b_data = np.reshape(np.array([0.405, 0.405, 0.405, 0.405]), (1,m,1,1))

D_data = np.zeros((1,m,n,n))
a_data = np.zeros((1,m,n,1))
for i in range(m):
    D_data[0,i,:,:] = D_scalar[i]*np.eye(n)
    ind = nonzero_index[i]
    a_data[0,i,ind,0] = a_scalar[i]
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
tall_data = np.linspace(t0, T, Lx+1)
tall_data_dense = np.linspace(t0, T, LP+1)

if ploterrflag:
    # batch size: note (Ndata % bs) need to be zero
    bs = batchsize
else:
    bs = Ndata

x_ph = tf.placeholder(tf.float64, shape=(bs, 1,n,1))
t_ph = tf.placeholder(tf.float64, shape=(bs, 1,1,1))
################# def input x,t end #####################

# negative of rhs of backward odes
def ode_sourceterm(P, q, r, dt, t):
    # expt and expmt
    expt = tf.exp(t)
    expmt = tf.exp(-t)
    # dr = - dt e^t |q|^2/4
    dr = tf.matmul(tf.transpose(q,perm=[0,1,3,2]), q)
    multi = -tf.multiply(dt, expt)/4
    dr = tf.multiply(multi, dr)
    # dq = dt (q/2 - e^t P'q/2)
    dq = tf.matmul(tf.transpose(P,perm=[0,1,3,2]), q)
    dq = tf.subtract(q/2, tf.multiply(expt, dq) /2)
    dq = tf.multiply(dt, dq)
    # dP = dt(-e^tP'P/2 + P +e^{-t}I/2)
    dP = tf.matmul(tf.transpose(P,perm=[0,1,3,2]), P)
    dP = tf.subtract(P, tf.multiply(dP, expt)/2)
    dP = tf.add(dP, tf.multiply(expmt/2, tf.eye(n, dtype=tf.dtypes.float64)))
    dP = tf.multiply(dt, dP)
    return dP, dq, dr

def ode_solver_RK4(P_in, q_in, r_in, t_in):
    dt = (T-t_in) / LP
    P = P_in
    q = q_in
    r = r_in
    Pall = tf.tile(tf.expand_dims(P_in, axis = 2), tf.constant([bs,1,1,1,1], tf.int32))
    qall = tf.tile(tf.expand_dims(q_in, axis = 2), tf.constant([bs,1,1,1,1], tf.int32))
    t0 = tf.zeros_like(t_in, dtype = tf.float64) + T
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
        dP4, dq4, dr4 = ode_sourceterm(P3, q3, r3, dt, t0 - dt)
        P = tf.add(P, tf.add(dP1/6, tf.add(dP2/3, tf.add(dP3/3, dP4/6))))
        q = tf.add(q, tf.add(dq1/6, tf.add(dq2/3, tf.add(dq3/3, dq4/6))))
        r = tf.add(r, tf.add(dr1/6, tf.add(dr2/3, tf.add(dr3/3, dr4/6))))
        t0 = t0 - dt
        Pall = tf.concat([Pall, tf.expand_dims(P, axis = 2)], 2)
        qall = tf.concat([qall, tf.expand_dims(q, axis = 2)], 2)
    return P, q, r, Pall, qall


# x_in is bs*1*n*1
def S_nn(x_in, P_out, q_out, r_out):
    x_in = tf.tile(x_in, tf.constant([1,m,1,1], tf.int32))
    xtranspose = tf.transpose(x_in, perm=[0,1,3,2])
    linear_term = tf.matmul(xtranspose, P_out)
    linear_term = tf.add(tf.matmul(linear_term, x_in)/2, r_out)
    linear_term = tf.add(linear_term, tf.matmul(xtranspose, q_out))
    linear_term = tf.reshape(linear_term, [bs, m]) 
    S = tf.reduce_min(linear_term, axis=1)
    k = tf.argmin(linear_term, axis=1)
    return S,k, linear_term

# x_in is bs*1*n*1, P_in is bs*n*n, q_in is bs*n*1
def dtAxpBu(x_in, dt, P_in, q_in, t):
    expt = tf.exp(t)
    P_in = tf.reshape(P_in, [bs,1,n,n])
    q_in = tf.reshape(q_in, [bs,1,n,1])
    dx = tf.add(tf.matmul(P_in, x_in), q_in)
    dx = tf.subtract(x_in/2, tf.multiply(dx, expt)/2)
    dx = tf.multiply(dt, dx)
    return dx

# x_in is bs*1*n*1, Pall is bs*(2Lx+1)*n*n, qall is bs*(2Lx+1)*n*1, t_in is bs*1*1*1
# xall is bs*(Lx+1)*n*1
def x_nn(x_in, t_in, Pall, qall):
    dt = (T-t_in) / Lx
    xall = x_in
    x0 = x_in
    t = t_in
    for l in range(0, Lx):
        dx1 = dtAxpBu(x0, dt, Pall[:,-2*l-1,:,:], qall[:,-2*l-1,:,:], t)
        dx2 = dtAxpBu(tf.add(x0, dx1/2), dt, Pall[:,-2*l-2,:,:], qall[:,-2*l-2,:,:], t+dt/2)
        dx3 = dtAxpBu(tf.add(x0, dx2/2), dt, Pall[:,-2*l-2,:,:], qall[:,-2*l-2,:,:], t+dt/2)
        dx4 = dtAxpBu(tf.add(x0, dx3), dt, Pall[:,-2*l-3,:,:], qall[:,-2*l-3,:,:], t+dt)
        x0 = tf.add(x0, tf.add(dx1/6, tf.add(dx2/3, tf.add(dx3/3, dx4/6))))
        t = tf.add(t, dt)
        xall = tf.concat([xall, x0], 1)
    return xall

# Pall is bs*(Lx+1)*n*n, qall is bs*(Lx+1)*n*1, xall is bs*(Lx+1)*n*1
# note: in this input, Pall and qall are reversed
# uall is bs*(Lx+1)*ldim*1 (for now, ldim = n)
# tall is 1*(Lx+1)*1*1
def u_nn(Pall, qall, xall, tall):
    expt = tf.exp(tall)
    uall = tf.add(tf.matmul(Pall, xall), qall)
    uall = -tf.multiply(uall, expt)/2
    return uall 

P_resnet, q_resnet, r_resnet, Pall_resnet, qall_resnet = ode_solver_RK4(D_data, a_data, b_data, t_ph)
S_resnet, k_resnet, Sk_resnet = S_nn(x_ph, P_resnet, q_resnet, r_resnet)

if ploterrflag:
    dSkdx = tf.concat([tf.expand_dims(tf.gradients(Sk_resnet[...,i], x_ph)[0], axis=0) for i in range(m)], axis = 0)
    dSkdt = tf.concat([tf.expand_dims(tf.gradients(Sk_resnet[...,i], t_ph)[0], axis=0) for i in range(m)], axis = 0)
    pde_err = -dSkdt + tf.reduce_sum(dSkdx*dSkdx * np.exp(t0)/4 - x_ph * x_ph *np.exp(-t0) /4 - dSkdx*x_ph/2, axis=-2, keepdims=True)

if plotxuflag:
    k_expand = tf.expand_dims(k_resnet, axis = -1)
    Pall_k = tf.gather(Pall_resnet, k_expand, axis=1, batch_dims=1)[:,0] # Pall_k is bs *(2Lx+1) *n*n
    qall_k = tf.gather(qall_resnet, k_expand, axis=1, batch_dims=1)[:,0] # qall_k is bs *(2Lx+1) *n*1
    xall_resnet = x_nn(x_ph, t_ph, Pall_k, qall_k)
    uall_resnet = u_nn(Pall_k[:,::-2,:,:], qall_k[:,::-2,:,:], xall_resnet, np.reshape(tall_data, (1,Lx+1,1,1)))

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
    uall_resnet_val = sess.run(uall_resnet, feed_dict = {x_ph: x_data, t_ph: t_data})
    xall_resnet_val = sess.run(xall_resnet, feed_dict = {x_ph: x_data, t_ph: t_data})

print('End')
end = time.time()
print('Time: ', end - start)

############################## plot begin #########################
if plotSflag:
    plt.figure()
    if n>1:
        a = plt.contourf(xv, yv, np.reshape(S_resnet_val, (nx, ny)), 20)
        plt.colorbar(a)
    else:
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
    plt.plot(np.linspace(t0, T,Lx+1), np.transpose(np.reshape(xall_resnet_val[:,:,0,0], (Ndata, Lx+1))))
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.savefig(filenamex)
    plt.figure()
    plt.plot(np.linspace(t0, T,Lx+1), np.transpose(np.reshape(uall_resnet_val[:,:,0,0], (Ndata, Lx+1))))
    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.savefig(filenameu)
############################## plot end #############################
