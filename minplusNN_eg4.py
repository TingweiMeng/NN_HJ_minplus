# This is the fourth example in the paper (see section 4.4)

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
n = 16
m = 2
ldim = n
T = args.T
t0 = args.t0
Ndata1d = 100
batchsize = 25   # 625 when Lx = 20, smaller when Lx is larger
xleft = -2.0 # range of x is [-2,2]

plotSflag = False
ploterrflag = False
plotxuflag = False

if args.plot == "S":
    plotSflag = True
    filenameS = 'admm_S{}d_t{:.2f}.png'.format(n, t0)
if args.plot == "err":
    ploterrflag = True
    filename_err = 'admm_err{}d_t{:.2f}'.format(n, t0)
if args.plot == "xu":
    plotxuflag = True
    filenamex = 'admm_x{}d_T{}.png'.format(n, int(T))
    filenameu = 'admm_u{}d_T{}.png'.format(n, int(T))
    Ndata1d = 25 # change the number of data points in order to plot trajs

np.random.seed(1)
tf.reset_default_graph()

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
add_term_ph = tf.placeholder(tf.float64, shape=(bs, m))  # the additional term to put in S_all
################# def input x,t end #####################


################# def initial cond start #################################################
rho = 1.0  # parameter in ADMM
y_admm = np.zeros((bs, m, n,1))
x_admm = np.zeros((bs, m, n,1))
w_admm = np.zeros((bs, m, n,1))
# J = rho/2* ||x - y + w||^2
D_data = np.zeros((bs,m,n,n))
for i in range(n):
    D_data[:,:,i,i] = rho
a_ph = tf.placeholder(tf.float64, shape=(bs, m,n,1))
b_ph = tf.placeholder(tf.float64, shape=(bs, m,1,1))
################# def initial cond end ###################################################


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

# P is bs * m * n * n, q is bs * m * n * 1, r is bs * m *1*1
# Pall is bs * m * (LP+1) * n * n, qall is bs * m* (LP+1) *n*1
def ode_solver_RK4(P_in, q_in, r_in, t_in):
    dt = (T-t_in) / LP
    P = P_in
    q = q_in
    r = r_in
    Pall = tf.expand_dims(P_in, axis = 2)
    qall = tf.expand_dims(q_in, axis = 2)
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


# x_in is bs*1*n*1, S is [bs]
def S_nn(x_in, P_out, q_out, r_out, add_term):
    x_in = tf.tile(x_in, tf.constant([1,m,1,1], tf.int32))
    xtranspose = tf.transpose(x_in, perm=[0,1,3,2])
    linear_term = tf.matmul(xtranspose, P_out)
    linear_term = tf.add(tf.matmul(linear_term, x_in)/2, r_out)  # x'Px/2 + r
    linear_term = tf.add(linear_term, tf.matmul(xtranspose, q_out))  # x'Px/2 + r + x'q
    linear_term = tf.reshape(linear_term, [bs, m]) 
    linear_term = linear_term + add_term # add_term is for correcting the init cond
    S = tf.reduce_min(linear_term, axis=1)
    k = tf.argmin(linear_term, axis=1)
    return S,k, linear_term

# x_in is bs*1*n*1, P_in is bs*n*n, q_in is bs*n*1
def dtAxpBu(x_in, dt, P_in, q_in, t):
    expt = tf.exp(t)
    P_in = tf.reshape(P_in, [bs,1,n,n])
    q_in = tf.reshape(q_in, [bs,1,n,1])
    dx = tf.add(tf.matmul(P_in, x_in), q_in)  # Px + q
    dx = tf.subtract(x_in/2, tf.multiply(dx, expt)/2)  # x/2 - e^t(Px+q)/2
    dx = tf.multiply(dt, dx)  # dx <- dt * dx
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

# x_in is bs*m *n*1, P_in is bs*m * n*n, q_in is bs*m * n*1, dx is bs * m * n *1
def dtAxpBu_mfns(x_in, dt, P_in, q_in, t):
    expt = tf.exp(t)
    dx = tf.add(tf.matmul(P_in, x_in), q_in)  # Px + q
    dx = tf.subtract(x_in/2, tf.multiply(dx, expt)/2)  # x/2 - e^t(Px+q)/2
    dx = tf.multiply(dt, dx)  # dx <- dt * dx
    return dx

# x_in is bs*1*n*1, Pall is bs*m * (2Lx+1)*n*n, qall is bs*m * (2Lx+1)*n*1, t_in is bs*1*1*1
# the output x0 is bs*m *n*1 (initial position of opt traj)
def x_nn_mfns(x_in, t_in, Pall, qall):
    dt = (T-t_in) / Lx
    x0 = tf.tile(x_in, tf.constant([1,m,1,1], tf.int32))
    t = t_in
    for l in range(0, Lx):
        dx1 = dtAxpBu_mfns(x0, dt, Pall[:,:,-2*l-1,:,:], qall[:,:,-2*l-1,:,:], t)
        dx2 = dtAxpBu_mfns(tf.add(x0, dx1/2), dt, Pall[:,:,-2*l-2,:,:], qall[:,:,-2*l-2,:,:], t+dt/2)
        dx3 = dtAxpBu_mfns(tf.add(x0, dx2/2), dt, Pall[:,:,-2*l-2,:,:], qall[:,:,-2*l-2,:,:], t+dt/2)
        dx4 = dtAxpBu_mfns(tf.add(x0, dx3), dt, Pall[:,:,-2*l-3,:,:], qall[:,:,-2*l-3,:,:], t+dt)
        x0 = tf.add(x0, tf.add(dx1/6, tf.add(dx2/3, tf.add(dx3/3, dx4/6))))
        t = tf.add(t, dt)
    return x0

# Pall is bs*(Lx+1)*n*n, qall is bs*(Lx+1)*n*1, xall is bs*(Lx+1)*n*1
# note: in this input, Pall and qall are reversed
# uall is bs*(Lx+1)*ldim*1 (for now, ldim = n)
# tall is 1*(Lx+1)*1*1
def u_nn(Pall, qall, xall, tall):
    expt = tf.exp(tall)
    uall = tf.add(tf.matmul(Pall, xall), qall)
    uall = -tf.multiply(uall, expt)/2
    return uall 

# J0(y) = |y-y0|_1; J1 = |y-y1|_2
# y and z are bs * m*n*1 np array, y_shift is 1*m*n*1, equals [y0,y1]
def compute_admm_y(z, rho, y_shift):
    w = 0 * z
    rho_inv = 1/rho
    z_shift = z - y_shift
    # y(:,0,:,:) is minimizer of |y-y0|_1 + rho/2 * |y-z0|^2, let w = y-y0
    # then w is minimizer of |w|_1 + rho/2 * |w-(z0-y0)|^2, and y=w+y0
    z_sft0 = z_shift[:,0,:,:]
    w[:,0,:,:] = np.maximum(z_sft0 - rho_inv, 0) - np.maximum(-z_sft0 - rho_inv, 0)
    # y1 is the minimizer of |y-y1|_2 + rho/2|y-z|^2, i.e., y = z-z/(rho|z|) if |z|>1/rho; 0 otherwise
    # Let w =y-y1, minimizer of |w|_2+rho/w|y-(z-y1)|^2, and y=w+y1
    z_sft1 = z_shift[:,1,:,:]
    z1norm = np.sqrt(np.sum(z_sft1**2, axis = -2, keepdims = True))
    w[:,1,:,:] = z_sft1 - z_sft1 / z1norm * np.minimum(z1norm, rho_inv)
    y = w + y_shift
    return y

P_resnet, q_resnet, r_resnet, Pall_resnet, qall_resnet = ode_solver_RK4(D_data, a_ph, b_ph, t_ph)
x0_resnet = x_nn_mfns(x_ph, t_ph, Pall_resnet, qall_resnet)
S_resnet, k_resnet, Sall_resnet = S_nn(x_ph, P_resnet, q_resnet, r_resnet, add_term_ph)

if ploterrflag:
    dSkdx = tf.concat([tf.expand_dims(tf.gradients(Sall_resnet[...,i], x_ph)[0], axis=0) for i in range(m)], axis = 0)
    dSkdt = tf.concat([tf.expand_dims(tf.gradients(Sall_resnet[...,i], t_ph)[0], axis=0) for i in range(m)], axis = 0)
    pde_err = -dSkdt + tf.reduce_sum(dSkdx*dSkdx * np.exp(t0)/4 - x_ph * x_ph *np.exp(-t0) /4 - dSkdx*x_ph/2, axis=-2, keepdims=True)
    pde_err_val = np.zeros((m,Ndata,1,1,1))

if plotxuflag:
    k_expand = tf.expand_dims(k_resnet, axis = -1)
    Pall_k = tf.gather(Pall_resnet, k_expand, axis=1, batch_dims=1)[:,0] # Pall_k is bs *(2Lx+1) *n*n
    qall_k = tf.gather(qall_resnet, k_expand, axis=1, batch_dims=1)[:,0] # qall_k is bs *(2Lx+1) *n*1
    xall_resnet = x_nn(x_ph, t_ph, Pall_k, qall_k)
    uall_resnet = u_nn(Pall_k[:,::-2,:,:], qall_k[:,::-2,:,:], xall_resnet, np.reshape(tall_data, (1,Lx+1,1,1)))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# J0(y) = |y-1|_1, J1(y) = |y+1|_2
y_shift = np.zeros((1,m,n,1))
y_shift[:,0,:2,:] = 1.0
y_shift[:,1,:2,:] = -1.0


for kk in range(Ndata//bs):
    x_batch = x_data[kk*bs: (kk+1)*bs,:,:,:]
    t_batch = t_data[kk*bs: (kk+1)*bs,:,:,:]
    # ADMM
    max_iters = 1000
    tol = 1e-4
    for i in range(max_iters):
        # save prev results
        x_admm_prev, y_admm_prev, w_admm_prev = x_admm, y_admm, w_admm
        a_data = rho * (w_admm - y_admm)
        b_data = 0.5 * rho * np.sum((w_admm - y_admm)**2, axis = -2, keepdims = True)
        # compute x: initial pos of linear-quadratic opt ctrl problem with J = rho/2 * ||x - y + w||^2
        # all the admm variables have size bs * m * n * 1
        x_admm = sess.run(x0_resnet, feed_dict={x_ph: x_batch, t_ph: t_batch, a_ph: a_data, b_ph: b_data})
        # compute y: yi = argmin_y J_i(y) + rho/2 * ||x - y + w||^2
        y_admm = compute_admm_y(x_admm + w_admm, rho, y_shift)
        # update w <- w + x-y 
        w_admm = w_admm + x_admm - y_admm
        # check convergence criteria
        err_xy = np.amax(np.sum((x_admm - y_admm)**2, axis = -2))
        err_x = np.amax(np.sum((x_admm - x_admm_prev)**2, axis = -2))
        err_y = np.amax(np.sum((y_admm - y_admm_prev)**2, axis = -2))
        err_w = np.amax(np.sum((w_admm - w_admm_prev)**2, axis = -2))
        print('iter {}, err xy {}, err x {}, err y {}, err w {}\n'.format(i, err_xy, err_x, err_y, err_w), flush = True)
        if err_xy < tol and err_x < tol and err_y < tol and err_w < tol:
            break
    
    a_data = rho * (w_admm - y_admm)
    b_data = 0.5 * rho * np.sum((w_admm - y_admm)**2, axis = -2, keepdims = True)
    
    # define J(y_admm)
    y_m_yshift = y_admm[:,:,:,0] - y_shift[:,:,:,0]  # size bs*m*n
    init_J = np.zeros((bs,m))
    # J0 = |y-y0|_1
    init_J[:,0] = np.sum(np.abs(y_m_yshift[:,0,:]), axis = -1)  # size bs * m
    # J1 = |y-y1|_2
    init_J[:,1] = np.sqrt(np.sum(y_m_yshift[:,1,:]**2, axis = -1))
    
    # add_term_i = -rho/2 |x_i-y_i+w_i|^2 + J_i(y_i)
    add_term_data = - 0.5 * rho * np.sum((x_admm + w_admm - y_admm)**2, axis= -2)[:,:,0] + init_J  # size bs * m
    
    if plotSflag:
        S_resnet_val, k_resnet_val = sess.run([S_resnet, k_resnet], feed_dict={x_ph: x_data, t_ph: t_data, a_ph: a_data, b_ph: b_data, add_term_ph: add_term_data})
    
    if ploterrflag:
        pde_err_val[:,kk*bs:(kk+1)*bs,...] = sess.run(pde_err, feed_dict={x_ph: x_batch, t_ph: t_batch, 
                a_ph: a_data, b_ph: b_data, add_term_ph: add_term_data})
    
    if plotxuflag:
        uall_resnet_val = sess.run(uall_resnet, feed_dict = {x_ph: x_data, t_ph: t_data, a_ph: a_data, b_ph: b_data, add_term_ph: add_term_data})
        xall_resnet_val = sess.run(xall_resnet, feed_dict = {x_ph: x_data, t_ph: t_data, a_ph: a_data, b_ph: b_data, add_term_ph: add_term_data})

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
    print('max err: {}\n'.format(np.amax(np.abs(pde_err_val))))
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
