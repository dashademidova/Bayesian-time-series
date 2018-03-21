import numpy as np
import numpy.linalg as la
import scipy
import scipy.special
from scipy.special import binom

def initial_var(phi, Shrink=1, Delta=0.01):
    m = phi.shape[0]
    w = np.max(np.abs(la.eigvals(phi)))
    if w >= 1:
        phi /= w + Delta
    return phi    

def sqrtm(A):
    w, v = la.eigh(A)
    return v.dot(np.diag(np.sqrt(w)).dot(v.T))

def V2LDL(v):
    m = v.shape[0]
    pre = np.zeros(np.int32(binom(m+1, 2)))
    c = la.cholesky(v).T #la.cholesky() returns lower triangular, not upper triangular
    d = np.diagonal(c)
    l = c.T / d
    pre[:np.int32(binom(m, 2))] = l[np.tril_indices(m, k=-1)]
    pre[np.int32(binom(m, 2)):np.int32(binom(m+1, 2))] = np.log(d**2)
    return pre

def par2pre_varp(phi):
    m = phi.shape[0]
    pre = np.zeros(m**2)
    U = la.solve(np.eye(m**2) - np.kron(phi, phi), np.eye(m).reshape((-1, 1), order="F")).reshape((m, m))
    # v = U - diag(m)
    v = U - 0.9 * np.eye(m)
    q = la.inv(sqrtm(v)).dot(phi.dot(sqrtm(U)))
    pre[:np.int32(binom(m + 1,2))] = V2LDL(v)
    delta, _ = la.slogdet(q)
    s = 2 * la.inv(np.eye(m) + np.diag(np.r_[delta, np.ones(m - 1)])).dot(q) - np.eye(m)
    pre[np.int32(binom(m + 1,2)):m**2] = s[np.tril_indices(m, k=-1)]
    return {"pre": pre, "delta": delta, "U": U}

def VQ2par(v, q):
    m = v.shape[0]
    phi = np.zeros((m, m))
    u1 = np.eye(m) * 0.9 + v
    u2 = sqrtm(v).dot(q).dot(sqrtm(u1))
    txi = u2
    bigu = u1
    phi = txi.dot(la.inv(bigu))
    return phi

def pre2par_varp( pre, delta):
    m = np.int32(np.sqrt(pre.size))   
    v = np.zeros((m, m)) 
    q = v 
    l = np.eye(m)
    l[np.tril_indices(m, k=-1)] = pre[:np.int32(binom(m, 2))]
    d = np.diag(np.exp(pre[np.int32(binom(m, 2)):np.int32(binom(m + 1, 2))]))   #diag -> diagonal, lost +1
    v = l.dot(d).dot(l.T)
    s = np.zeros((m, m)) 
    s[np.tril_indices(m, k=-1)] = pre[np.int32(binom(m + 1, 2)):m**2]
    s = s - s.T
    q = np.diag(np.r_[delta, np.ones(m-1)]).dot(np.eye(m) - s).dot(la.inv(np.eye(m) + s))
    phi = VQ2par(v, q)
    return phi

def varp_lkhd(y, phi, sigma, sigma_inv):
    m, n = y.shape    
    y1 = y[:, :n-1]
    y2 = y[:, 1:] - phi.dot(y1)
    gp = (la.solve(np.eye(m**2) - np.kron(phi, phi), sigma.reshape((-1, 1), order="F"))).reshape(m, m)
    _, logdet_gp = la.slogdet(gp)
    _, logdet_sigma = la.slogdet(sigma)
    q = np.sum(np.diag(la.solve(gp, y[:, :1]).T.dot(y[:, :1]))) + np.sum(np.diag(sigma_inv.T.dot(y2).dot(y2.T)))   #diag -> diagonal
    l = - m * n * np.log(2 * np.pi) / 2 - logdet_gp + (n - 1) * logdet_sigma / 2 - q / 2
    return l

def varppre_lkhd(y, pre, delta, sigma, sigma_inv):
    m = y.shape[0]
    p = delta.size 
    phi = pre2par_varp(pre, delta)
    l = - varp_lkhd(y, phi, sigma, sigma_inv)
    if np.isnan(l):
        l = 1e5
    return l

def pre2VQ( pre, delta):
    m = np.int32(np.sqrt(pre.size))   
    l = np.eye(m)
    l[np.tril_indices(m, k=-1)] = pre[:np.int32(binom(m, 2))]
    d = np.diag(np.exp(pre[np.int32(binom(m, 2)):np.int32(binom(m + 1, 2))]))   #diag -> diagonal, lost +1
    v = l.dot(d).dot(l.T)
    s = np.zeros((m, m))
    s[np.tril_indices(m, k=-1)] = pre[np.int32(binom(m + 1, 2)):m**2]
    s = s - s.T
    q = np.diag(np.r_[delta, np.ones(m-1)]).dot(np.eye(m) - s).dot(la.inv(np.eye(m) + s))
    phi = VQ2par(v, q)
    return v, q
