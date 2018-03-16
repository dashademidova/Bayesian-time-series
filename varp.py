import numpy as np
import numpy.linalg as la
import scipy
import scipy.special
from scipy.special import binom

# sqrtm <- function(A){
#   return(eigen(A)$vectors%*%diag(sqrt(eigen(A)$values))%*%t(eigen(A)$vectors))  
# }

def sqrtm(A):
    w, v = la.eigh(A)
    return v.dot(np.diag(np.sqrt(w)).dot(v.T))

################################
# varp_lkhd<-function(y,phi,sigma,sigma.inv){
#   m = nrow(y);n = ncol(y); 
#   y1 = y[, 1:(n-1)]
#   y2 = y[,2:n] - phi%*%y1  
#   gp = matrix(solve((diag(m^2) - phi%x%phi),as.vector(sigma),tol=1e-40),m,m)
#   q = sum(diag( solve(gp,as.vector(y[,1]),tol=1e-30)%*%as.vector(y[,1]))) +  
#     sum(diag(crossprod(sigma.inv,y2)%*%t(y2) ))
#   l = -m*n*log(2*pi)/2 - (log(det(gp)) + (n-1)*log(det(sigma)))/2 - q/2 
#   return(l)
# }

def varp_lkhd(y, phi, sigma, sigma_inv):
    m, n = y.shape
    y1 = y[:, :n-1]
    y2 = y[:, 1:] - phi.dot(y1)
    gp = (la.solve(np.eye(m**2) - np.kron(phi, phi), sigma.reshape((-1, 1), order="F"))).reshape(m, m)
    _, logdet_gp = la.slogdet(gp)
    _, logdet_sigma = la.slogdet(sigma)
    g = np.sum(np.diag(la.solve(gp, y[:, 0]).dot(y[:, 0]))) + np.sum(np.diag(sigma_inv.T.dot(y2).dot(y2.T)))   #diag -> diagonal
    l = - m * n * np.log(2 * np.pi) / 2 - logdet_gp + (n - 1) * logdet_sigma / 2 - q / 2
    return l

# varppre_lkhd <- function(y,pre,delta,sigma,sigma.inv){
#   m = dim(y)[1]
#   p = length(delta)
#   phi = pre2par_varp(pre,delta)
#   l = -varp_lkhd(y,phi,sigma,sigma.inv)
#   if (is.nan(l)){l = 1e+5}
#   return(l)
# }

def varppre_lkhd(y, pre, delta, sigma, sigma_inv):
    m = y.shape[0]
    p = delta.size #change to len
    phi = pre2par_varp(pre, delta)
    l = - varp_lkhd(y, phi, sigma, sigma_inv)
    if np.isnan(l):
        l = 1e5
    return l

# pre2par_varp <- function(pre,delta){
#   m = sqrt(length(pre))
#   v = matrix(0,m,m)
#   q = v
#   l = diag(m)
#   l[lower.tri(l)] = pre[1:choose(m,2)]
#   d = diag(exp(pre[(choose(m,2)+1):choose(m+1,2)]))
#   v = l%*%d%*%t(l)
#   s= diag(0,m)
#   s[lower.tri(s)] = pre[(choose(m+1,2)+1):m^2]
#   s = s - t(s)
#   q = diag(c(delta,rep(1,(m-1))))%*%(diag(m) - s)%*%solve(diag(m) + s) 
#   phi = VQ2par(v,q)
#   return(phi)
# }

def pre2par_varp( pre, delta):
    m = np.sqrt(pre.size)   #change to len
    v = np.zeros((m, m)) #why?
    q = v #why?
    l = np.eye(m)
    l[np.tril_indices(m)] = pre[:binom(m, 2)]
    d = np.diag(np.exp(pre[binom(m, 2):binom(m+1, 2)]))   #diag -> diagonal, lost +1
    v = l.dot(d).dot(l.T)
    s = np.zeros((m, m)) #unknown function
    s[np.tril_indices(m)] = pre[binom(m+1, 2):m**2]
    s = s - s.T
    q = np.diag(np.r[delta, np.ones(m-1)]).dot(np.eye(m) - s).dot(la.inv(np.eye(m) + s))
    phi = VQ2par(v, q)
    return phi

# # use v, q transform phi
# VQ2par <- function(v,q){
#   m = dim(v)[1]
#   phi = matrix(0,m,m)
#   # u1 = diag(m) + v
#   u1 = diag(m)*0.9 + v
#   u2 = sqrtm(v)%*%q%*%sqrtm(u1)
#   txi = u2
#   bigu = u1
#   phi = txi%*%solve(bigu)
#   return(phi)
# }

def VQ2par(v, q):
    m = v.shape[0]
    phi = np.zeros((m, m))
    u1 = np.eye(m) * 0.9 + v
    u2 = sqrtm(v).dot(q).dot(sqrtm(u1))
    txi = u2
    bigu = u1
    phi = txi.dot(la.inv(bigu))
    return phi

# pre2VQ <- function(pre,delta){
#   m = sqrt(length(pre)) 
#   l = diag(m)
#   l[lower.tri(l)] = pre[1:choose(m,2)] 
#   d = diag(exp(pre[(choose(m,2)+1):choose(m+1,2)]))
#   v = l%*%d%*%t(l)
#   s = diag(0,m)
#   s[lower.tri(s)] = pre[(choose(m+1,2)+1):(m^2)]
#   s = s - t(s)
#   # q = diag(c(delta,rep(1,(m-1))))%*%(diag(m) - s)%*%solve(diag(m) + s) %*%
#   #  ((diag(m) - s)%*%solve(diag(m) + s))
#   q <- diag(c(delta,rep(1,(m-1))))%*%(diag(m) - s)%*%solve(diag(m) + s)
#   return(list(v = v, q=q))
# }


def pre2VQ( pre, delta):
    m = np.sqrt(pre.size)   #change to len
    l = np.eye(m)
    l[np.tril_indices(m)] = pre[:binom(m, 2)]
    d = np.diag(np.exp(pre[binom(m, 2):binom(m+1, 2)]))   #diag -> diagonal, lost +1
    v = l.dot(d).dot(l.T)
    s = np.zeros((m, m)) #unknown function
    s[np.tril_indices(m)] = pre[binom(m+1, 2):m**2]
    s = s - s.T
    q = np.diag(np.r[delta, np.ones(m-1)]).dot(np.eye(m) - s).dot(la.inv(np.eye(m) + s))
    phi = VQ2par(v, q)
    return v, q

# V2LDL <-function(v){
#   m = dim(v)[1]
#   pre = rep(0,choose(m+1,2))
#   c = chol(v, tol = 1e-40)
#   d = diag(c)
#   l = t(c/d);
#   pre[1:choose(m,2)] = l[lower.tri(l)]
#   pre[(choose(m,2)+1):choose(m+1,2)] = log(d^2)
#   return(pre)
# }

def V2LDL(v):
    m = v.shape[0]
    pre = np.zeros(binom(m+1, 2))
    c = la.cholesky(v).T #la.cholesky() returns lower triangular, not upper triangular
    d = np.diagonal(c)
    l = c.T / d
    pre[:binom(m, 2)] = l[np.tril_indices(m)]
    pre[binom(m, 2):binom(m+1, 2)] = np.log(d**2)
    return pre

# initial_var <- function(phi, Shrink = 1, Delta = 0.01){
#   # Function initializes VARMA(p,q) fit using a VAR(L) and then from the residuals 
#   # estimates a VMA(q) to give a VARMA(p,q) estimate
#   # Shink allows projection to stationary space, set to 1
#   # Inputs: 
#   #    y: m x n data matrix
#   #    L: integer for VAR(L) fit
#   #    p: integer 
#   #    q: integer
#   #    Shrink: 1 if you want to constrain
#   #    delta: small epsilon-style number (distance away from boundary)
#   # Outputs:
#   #    phi: m x m x p Array
#   #    theta: m x m x q Array
#   #    Sigma: m x m positive definite covariance matrix
#   m <- dim(phi)[1] 
#   if(max(abs(eigen(phi)$value))>=1){ 
#     phi = phi/((max(abs(eigen(phi)$value)) + Delta))  
#   }
#   return(phi=phi)
# }    

def initial_var(phi, Shrink = 1, Delta = 0.01):
    m = phi.shape[0]
    w = np.max(np.abs(la.eigenvals(phi)))
    if w >= 1:
        phi /= w + Delta
    return phi    