import numpy as np
from scipy import optimize
from scipy.special import expit
def fit(X, y, K):
    """Proportional-odds cumulative logit: P(y<=j)=F(th_j - xb). y in 0..K-1."""
    X=np.asarray(X,float); y=np.asarray(y,int); n,p=X.shape
    def unpack(w):
        return w[:p], np.cumsum(np.r_[w[p],np.exp(w[p+1:])])
    def nll_grad(w):
        b,th=unpack(w); xb=X@b; te=np.r_[-np.inf,th,np.inf]
        Fu=expit(te[y+1]-xb); Fl=expit(te[y]-xb); pr=np.clip(Fu-Fl,1e-300,None)
        fu=Fu*(1-Fu); fl=Fl*(1-Fl)
        gb=-(X*((fu-fl)/pr)[:,None]).sum(0)
        gth=np.array([(fu[y==j]/pr[y==j]).sum()-(fl[y==j+1]/pr[y==j+1]).sum() for j in range(K-1)])
        J=np.zeros((K-1,K-1)); J[:,0]=1
        for k in range(1,K-1): J[k:,k]=np.exp(w[p+k])
        return -np.log(pr).sum(), -np.r_[gb,J.T@gth]
    w0=np.zeros(p+K-1)
    r=optimize.minimize(nll_grad,w0,jac=True,method='BFGS',options={'gtol':1e-6,'maxiter':20000})
    r=optimize.minimize(nll_grad,r.x,jac=True,method='L-BFGS-B',options={'gtol':1e-9,'ftol':1e-15,'maxiter':20000})
    gnorm=np.abs(nll_grad(r.x)[1]).max()
    b,th=unpack(r.x)
    def g_bt(v):
        c=np.r_[v[p],np.log(np.diff(v[p:]))]; f,g=nll_grad(np.r_[v[:p],c])
        J=np.zeros((K-1,K-1)); J[:,0]=1
        for k in range(1,K-1): J[k:,k]=np.exp(c[k])
        return np.r_[g[:p],np.linalg.solve(J.T,g[p:])]
    v=np.r_[b,th]; H=np.zeros((len(v),)*2); h=1e-5
    for i in range(len(v)):
        e=np.zeros(len(v)); e[i]=h; H[:,i]=(g_bt(v+e)-g_bt(v-e))/(2*h)
    cov=np.linalg.inv((H+H.T)/2)
    return dict(b=b,th=th,se=np.sqrt(np.diag(cov))[:p],cov=cov[:p,:p],ll=-r.fun,conv=bool(gnorm<1e-3),n=n,k=len(v))
