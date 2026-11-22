"""Stabilized HPSR, calibration and fixed selection rules."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .readouts import calibrate, apply_cal

def objective(v,x,u,ub,y,lam,rho,cues):
 n,k,d=x.shape;w=v[:d];off=d
 delta=np.zeros((k,d)) if np.isinf(rho) else v[off:off+k*d].reshape(k,d)
 if not np.isinf(rho):off+=k*d
 b=v[off:off+k];a,be=v[-2:] if cues else (0.,0.)
 z=np.einsum('nkd,kd->nk',x,w[None,:]+delta)+b+a*u+be*ub[:,None]
 mask=y>=0;weight=mask/np.maximum(mask.sum(0),1)/k;t=np.maximum(y,0)
 loss=float(((np.logaddexp(0,z)-t*z)*weight).sum())+lam*float(w@w)+.005*float(b@b)
 dz=(expit(z)-t)*weight;g=np.einsum('nkd,nk->kd',x,dz)
 grad=[g.sum(0)+2*lam*w]
 if not np.isinf(rho):loss+=lam*rho*float((delta*delta).sum());grad.append((g+2*lam*rho*delta).ravel())
 grad.append(dz.sum(0)+.01*b)
 if cues:loss+=lam*(a*a+be*be);grad.append(np.array([(dz*u).sum()+2*lam*a,(dz*ub[:,None]).sum()+2*lam*be]))
 return loss,np.concatenate(grad)

def hfit(x,u,ub,y,lam,rho,cues):
 d=x.shape[-1];dim=d+12+(0 if np.isinf(rho) else 12*d)+(2 if cues else 0)
 r=minimize(objective,np.zeros(dim),args=(x,u,ub,y,lam,rho,cues),jac=True,method='L-BFGS-B',options={'maxiter':400,'gtol':1e-7,'ftol':1e-11,'maxcor':10})
 assert np.isfinite(r.fun) and np.isfinite(r.x).all()
 return dict(v=r.x,lam=lam,rho=rho,cues=cues,success=bool(r.success),nit=int(r.nit),grad_max=float(abs(r.jac).max()),loss=float(r.fun))

def hpredict(m,x,u,ub):
 d=x.shape[-1];v=m['v'];off=d
 delta=np.zeros((12,d)) if np.isinf(m['rho']) else v[off:off+12*d].reshape(12,d)
 if not np.isinf(m['rho']):off+=12*d
 z=np.einsum('nkd,kd->nk',x,v[:d][None,:]+delta)+v[off:off+12]
 if m['cues']:z+=v[-2]*u+v[-1]*ub[:,None]
 return z

def hcal(z,y,mode):
 if mode=='temp':return calibrate(z,y,'temp')
 # Candidate offsets shrink to common intercept; both-class sparse support raises precision.
 k=z.shape[1];weight=(y>=0)/np.maximum((y>=0).sum(0),1)/k;t=np.maximum(y,0)
 precision=1+10/np.maximum(np.minimum((y==0).sum(0),(y==1).sum(0)),1)
 def fn(v):
  a=np.exp(v[0]);zz=a*z+v[1]+v[2:];dz=(expit(zz)-t)*weight
  loss=float(((np.logaddexp(0,zz)-t*zz)*weight).sum())+.5*float((precision*v[2:]**2).sum())
  return loss,np.r_[(dz*a*z).sum(),dz.sum(),dz.sum(0)+precision*v[2:]]
 r=minimize(fn,np.zeros(k+2),jac=True,method='L-BFGS-B',bounds=[(-5,5)]+[(None,None)]*(k+1),options={'maxiter':200,'ftol':1e-12})
 return dict(slope=np.exp(r.x[0]),bias=r.x[1]+r.x[2:],mode='candidate_shrunk',precision=precision,converged=bool(r.success))

def oof(z,y,fold,mode,hier=False):
 p=np.zeros_like(z);cal=hcal if hier else calibrate
 for f in range(5):
  m=fold==f;assert m.any();p[m]=apply_cal(cal(z[~m],y[~m],mode),z[m])
 return float(np.mean([np.mean((p[y[:,k]>=0,k]-y[y[:,k]>=0,k])**2) for k in range(12)]))

def rules(p,logq,y):
 s=logq+np.log(p.clip(1e-12,1));feasible=p>=.5;empty=~feasible.any(1)
 gate=np.where(feasible,s,-np.inf).argmax(1);gate[empty]=p[empty].argmax(1)
 indices={'Usage-Only':logq.argmax(1),'Rule-PoE':s.argmax(1),'Rule-BayesGate':gate}
 out={}
 for name,ix in indices.items():
  v=y[np.arange(len(y)),ix];out[name]={'A':int((v==1).sum()),'N':int((v==0).sum()),'U':int((v<0).sum()),'n':len(y),'fallbacks':int(empty.sum()) if name=='Rule-BayesGate' else 0,'indices':ix.tolist()}
 for name in ['Rule-PoE','Rule-BayesGate']:out[name]['N_minus_Usage-Only']=out[name]['N']-out['Usage-Only']['N']
 return out
