"""Same-grid frozen-feature controls; mathematical bodies preserved."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
L2=[.01,.03,.1,.3,1.,3.,10.,30.,100.]

def scores(p,y):
 from sklearn.metrics import roc_auc_score
 br=[];au=[];counts=[]
 for k in range(12):
  m=y[:,k]>=0;t=y[m,k];v=p[m,k];br.append(float(((v-t)**2).mean()))
  au.append(float(roc_auc_score(t,v)) if len(set(t.tolist()))==2 else None);counts.append({'A':int((t==1).sum()),'N':int((t==0).sum()),'U':int((~m).sum())})
 mask=y>=0;v=p[mask];t=y[mask];bins=np.minimum((v*10).astype(int),9)
 ece=sum(float(np.mean(bins==b))*abs(float(v[bins==b].mean()-t[bins==b].mean())) for b in range(10) if np.any(bins==b))
 return {'macro_brier':float(np.mean(br)),'macro_auroc':float(np.mean([x for x in au if x is not None])),'auroc_supported_candidates':sum(x is not None for x in au),'pooled_ece_10_equal_width_bins':ece,'per_candidate_brier':br,'per_candidate_auroc':au,'support':counts}

def prepare(h,u,arm,trainh=None,stats=None):
 h=np.asarray(h,dtype='float64');u=np.asarray(u,dtype='float64')
 if arm in ['full','center_linear','center_no_pair','base']:h=h-h.mean(1,keepdims=True)
 if arm in ['scalar_flex','scalar_mono']:
  x=u[:,:,None]
  if stats is None:
   mean=x.mean((0,1));scale=x.std((0,1)).clip(1e-5);knots=np.quantile(((x-mean)/scale).reshape(-1),[.1,.25,.5,.75,.9]);stats={'mean':mean,'scale':scale,'knots':knots}
  x=(x-stats['mean'])/stats['scale']
  if arm=='scalar_flex':x=np.concatenate([x,np.maximum(x-stats['knots'][None,None,:],0)],-1)
 else:
  if stats is None:stats={'mean':h.mean((0,1)),'scale':h.std((0,1)).clip(1e-5)}
  x=(h-stats['mean'])/stats['scale']
 return x,stats

def fit(x,y,l2,gamma,mono=False):
 n,k,d=x.shape;mask=y>=0;den=np.maximum(mask.sum(0),1);weight=mask/den[None,:]/12
 pairs=[(i,np.where(y[i]==1)[0],np.where(y[i]==0)[0]) for i in range(n) if np.any(y[i]==1) and np.any(y[i]==0)]
 def fun(v):
  w=v[:d];b=v[d:];z=x@w+b;p=expit(z);loss=float(((np.logaddexp(0,z)-np.maximum(y,0)*z)*weight).sum())+l2*float(w@w)/2+.01*float(b@b)/2
  dz=(p-np.maximum(y,0))*weight
  if gamma and pairs:
   for i,a,neg in pairs:
    delta=z[i,a][:,None]-z[i,neg][None,:];factor=gamma/len(pairs)/delta.size
    loss+=factor*float(np.logaddexp(0,-delta).sum());g=-factor*expit(-delta);dz[i,a]+=g.sum(1);dz[i,neg]-=g.sum(0)
  return loss,np.r_[np.einsum('nkd,nk->d',x,dz)+l2*w,dz.sum(0)+.01*b]
 bounds=[(0,None)]*d+[(None,None)]*12 if mono else None
 r=minimize(fun,np.zeros(d+12),jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':200,'gtol':1e-7,'ftol':1e-11})
 assert np.isfinite(r.fun) and np.isfinite(r.x).all()
 return {'w':r.x[:d],'b':r.x[d:],'l2':l2,'gamma':gamma,'success':bool(r.success),'nit':int(r.nit),'grad_max':float(np.abs(r.jac).max())}

def predict(m,x):return x@m['w']+m['b']

def calibrate(z,y,mode,l2=1.):
 # Eligible candidate offsets only; same5A/5N rule used by historical calibration.
 mask=y>=0;den=np.maximum(mask.sum(0),1);weight=mask/den[None,:]/12
 eligible=((y==0).sum(0)>=5)&((y==1).sum(0)>=5);ix=np.where(eligible)[0]
 if mode=='raw':return {'slope':1.,'bias':np.zeros(12),'mode':mode}
 dim=1 if mode=='temp' else 2 if mode=='affine' else 2+len(ix)
 def fun(v):
  a=np.exp(v[0]);b=np.zeros(12)
  if mode!='temp':b[:]=v[1]
  if mode=='candidate':b[ix]+=v[2:]
  t=a*z+b;p=expit(t);dz=(p-np.maximum(y,0))*weight;val=float(((np.logaddexp(0,t)-np.maximum(y,0)*t)*weight).sum())
  g=[float((dz*a*z).sum())]
  if mode!='temp':g.append(float(dz.sum()))
  if mode=='candidate':val+=l2*float(v[2:]@v[2:])/2;g.extend((dz.sum(0)[ix]+l2*v[2:]).tolist())
  return val,np.array(g)
 res=minimize(fun,np.zeros(dim),jac=True,method='L-BFGS-B',bounds=[(-5,5)]+[(None,None)]*(dim-1),options={'maxiter':200,'ftol':1e-12})
 a=np.exp(res.x[0]);b=np.zeros(12)
 if mode!='temp':b[:]=res.x[1]
 if mode=='candidate':b[ix]+=res.x[2:]
 return {'slope':a,'bias':b,'mode':mode,'eligible':eligible,'regularization':l2,'converged':bool(res.success)}

def apply_cal(c,z):return expit(c['slope']*z+c['bias'])
