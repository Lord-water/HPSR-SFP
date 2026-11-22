"""Paired context-cluster bootstrap/randomization and Holm correction."""
import numpy as np

def paired(p,b,y,groups,reps=2000):
 mask=y>=0;loss=(((p-y)**2-(b-y)**2)*mask);count=mask.sum(0);n=len(y)
 unique,gi=np.unique(groups,return_inverse=True);g=len(unique);rng=np.random.default_rng(20261117)
 grouped=np.zeros((g,12));gm=np.zeros((g,12));np.add.at(grouped,gi,loss);np.add.at(gm,gi,mask)
 draws=rng.integers(0,g,(reps,g));w=np.zeros((reps,g));np.add.at(w,(np.repeat(np.arange(reps),g),draws.ravel()),1)
 den=w@gm;bs=np.mean((w@grouped)/np.maximum(den,1),axis=1)
 flips=rng.choice([-1.,1.],size=(reps,g));perms=np.mean((flips@grouped)/np.maximum(count,1),axis=1)
 mean=float(np.mean(loss.sum(0)/np.maximum(count,1)))
 return dict(macro_brier_difference=mean,context_cluster_bootstrap95=np.quantile(bs,[.025,.975]).tolist(),two_sided_randomization_p=float((1+(np.abs(perms)>=abs(mean)-1e-15).sum())/(reps+1)),resamples=reps,clusters=g,scope='Exploratory; conditional on frozen selected models, excludes training/selection uncertainty')

def holm(ds):
 order=sorted(ds,key=lambda k:ds[k]['two_sided_randomization_p']);prior=0.
 for j,k in enumerate(order):prior=max(prior,min(1.,(len(order)-j)*ds[k]['two_sided_randomization_p']));ds[k]['holm_p']=prior
