"""Candidate-conditioned encoder; scientific class/function bodies preserved."""
import math, random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from . import LABELS

def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)

class LoRALinear(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base;self.a=nn.Parameter(torch.empty(8,base.in_features,dtype=torch.float32));self.b=nn.Parameter(torch.zeros(base.out_features,8,dtype=torch.float32));nn.init.kaiming_uniform_(self.a,a=math.sqrt(5));self.drop=nn.Dropout(.05)
 def forward(self,x):return self.base(x)+(F.linear(F.linear(self.drop(x).float(),self.a),self.b)*2).to(x.dtype)

class UsageEncoder(nn.Module):
 def __init__(self,path,device='cuda',adapter=True):
  super().__init__();self.encoder=AutoModel.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16 if device=='cuda' else torch.float32,attn_implementation='sdpa');self.encoder.config.use_cache=False
  for p in self.encoder.parameters():p.requires_grad_(False)
  self.adapted=[]
  if adapter:
   for n,m in list(self.encoder.named_modules()):
    if n.endswith(('q_proj','v_proj','query','value')) and isinstance(m,nn.Linear):
     parent,child=n.rsplit('.',1);setattr(self.encoder.get_submodule(parent),child,LoRALinear(m));self.adapted.append(n)
   assert self.adapted
  self.usage=nn.Linear(self.encoder.config.hidden_size,1,bias=False,dtype=torch.float32);self.to(device)
  if device=='cuda' and adapter:self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 def forward(self,ids,mask,pos):
  h=self.encoder(input_ids=ids,attention_mask=mask).last_hidden_state
  v=h[torch.arange(len(ids),device=ids.device),pos].float()
  return self.usage(v).squeeze(-1),v
 def state(self):return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}
 def restore(self,s):
  assert set(s)=={n for n,p in self.named_parameters() if p.requires_grad};self.load_state_dict(s,strict=False)

def tokenize(tok,r,limit=256):
 causal=tok.cls_token_id is None
 pre=tok.encode('对话：\n'+'\n'.join(r['history'])+'\n回复：',add_special_tokens=False)
 body=tok.encode(r['target_body'],add_special_tokens=False);suffixes=[tok.encode(k,add_special_tokens=False) for k in LABELS]
 end=tok.encode(r.get('target_punctuation',''),add_special_tokens=False)
 budget=limit-max(map(len,suffixes))-len(end)-(2 if not causal else 0)
 assert budget>16
 if len(body)>budget-8:body=body[:(budget-8)//2]+body[-((budget-8)-(budget-8)//2):]
 pre=pre[-max(0,budget-len(body)):];base=pre+body
 seq=[]
 for k in suffixes:
  ids=base+k+end;pos=len(base+k)-1
  if not causal:ids=[tok.cls_token_id]+ids+[tok.sep_token_id];pos+=1
  assert len(ids)<=limit;seq.append((ids,pos))
 return seq

def batch(seq,pad,device='cuda'):
 n=max(len(x[0]) for x in seq);ids=torch.full((len(seq),n),pad,dtype=torch.long);mask=torch.zeros_like(ids);pos=[]
 for j,(x,k) in enumerate(seq):ids[j,:len(x)]=torch.tensor(x);mask[j,:len(x)]=1;pos.append(k)
 return ids.to(device),mask.to(device),torch.tensor(pos,device=device)

def usage_loss(logits,target):return F.cross_entropy(logits.reshape(-1,12),target)
