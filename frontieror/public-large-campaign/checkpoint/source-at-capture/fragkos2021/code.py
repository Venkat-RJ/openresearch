import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse,json,time,tempfile
import numpy as np

class Reader:
 def __init__(self,f):self.f=f;self.buf='';self.pos=0;self.decoder=json.JSONDecoder()
 def fill(self):
  self.buf=self.buf[self.pos:]+self.f.read(1048576);self.pos=0
 def space(self):
  while True:
   while self.pos<len(self.buf) and self.buf[self.pos].isspace():self.pos+=1
   if self.pos<len(self.buf):return
   old=len(self.buf)-self.pos;self.fill()
   if len(self.buf)==old:raise ValueError('Unexpected JSON EOF')
 def take(self,char):
  self.space()
  if self.buf[self.pos]!=char:raise ValueError('Expected '+char)
  self.pos+=1
 def value(self):
  self.space()
  while True:
   try:value,end=self.decoder.raw_decode(self.buf,self.pos)
   except json.JSONDecodeError:
    old=len(self.buf)-self.pos;self.fill()
    if len(self.buf)==old:raise
   else:self.pos=end;return value

def load(path,root):
 with open(path) as f:
  r=Reader(f);r.take('{');d={};matrix=None;near=None;near_cost=None
  while True:
   key=r.value();r.take(':')
   if key=='transportation_cost_matrix':
    n=d['num_facilities'];nc=d['num_customers'];width=min(64,n);matrix=np.memmap(os.path.join(root,'costs.bin'),mode='w+',dtype=np.float64,shape=(nc,n));near=np.empty((nc,width),dtype=np.int32);near_cost=np.empty((nc,width))
    r.take('[')
    for i in range(nc):
     row=np.asarray(r.value(),dtype=np.float64)
     if len(row)!=n:raise ValueError('Cost matrix width mismatch')
     matrix[i]=row;chosen=np.argpartition(row,width-1)[:width];chosen=chosen[np.argsort(row[chosen])];near[i]=chosen;near_cost[i]=row[chosen]
     if i+1<nc:r.take(',')
    r.take(']')
   else:d[key]=r.value()
   r.space();char=r.buf[r.pos];r.pos+=1
   if char=='}':break
   if char!=',':raise ValueError('Expected object delimiter')
 if matrix is None:raise ValueError('Missing cost matrix')
 return d,matrix,near,near_cost

def expected(d,c,y):
 opened=np.flatnonzero(y);nc=d['num_customers'];q=np.array([d['nodes'][j]['survival_probability'] for j in range(d['num_facilities'])]);dem=np.array([d['nodes'][i]['demand'] for i in range(nc)]);p=np.array(d['penalty_cost']);result=sum(d['nodes'][j]['fixed_setup_cost'] for j in opened)
 if not len(opened):return float(result+dem@p)
 for low in range(0,nc,256):
  high=min(nc,low+256);cost=np.asarray(c[low:high][:,opened]);order=np.argsort(cost,axis=1);cost=np.take_along_axis(cost,order,axis=1);prob=q[opened][order];fail=np.concatenate((np.ones((high-low,1)),np.cumprod(1-prob,axis=1)),axis=1);values=np.sum(cost*prob*fail[:,:-1],axis=1)+p[low:high]*fail[:,-1];result+=float(dem[low:high]@values)
 return float(result)

def solve_path(path,t):
 started=time.monotonic();deadline=started+max(.1,t-7)
 with tempfile.TemporaryDirectory(prefix='fragkos-stream-') as root:
  d,c,near,ncost=load(path,root);n=d['num_facilities'];nc=d['num_customers'];setup=np.array([d['nodes'][j]['fixed_setup_cost'] for j in range(n)]);q=np.array([d['nodes'][j]['survival_probability'] for j in range(n)]);dem=np.array([d['nodes'][i]['demand'] for i in range(nc)]);penalty=np.array(d['penalty_cost']);y=np.zeros(n,dtype=np.int8)
  if n<=20:
   best=y.copy();value=expected(d,c,y);order=np.argsort(c,axis=1);rows=np.arange(nc)
   for low in range(0,1<<n,1024):
    if time.monotonic()>=deadline:break
    masks=np.arange(low,min(1<<n,low+1024),dtype=np.uint32);bits=((masks[:,None]>>np.arange(n,dtype=np.uint32))&1);fail=np.ones((len(masks),nc));shipping=np.zeros_like(fail)
    for rank in range(n):
     j=order[:,rank];surv=q[j][None,:]*bits[:,j];shipping+=np.asarray(c[rows,j])[None,:]*surv*fail;fail*=1-surv
    vals=bits@setup+(shipping+penalty[None,:]*fail)@dem;h=int(vals.argmin())
    if vals[h]<value:value=float(vals[h]);best=bits[h].astype(np.int8)
   y=best
  else:
   width=near.shape[1];survival=q[near]
   while time.monotonic()<deadline:
    active=y[near]*survival;tail=np.empty((nc,width+1));tail[:,width]=penalty
    for rank in range(width-1,-1,-1):tail[:,rank]=active[:,rank]*ncost[:,rank]+(1-active[:,rank])*tail[:,rank+1]
    fail=np.concatenate((np.ones((nc,1)),np.cumprod(1-active,axis=1)[:,:-1]),axis=1)
    gain=survival*fail*(tail[:,1:]-ncost)*dem[:,None];benefit=np.bincount(near.ravel(),weights=gain.ravel(),minlength=n)-setup;delta=np.where(y,-benefit,benefit);j=int(np.argmax(delta))
    if delta[j]<=1e-7:break
    y[j]^=1
  result={'objective_value':expected(d,c,y),'facility_locations':y.tolist()}
  del c
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();started=time.monotonic();result=solve_path(a.instance_path,a.time_limit)
 with open(a.solution_path,'w') as f:json.dump(result,f)
 if a.log_path:
  with open(a.log_path,'w') as f:f.write(json.dumps({'time':time.monotonic()-started,'objective_value':result['objective_value']})+'\n')
