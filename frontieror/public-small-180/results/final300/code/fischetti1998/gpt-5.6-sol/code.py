import argparse,json,time,random,ast,os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+max(.1,a.time_limit-4)
 z=json.load(open(a.instance_path));n=z['n'];depot=z['depot']-1;budget=z['t0'];d=np.zeros((n,n),dtype=np.int64)
 for key,v in z['travel_times'].items():i,j=ast.literal_eval(key);d[i-1,j-1]=d[j-1,i-1]=v
 prize=np.array([z['prizes'][str(i+1)] for i in range(n)]);logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None;rng=random.Random(0);best=[];bestobj=-1
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 def length(t):return int(sum(d[t[k-1],t[k]] for k in range(len(t))))
 def save(t):
  nonlocal best,bestobj
  val=int(prize[t].sum())
  if len(t)>=3 and length(t)<=budget and val>bestobj:
   best=t[:];bestobj=val;out=[v+1 for v in t];sol={'objective_value':val,'visited_nodes':out,'edges':[[out[k-1],out[k]] for k in range(len(out))],'tour':out+[out[0]],'runtime':time.monotonic()-start};json.dump(sol,open(a.solution_path,'w'))
   if logger:logger.log_solution(val,sol)
 def shorten(t):
  changed=True
  while changed and time.monotonic()<end:
   changed=False
   for i in range(len(t)-2):
    for j in range(i+2,len(t)):
     if i==0 and j==len(t)-1:continue
     aa,bb,cc,ee=t[i],t[i+1],t[j],t[(j+1)%len(t)]
     if d[aa,cc]+d[bb,ee]<d[aa,bb]+d[cc,ee]:t[i+1:j+1]=t[i+1:j+1][::-1];changed=True
  return t
 def fill(t):
  while time.monotonic()<end:
   used=set(t);candidates=[v for v in range(n) if v not in used]
   if not candidates:break
   q=np.array(t);cand=np.array(candidates);delta=d[q[:,None],cand[None,:]]+d[cand[None,:],np.roll(q,-1)[:,None]]-d[q,np.roll(q,-1)][:,None];idx=np.argmin(delta,axis=0);extra=delta[idx,np.arange(len(cand))];scores=prize[cand]/np.maximum(1,extra);scores[extra+length(t)>budget]=-1
   k=int(np.argmax(scores))
   if scores[k]<0:break
   t.insert(int(idx[k])+1,int(cand[k]));save(t);shorten(t)
  return t
 if n<=17:
  nodes=[v for v in range(n) if v!=depot];dp={};parent={};bmask=0;blast=0;bval=-1
  for k,v in enumerate(nodes):dp[(1<<k,k)]=int(d[depot,v]);parent[(1<<k,k)]=-1
  for mask in range(1,1<<len(nodes)):
   if time.monotonic()>=end:break
   value=int(sum(prize[nodes[k]] for k in range(len(nodes)) if mask>>k&1))
   for k,v in enumerate(nodes):
    if not mask>>k&1:continue
    state=(mask,k)
    if state not in dp:continue
    old=dp[state]
    if mask.bit_count()>=2 and old+d[v,depot]<=budget and value>bval:bmask,blast,bval=mask,k,value
    for j,u in enumerate(nodes):
     if mask>>j&1:continue
     new=old+int(d[v,u]);key=(mask|1<<j,j)
     if new<=budget and new<dp.get(key,10**18):dp[key]=new;parent[key]=k
  if bval>=0:
   tour=[];mask,k=bmask,blast
   while k>=0:tour.append(nodes[k]);prev=parent[(mask,k)];mask^=1<<k;k=prev
   save([depot]+tour[::-1]);return
 nodes=[v for v in range(n) if v!=depot];tri=d[depot,np.array(nodes)][:,None]+d[np.ix_(nodes,nodes)]+d[np.array(nodes),depot][None,:];np.fill_diagonal(tri,10**15);indices=np.argwhere(tri<=budget)
 if len(indices):i,j=max(indices,key=lambda ij:prize[nodes[int(ij[0])]]+prize[nodes[int(ij[1])]]);t=[depot,nodes[int(i)],nodes[int(j)]];save(t);fill(t)
 while best and time.monotonic()<end:
  t=best[:];count=min(len(t)-1,max(1,len(t)//5));remove=set(rng.sample(t[1:],count));t=[v for v in t if v not in remove];fill(t);save(t)
if __name__=='__main__':main()
