import argparse,json,time
import numpy as np

def solve(d,t):
 C=np.asarray(d['cost_matrix'],dtype=float);n=d['n'];p=d['p'];deadline=time.monotonic()+max(.1,t-1);rng=np.random.default_rng(2011);best=None;bv=float('inf')
 def scores(remaining):
  values=np.zeros(n)
  for start in range(0,n,64):
   end=min(n,start+64);values+=np.minimum(remaining[start:end,None],C[start:end]).sum(axis=0)
  return values
 def improve(opened):
  opened=list(opened)
  while time.monotonic()<deadline:
   selected=C[:,opened];nearest=selected.min(axis=1);cur=nearest.sum();gain=0;move=None
   for pos in range(p):
    others=np.delete(selected,pos,axis=1);remaining=others.min(axis=1) if p>1 else np.full(n,np.inf);values=scores(remaining);values[opened]=np.inf;j=int(values.argmin());g=cur-values[j]
    if g>gain+1e-8:gain=g;move=(pos,j)
   if move is None:break
   opened[move[0]]=move[1]
  return opened,float(C[:,opened].min(axis=1).sum())
 greedy=[];nearest=np.full(n,np.inf)
 for _ in range(p):
  values=scores(nearest);values[greedy]=np.inf;j=int(values.argmin());greedy.append(j);nearest=np.minimum(nearest,C[:,j])
 best,bv=improve(greedy)
 for restart in range(100000):
  if time.monotonic()>deadline:break
  if restart%3==0:
   opened=best.copy();positions=rng.choice(p,size=min(2,p),replace=False);remaining=[i for i in range(n) if i not in opened];replacement=rng.choice(remaining,size=len(positions),replace=False)
   for pos,j in zip(positions,replacement):opened[int(pos)]=int(j)
  else:opened=rng.choice(n,size=p,replace=False).tolist()
  opened,v=improve(opened)
  if v<bv:bv=v;best=opened
 best=sorted(best);assign=[{'customer':i,'facility':min(best,key=lambda j:C[i,j]),'cost':int(min(C[i,j] for j in best))} for i in range(n)]
 return {'objective_value':sum(a['cost'] for a in assign),'opened_facilities':best,'assignments':assign}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')
