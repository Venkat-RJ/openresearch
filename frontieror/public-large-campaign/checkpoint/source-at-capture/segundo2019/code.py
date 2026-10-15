import argparse,json,time,random,os
os.environ['OMP_NUM_THREADS']='1'
def solve(d,limit,solution_path=None):
 edgeweighted='graph' in d and 'vertex_weights' not in d
 if 'graph' in d:v=d['graph']['vertices']; edges=d['graph']['edges']
 else:v=list(range(d['n']));edges=d['edges']
 n=len(v);pos={u:i for i,u in enumerate(v)};adj=[0]*n;ew={}
 for e in edges:
  i,j=pos[e[0]],pos[e[1]];adj[i]|=1<<j;adj[j]|=1<<i
  if edgeweighted:ew[min(i,j),max(i,j)]=e[2]
 weights=d.get('vertex_weights',[1]*n); best=[];bestval=-1;deadline=time.monotonic()+max(.01,limit-.5)
 def persist():
  if solution_path:
   out={'objective_value':bestval,'clique_vertices':sorted(v[i] for i in best)}
   if edgeweighted:out['clique_edges']=[[u,w] for ix,u in enumerate(out['clique_vertices']) for w in out['clique_vertices'][ix+1:]]
   with open(solution_path,'w') as f:json.dump(out,f)
 def score(c):return sum(ew.get((min(i,j),max(i,j)),0) for ix,i in enumerate(c) for j in c[ix+1:]) if edgeweighted else sum(weights[i] for i in c)
 rng=random.Random(0)
 for rep in range(min(100,n+1)):
  cand=(1<<n)-1;c=[]
  while cand:
   if time.monotonic()>deadline:break
   choices=[];r=cand
   while r:b=r&-r;i=b.bit_length()-1;r-=b;choices.append(i)
   i=max(choices,key=lambda i:((adj[i]&cand).bit_count() if rep==0 else weights[i]*rng.uniform(.5,1.5)))
   c.append(i);cand&=adj[i]
  z=score(c)
  if z>bestval:best,bestval=c,z;persist()
  if time.monotonic()>deadline:break
 def expand(c,P,value):
  nonlocal best,bestval
  if time.monotonic()>deadline:return
  if edgeweighted:
   vertices=[];r=P
   while r:b=r&-r;r-=b;vertices.append(b.bit_length()-1)
   upper=value+sum(ew.get((min(i,j),max(i,j)),0) for i in vertices for j in c)+sum(ew.get((min(i,j),max(i,j)),0) for k,i in enumerate(vertices) for j in vertices[k+1:])
   if upper<=bestval:return
   order=sorted(vertices,key=lambda i:sum(ew.get((min(i,j),max(i,j)),0) for j in vertices if j!=i))
   bounds=None
  else:
   order=[];bounds=[];rem=P;total=0
   while rem:
    if time.monotonic()>deadline:return
    Q=rem;col=[];mx=0
    while Q:
     if time.monotonic()>deadline:return
     b=Q&-Q;i=b.bit_length()-1;Q-=b;rem-=b;Q&=~adj[i];col.append(i);mx=max(mx,weights[i])
    total+=mx
    for i in col:order.append(i);bounds.append(total)
  for k in range(len(order)-1,-1,-1):
   if bounds is not None and value+bounds[k]<=bestval:return
   i=order[k];inc=sum(ew.get((min(i,j),max(i,j)),0) for j in c) if edgeweighted else weights[i];nv=value+inc
   cc=c+[i]
   if nv>bestval:best,bestval=cc,nv;persist()
   q=P&adj[i]
   if q:expand(cc,q,nv)
   P&=~(1<<i)
   if time.monotonic()>deadline:return
 expand([], (1<<n)-1,0)
 out={'objective_value':bestval,'clique_vertices':sorted(v[i] for i in best)}
 if edgeweighted:out['clique_edges']=[[u,w] for ix,u in enumerate(out['clique_vertices']) for w in out['clique_vertices'][ix+1:]]
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 out=solve(json.load(open(a.instance_path)),a.time_limit,a.solution_path)
 with open(a.solution_path,'w') as f:json.dump(out,f)
