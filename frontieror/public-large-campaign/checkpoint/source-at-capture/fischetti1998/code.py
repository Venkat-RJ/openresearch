import argparse,json,ast,time,random

def heuristic(d,t):
 start=time.monotonic();deadline=start+max(.1,t-5)
 ids=sorted(int(k) for k in d['prizes']);n=len(ids);idx={v:i for i,v in enumerate(ids)};dep=idx[d['depot']]
 c=[[0]*n for _ in ids]
 for k,v in d['travel_times'].items():
  a,b=ast.literal_eval(k);i,j=idx[a],idx[b];c[i][j]=c[j][i]=int(v)
 p=[d['prizes'][str(v)] for v in ids];budget=d['t0'];others=[i for i in range(n) if i!=dep]
 seed=None;seedcost=float('inf')
 for a in others:
  for b in others:
   if a>=b:continue
   z=c[dep][a]+c[a][b]+c[b][dep]
   if z<=budget and (seed is None or (p[a]+p[b],-z)>(p[seed[1]]+p[seed[2]],-seedcost)):
    seed=[dep,a,b];seedcost=z
 if seed is None:raise RuntimeError('No feasible triangle')
 rng=random.Random(0);best=seed[:];bestvalue=sum(p[i] for i in best)
 def length(r):return sum(c[r[k]][r[(k+1)%len(r)]] for k in range(len(r)))
 def improve(r):
  z=length(r);changed=True
  while changed and time.monotonic()<deadline:
   changed=False
   for i in range(len(r)-2):
    for j in range(i+2,len(r)):
     if i==0 and j==len(r)-1:continue
     a,b=r[i],r[i+1];u,v=r[j],r[(j+1)%len(r)]
     delta=c[a][u]+c[b][v]-c[a][b]-c[u][v]
     if delta<0:r[i+1:j+1]=reversed(r[i+1:j+1]);z+=delta;changed=True
   if time.monotonic()>=deadline:break
  return z
 iteration=0
 while time.monotonic()<deadline:
  if iteration==0:r=seed[:]
  else:
   r=best[:];remove=max(1,int((len(r)-1)*rng.uniform(.08,.35)))
   discard=set(rng.sample(r[1:],min(remove,len(r)-3)));r=[i for i in r if i not in discard]
  z=improve(r);remaining=set(others)-set(r);weights={v:rng.uniform(.75,1.25) if iteration else 1 for v in remaining};power=rng.uniform(.65,1.3) if iteration else 1
  while remaining and time.monotonic()<deadline:
   choice=None;score=-1
   for v in remaining:
    delta,k=min((c[r[k]][v]+c[v][r[(k+1)%len(r)]]-c[r[k]][r[(k+1)%len(r)]],k) for k in range(len(r)))
    if z+delta<=budget:
     rank=weights[v]*p[v]/max(1,delta)**power
     if rank>score:score=rank;choice=(v,k,delta)
   if choice is None:break
   v,k,delta=choice;r.insert(k+1,v);remaining.remove(v);z+=delta
   if len(r)%10==0:z=improve(r)
  value=sum(p[i] for i in r)
  if value>bestvalue or (value==bestvalue and length(r)<length(best)):best=r[:];bestvalue=value
  iteration+=1
 tour=[ids[i] for i in best]
 return {'objective_value':bestvalue,'visited_nodes':tour,'edges':[[tour[i],tour[(i+1)%len(tour)]] for i in range(len(tour))],'tour':tour+[tour[0]]}
from ortools.sat.python import cp_model

def solve(d,t):
 if len(d['prizes'])>80:return heuristic(d,t)
 ids=sorted(int(k) for k in d['prizes']);idx={v:i for i,v in enumerate(ids)};n=len(ids);dep=idx[d['depot']];cost={}
 for k,v in d['travel_times'].items():
  i,j=ast.literal_eval(k);cost[idx[i],idx[j]]=cost[idx[j],idx[i]]=int(v)
 m=cp_model.CpModel();y=[m.new_bool_var(f'y{i}') for i in range(n)];m.add(y[dep]==1);m.add(sum(y)>=3);arcs=[(i,i,y[i].Not()) for i in range(n)];edges={}
 for i in range(n):
  for j in range(n):
   if i==j:continue
   e=m.new_bool_var(f'e{i}_{j}');edges[i,j]=e;arcs.append((i,j,e))
 m.add_circuit(arcs);m.add(sum(cost[i,j]*e for (i,j),e in edges.items())<=d['t0']);m.maximize(sum(int(d['prizes'][str(v)])*y[i] for i,v in enumerate(ids)));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,t-1);r=s.solve(m)
 if r not in (cp_model.FEASIBLE,cp_model.OPTIMAL):raise RuntimeError('No feasible tour')
 nxt={i:j for (i,j),e in edges.items() if s.value(e)};tour=[dep]
 while nxt[tour[-1]]!=dep:tour.append(nxt[tour[-1]])
 tour=[ids[i] for i in tour];return {'objective_value':sum(d['prizes'][str(v)] for v in tour),'visited_nodes':tour,'edges':[[tour[i],tour[(i+1)%len(tour)]] for i in range(len(tour))],'tour':tour+[tour[0]]}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=__import__('time').time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':__import__('time').time()-start,'objective_value':r['objective_value']})+'\n')
