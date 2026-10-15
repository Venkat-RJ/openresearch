import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,random
import numpy as np
from solution_logger import SolutionLogger
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();deadline=start+max(.1,a.time_limit-2)
d=json.load(open(a.instance_path));D=np.array(d['truck_travel_times']);B=np.array(d['drone_battery_consumption']);T=np.array(d['drone_delivery_time']);caps=np.array(d['parameters']['drone_battery_capacities']);depot=d['depot_index'];customers=d['customer_indices'];L=len(caps);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
route=[depot];left=set(customers)
while left:
 j=min(left,key=lambda j:D[route[-1],j]);route.append(j);left.remove(j)
route.append(depot);best=float('inf');rng=random.Random(0)
def optimize(r):
 for _ in range(20):
  change=False
  for i in range(1,len(r)-2):
   for j in range(i+1,len(r)-1):
    if time.monotonic()>deadline:return r
    old=sum(D[r[k],r[k+1]] for k in range(i-1,j+1));nr=r[:i]+r[i:j+1][::-1]+r[j+1:];new=sum(D[nr[k],nr[k+1]] for k in range(i-1,j+1))
    if new<old:r=nr;change=True
  if not change or time.monotonic()>deadline:break
 return r
def search(r):
 assignments=[];used=np.zeros(L);loads={i:np.zeros(L) for i in r[:-1]}
 while len(r)>3 and time.monotonic()<deadline:
  choice=None
  for pos in range(1,len(r)-1):
   j=r[pos]
   if any(z['dispatch_node']==j for z in assignments):continue
   saving=D[r[pos-1],j]+D[j,r[pos+1]]-D[r[pos-1],r[pos+1]]
   for i in r[:-1]:
    if i==j:continue
    for l in range(L):
     if used[l]+B[l,i,j]>caps[l]+1e-8:continue
     extra=max(loads[i].max(),loads[i][l]+T[l,i,j])-loads[i].max();gain=saving-extra
     if gain>1e-8 and (choice is None or gain>choice[0]):choice=(gain,pos,i,l)
  if choice is None:break
  _,pos,i,l=choice;j=r.pop(pos);used[l]+=B[l,i,j];loads[i][l]+=T[l,i,j];assignments.append({'drone':l,'dispatch_node':i,'delivery_node':j})
 return r,assignments,loads
def save(r,h,w):
 global best
 obj=float(sum(D[r[k],r[k+1]] for k in range(len(r)-1))+sum(x.max() for x in w.values()))
 if obj<best:
  best=obj;s={'objective_value':obj,'truck_route':[[r[k],r[k+1] if k+1<len(r)-1 else d['num_nodes']] for k in range(len(r)-1)],'drone_assignments':h,'waiting_times':{str(i):float(w[i].max()) for i in r[:-1]}};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(obj,s)
save(route,[],{i:np.zeros(L) for i in route[:-1]})
route=optimize(route)
save(route,[],{i:np.zeros(L) for i in route[:-1]})
full_deadline=deadline
if len(customers)<=70:deadline=min(deadline,start+1)
for it in range(100000):
 if time.monotonic()>deadline:break
 r=route.copy()
 if it and len(r)>3:
  i,j=sorted(rng.sample(range(1,len(r)-1),2));r[i:j+1]=reversed(r[i:j+1]);r=optimize(r)
 r,h,w=search(r);save(r,h,w)
if len(customers)<=70 and time.monotonic()<full_deadline:
 from ortools.sat.python import cp_model
 model=cp_model.CpModel();sink=d['num_nodes'];arcs=[];truck={};vis={};incoming={j:[] for j in customers};objective=[]
 for j in customers:
  vis[j]=model.new_bool_var(f'visit{j}');arcs.append((j,j,vis[j].Not()))
 for i in [depot]+customers:
  for j in customers+[sink]:
   if i==j or (i==depot and j==sink):continue
   v=model.new_bool_var(f'x{i}_{j}');truck[i,j]=v;arcs.append((i,j,v));objective.append(int(D[i,depot if j==sink else j])*v)
   if j in incoming:incoming[j].append(v)
 arcs.append((sink,depot,model.new_constant(1)));model.add_circuit(arcs)
 drone={};waiting={};batteries=[[] for _ in range(L)];deliveries={j:[] for j in customers}
 for i in [depot]+customers:
  ws=model.new_int_var(0,int(T.max())*len(customers),f'w{i}');waiting[i]=ws;objective.append(ws)
  for l in range(L):
   load=[]
   for j in customers:
    if i==j or B[l,i,j]>caps[l]:continue
    v=model.new_bool_var(f'h{l}_{i}_{j}');drone[l,i,j]=v;deliveries[j].append(v);load.append(int(T[l,i,j])*v);batteries[l].append(int(B[l,i,j])*v)
    if i!=depot:model.add(v<=vis[i])
   model.add(ws>=sum(load))
 for j in customers:model.add(sum(incoming[j])+sum(deliveries[j])==1)
 for l in range(L):model.add(sum(batteries[l])<=int(caps[l]))
 model.minimize(sum(objective));solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.random_seed=0;solver.parameters.max_time_in_seconds=max(.1,full_deadline-time.monotonic())
 class CB(cp_model.CpSolverSolutionCallback):
  def on_solution_callback(self):
   successor={i:j for (i,j),v in truck.items() if self.value(v)};r=[depot]
   while successor[r[-1]]!=sink:r.append(successor[r[-1]])
   r.append(depot);h=[];w={i:np.zeros(L) for i in r[:-1]}
   for (l,i,j),v in drone.items():
    if self.value(v):h.append({'drone':l,'dispatch_node':i,'delivery_node':j});w[i][l]+=T[l,i,j]
   save(r,h,w)
 solver.solve(model,CB())
