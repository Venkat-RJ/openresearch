import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from collections import defaultdict,Counter
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from solution_logger import SolutionLogger
p0=argparse.ArgumentParser();p0.add_argument('--instance_path',required=True);p0.add_argument('--solution_path',required=True);p0.add_argument('--time_limit',type=int,required=True);p0.add_argument('--log_path');a=p0.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));T=d['time_parameters']['time_horizon_T'];Z=[n['id'] for n in d['network']['nodes'] if n['type']=='centroid'];links={(l['from'],l['to']):l for l in d['network']['links']};road=[e for e in links if links[e]['type']=='road'];out=[e for e in links if e[0] in Z];inc=[e for e in links if e[1] in Z];outs=defaultdict(list);ins=defaultdict(list)
for i,j in links:outs[i].append(j);ins[j].append(i)
names=[];families=[];c=[];bounds=[];lookup={}
def var(f,key,cost=0,upper=None):
 ix=len(c);lookup[f,key]=ix;names.append('|'.join(map(str,key)));families.append(f);c.append(cost);bounds.append((0,upper));return ix
for e in links:
 for s in Z:
  for t in range(T+1):var('N_U',(*e,s,t),1,0 if not t else None);var('N_D',(*e,s,t),-1,0 if not t else None)
for i,j in road+out:
 for k in outs[j]:
  for s in Z:
   for t in range(T+1):var('y',(i,j,k,s,t),0,0 if (k in Z and s!=k) or t<links[i,j]['free_flow_travel_time_steps']-1 else None)
for i,j in out:
 for s in Z:
  for t in range(T+1):var('y_centroid',(i,j,s,t))
for z in Z:
 for t in range(T+1):var('p',(z,t))
for r in Z:
 for s in Z:
  for t in range(T+1):var('e',(r,s,t));var('omega',(r,s,t),1,0 if t in (0,T) else None)
E=[];er=[];U=[];ur=[]
def term(f,key,value=1):
 ix=lookup.get((f,key));return {} if ix is None else {ix:value}
def combine(*parts):
 result=defaultdict(float)
 for p in parts:
  for k,v in p.items():result[k]+=v
 return dict(result)
def row(parts,rhs=0,eq=True):
 (E if eq else U).append(combine(*parts));(er if eq else ur).append(rhs)
def turns(i,j,s,t):return combine(*[term('y',(i,j,k,s,t)) for k in outs[j]])
def arrivals(j,k,s,t):return combine(*[term('y',(i,j,k,s,t)) for i in ins[j]])
for i,j in road:
 link=links[i,j];cap=link['capacity_vph']*d['time_parameters']['time_step_duration_seconds']/3600;lv=link['free_flow_travel_time_steps'];lw=link['congested_travel_time_steps']
 for s in Z:
  for t in range(T):row([term('N_U',(i,j,s,t+1)),term('N_U',(i,j,s,t),-1),{k:-v for k,v in turns(i,j,s,t).items()}]);row([term('N_D',(i,j,s,t+1)),term('N_D',(i,j,s,t),-1),{k:-v for k,v in arrivals(i,j,s,t).items()}])
  for t in range(lv-1,T+1):row([turns(i,j,s,t),term('N_U',(i,j,s,t-lv+1),-1),term('N_D',(i,j,s,t))],eq=False)
 for t in range(T+1):row([turns(i,j,s,t) for s in Z],cap,False);row([arrivals(i,j,s,t) for s in Z],cap,False)
 for t in range(lw-1,T+1):row([arrivals(i,j,s,t) for s in Z]+[term('N_U',(i,j,s,t-lw+1),-1) for s in Z]+[term('N_D',(i,j,s,t)) for s in Z],link['jam_density_vehicles'],False)
for i,j in out:
 for s in Z:
  for t in range(T):row([term('N_U',(i,j,s,t+1)),term('N_U',(i,j,s,t),-1),term('y_centroid',(i,j,s,t),-1)])
for i,j in inc:
 for s in Z:
  for t in range(T):row([term('N_D',(i,j,s,t+1)),term('N_U',(i,j,s,t),-1)])
for z in Z:
 row([term('p',(z,0))],d['fleet']['initial_distribution'].get(z,0))
 for t in range(T):row([term('p',(z,t+1)),term('p',(z,t),-1)]+[term('N_U',(i,z,z,t),-1) for i in ins[z]]+[term('N_D',(i,z,z,t)) for i in ins[z]]+[term('y_centroid',(z,k,s,t)) for k in outs[z] for s in Z])
 for t in range(T+1):row([term('y_centroid',(z,k,s,t)) for k in outs[z] for s in Z]+[term('p',(z,t),-1)],eq=False)
row([term('p',(z,T)) for z in Z],sum(d['fleet']['initial_distribution'].values()))
demand=Counter((od['origin'],od['destination'],t) for od in d['demand']['od_pairs'] for t in od['departure_times'])
for r in Z:
 for s in Z:
  for t in range(T+1):row([term('e',(r,s,t)),term('omega',(r,s,t),-1)],eq=False);row([term('e',(r,s,t))]+[term('y_centroid',(r,j,s,t),-1) for j in outs[r]],eq=False)
  for t in range(T):row([term('omega',(r,s,t+1)),term('omega',(r,s,t),-1),term('e',(r,s,t))],demand[r,s,t])
for e in links:
 for s in Z:
  for t in range(T+1):row([term('N_D',(*e,s,t)),term('N_U',(*e,s,t),-1)],eq=False)
def matrix(rows):
 rr=[];cc=[];vv=[]
 for i,row in enumerate(rows):
  for j,v in row.items():rr.append(i);cc.append(j);vv.append(v)
 return coo_matrix((vv,(rr,cc)),shape=(len(rows),len(c))).tocsc()
r=linprog(c,A_ub=matrix(U),b_ub=ur,A_eq=matrix(E),b_eq=er,bounds=bounds,method='highs',options={'threads':1,'time_limit':max(.1,a.time_limit-(time.monotonic()-start)-2)})
if r.x is not None:
 result={f:{} for f in ('y','y_centroid','N_U','N_D','p','e','omega')}
 for f,key,v in zip(families,names,r.x):
  if abs(v)>1e-9:result[f][key]=float(v)
 result['objective_value']=float(np.dot(c,r.x));json.dump(result,open(a.solution_path,'w'))
 if a.log_path:SolutionLogger(a.log_path,sense='minimize').log_solution(result['objective_value'],result)
