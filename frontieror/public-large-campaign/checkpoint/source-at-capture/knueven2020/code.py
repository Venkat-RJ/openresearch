import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,math
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));T=d['num_time_periods'];gens=d['generators'];c=[];lb=[];ub=[];integ=[];rr=[];cc=[];vv=[];rl=[];ru=[]
def var(cost=0,upper=np.inf,binary=False):
 i=len(c);c.append(cost);lb.append(0);ub.append(upper);integ.append(int(binary));return i
def row(terms,l=-np.inf,u=np.inf):
 i=len(rl);rl.append(l);ru.append(u)
 for j,v in terms.items():rr.append(i);cc.append(j);vv.append(v)
U=[];P=[];V=[];W=[]
for g in gens:
 u=[var(g['no_load_cost'],1,True) for _ in range(T)];pw=[var() for _ in range(T)];v=[var(g['startup_costs'][-1]['cost'],1,True) for _ in range(T)];w=[var(0,1,True) for _ in range(T)];U.append(u);P.append(pw);V.append(v);W.append(w);init=g['initial_status'];u0=int(init>0);p0=g['initial_power'] if u0 else 0;UT=g['min_up_time'];DT=g['min_down_time'];PM=g['p_max'];RU=g['ramp_up_limit'];RD=g['ramp_down_limit'];SU=g['startup_ramp_limit'];SD=g['shutdown_ramp_limit']
 for t in range(T):
  terms={u[t]:1,v[t]:-1,w[t]:1}
  if t:terms[u[t-1]]=-1;row(terms,0,0)
  else:row(terms,u0,u0)
  row({v[t]:1,w[t]:1},u=1);row({pw[t]:1,u[t]:-g['p_min']},l=0);row({pw[t]:1,u[t]:-PM},u=0)
  terms={pw[t]:1,v[t]:-SU}
  if t:terms[pw[t-1]]=-1;terms[u[t-1]]=-RU;row(terms,u=0)
  else:row(terms,u=p0+RU*u0)
  terms={pw[t]:-1,u[t]:-RD,w[t]:-SD}
  if t:terms[pw[t-1]]=1;row(terms,u=0)
  else:row(terms,u=-p0)
  row({**{v[s]:1 for s in range(max(0,t-UT+1),t+1)},u[t]:-1},u=0);row({**{w[s]:1 for s in range(max(0,t-DT+1),t+1)},u[t]:1},u=1)
  if init>0 and t<max(0,UT-init):lb[u[t]]=1
  if init<0 and t<max(0,DT+init):ub[u[t]]=0
  segvars=[]
  for seg in g['production_cost_segments']:
   z=var(seg['marginal_cost_per_mwh'],seg['output_mw_end']-seg['output_mw_start']);segvars.append(z)
  row({**{z:1 for z in segvars},pw[t]:-1,u[t]:g['p_min']},0,0)
  TRU=max(0,math.ceil((PM-SU)/RU)) if RU else 0;TRD=max(0,math.ceil((PM-SD)/RD)) if RD else 0
  if UT>=TRU+TRD+2:
   terms={pw[t]:1,u[t]:-PM}
   for i in range(TRU+1):
    if t-i>=0:terms[v[t-i]]=max(0,PM-SU-i*RU)
   for i in range(TRD+1):
    if t+1+i<T:terms[w[t+1+i]]=max(0,PM-SD-i*RD)
   row(terms,u=0)
for t in range(T):row({pw[t]:1 for pw in P},d['demand'][t],d['demand'][t]);row({u[t]:g['p_max'] for u,g in zip(U,gens)},l=d['demand'][t]+d['reserve_requirement'][t])
A=coo_matrix((vv,(rr,cc)),shape=(len(rl),len(c))).tocsc()
if len(gens)>200:
 import highspy
 from types import SimpleNamespace
 events=[];active=False
 for t in range(1,T):
  change=d['demand'][t]-d['demand'][t-1]
  if not active and change>.1*d['demand'][t-1]:events.append((t-1,1));active=True
  elif active and change<-.16*d['demand'][t-1]:events.append((t-1,0));active=False
 for k,g in enumerate(gens):
  prev=int(g['initial_status']>0);age=abs(g['initial_status'])
  for t in range(T):
   wanted=int(g['initial_status']>0)
   if not wanted and g['type'] in ('gas_cc','gas_ct'):
    for first,status in events:
     if t>=first+(k%4 if status else k%3):wanted=status
   on=prev if wanted!=prev and age<(g['min_up_time'] if prev else g['min_down_time']) else wanted
   for j,value in ((U[k][t],on),(V[k][t],int(on and not prev)),(W[k][t],int(prev and not on))):lb[j]=ub[j]=value
   age=age+1 if on==prev else 1;prev=on
 engine=highspy.Highs();engine.setOptionValue('threads',1);engine.setOptionValue('output_flag',False)
 model=highspy.HighsLp();model.num_col_=len(c);model.num_row_=len(rl);model.col_cost_=c;model.col_lower_=lb;model.col_upper_=ub;model.row_lower_=rl;model.row_upper_=ru
 model.a_matrix_.format_=highspy.MatrixFormat.kColwise;model.a_matrix_.start_=A.indptr;model.a_matrix_.index_=A.indices;model.a_matrix_.value_=A.data
 engine.passModel(model);engine.setOptionValue('time_limit',max(.1,a.time_limit-(time.monotonic()-start)-10));engine.run()
 r=SimpleNamespace(x=engine.getSolution().col_value if engine.getInfo().primal_solution_status==highspy.kSolutionStatusFeasible else None)
else:
 r=milp(c,integrality=integ,bounds=Bounds(lb,ub),constraints=LinearConstraint(A,rl,ru),options={'time_limit':max(.1,a.time_limit-(time.monotonic()-start)-5),'mip_rel_gap':1e-4,'threads':1})
if r.x is not None:
 schedule={};obj=0
 for k,g in enumerate(gens):
  ua=[int(round(r.x[i])) for i in U[k]];pa=[max(0,float(r.x[i])) for i in P[k]];schedule[g['id']]={'u':ua,'p':pa,'pb':[g['p_max']*v for v in ua]};prev=int(g['initial_status']>0);shutdowns=[]
  for t in range(T):
   obj+=g['no_load_cost']*ua[t];remaining=pa[t]-g['p_min']*ua[t]
   for seg in g['production_cost_segments']:take=min(max(remaining,0),seg['output_mw_end']-seg['output_mw_start']);obj+=take*seg['marginal_cost_per_mwh'];remaining-=take
   if prev and not ua[t]:shutdowns.append(t)
   if ua[t] and not prev:
    costs=[g['startup_costs'][-1]['cost']];low=1
    for tier in g['startup_costs'][:-1]:
     high=tier['max_downtime_hours'];eligible=any(low<=t-s<=high for s in shutdowns) or (g['initial_status']<0 and low<=t+abs(g['initial_status'])<=high)
     if eligible:costs.append(tier['cost'])
     low=high+1
    obj+=min(costs)
   prev=ua[t]
 s={'objective_value':float(obj),'schedule':schedule};json.dump(s,open(a.solution_path,'w'))
 if a.log_path:SolutionLogger(a.log_path,sense='minimize').log_solution(obj,s)
