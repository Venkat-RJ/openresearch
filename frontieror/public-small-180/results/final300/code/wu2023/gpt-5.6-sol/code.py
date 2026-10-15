import argparse,json,os,time,itertools
wall_started=time.time()
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']): del os.environ['GRB_LICENSE_FILE']
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger
p=argparse.ArgumentParser()
for k in ['instance_path','solution_path','log_path']: p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True)
a=p.parse_args(); end=time.monotonic()+a.time_limit-1; d=json.load(open(a.instance_path)); c=d['scenarios']; N=len(c); n=len(c[0]); Ks=d['K_values']; logger=configured_logger(a.log_path,sense='maximize'); ratio=[[min(c[l][j]/c[i][j] if c[i][j]>0 else 1. if c[l][j]==0 else float('inf') for j in range(n)) for l in range(N)] for i in range(N)]; out={'one_stage_IP_mu':{},'two_stage_IP':{}}
def scalar(): return max([v['t'] for v in out['one_stage_IP_mu'].values()]+[v['t'] for v in out['two_stage_IP'].values()])
def save():
 out['objective_value']=scalar(); json.dump(out,open(a.solution_path,'w')); logger.log_solution(out['objective_value'],out)
for K in Ks:
 bs=list(range(K)); best=min(max(ratio[i][l] for l in bs) for i in range(N))
 for j,s in enumerate(itertools.combinations(range(N),K)):
  if j%32==0 and time.monotonic()>end: break
  z=min(max(ratio[i][l] for l in s) for i in range(N))
  if z>best: best=z; bs=list(s)
 out['two_stage_IP'][str(K)]={'t':best,'approximation_ratio':1/best,'selected_scenarios':bs}
 assignment={str(i):max(range(K),key=lambda k:ratio[i][bs[k]]) for i in range(N)}
 out['one_stage_IP_mu'][str(K)]={'t':best,'approximation_ratio':1/best,'assignment':assignment,'lambda_weights':{str(k):{str(bs[k]):1.} for k in range(K)}}
save()
for ki,K in enumerate(Ks):
 if K==N or time.monotonic()>=end: continue
 m=gp.Model(); m.Params.OutputFlag=0; m.Params.Threads=1; m.Params.Seed=0; m.Params.MIPGap=1e-4; m.Params.NumericFocus=0
 t=m.addVar(ub=1,obj=-1); lam=m.addVars(K,N,lb=0,ub=1); mu=m.addVars(N,K,vtype=GRB.BINARY)
 for k in range(K): m.addConstr(lam.sum(k,'*')==1)
 for i in range(N):
  m.addConstr(mu.sum(i,'*')==1)
  for k in range(K):
   for j in range(n): m.addConstr(t*c[i][j]<=gp.quicksum(lam[k,l]*c[l][j] for l in range(N))+max(max(v) for v in c)*(1-mu[i,k]))
 m.update(); assert m.NumVars<=2000 and m.NumConstrs<=2000
 prev=out['one_stage_IP_mu'][str(K)]; t.Start=prev['t']
 for i in range(N):
  for k in range(K): mu[i,k].Start=int(prev['assignment'][str(i)]==k)
 for k in range(K):
  for l in range(N): lam[k,l].Start=prev['lambda_weights'][str(k)].get(str(l),0)
 m.Params.TimeLimit=max(.1,(end-time.monotonic())/max(1,len(Ks)-ki)); m.optimize()
 if m.SolCount:
  weights={str(k):{str(l):max(0,lam[k,l].X) for l in range(N)} for k in range(K)}
  for k in range(K):
   total=sum(weights[str(k)].values()); weights[str(k)]={l:v/total for l,v in weights[str(k)].items()}
  assignment={str(i):max(range(K),key=lambda k:mu[i,k].X) for i in range(N)}
  actual=min(sum(weights[str(assignment[str(i)])][str(l)]*c[l][j] for l in range(N))/c[i][j] for i in range(N) for j in range(n) if c[i][j]>0)
  if actual>prev['t']+1e-8: out['one_stage_IP_mu'][str(K)]={'t':actual,'approximation_ratio':1/actual,'assignment':assignment,'lambda_weights':weights}; save()
