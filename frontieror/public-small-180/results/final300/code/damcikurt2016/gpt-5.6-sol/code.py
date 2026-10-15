import argparse,json,time,os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();deadline=time.monotonic()+max(.1,a.time_limit-2);d=json.load(open(a.instance_path));g=d['generator'];tree=d['scenario_tree'];nodes={v['id']:v for v in tree['nodes']};L=tree['L_min_up_time'];ell=tree['ell_min_down_time'];lo=int(g['C_lower_MW']);hi=int(g['C_upper_MW']);up=int(g['V_plus_MW_per_h']);down=int(g['V_minus_MW_per_h']);q=np.arange(lo,hi+1);N=len(q);on={};off={};startup=g['U_bar_startup_cost_dollars'];shutdown=g['U_lower_shutdown_cost_dollars'];logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 for i in sorted(nodes,key=lambda i:nodes[i]['time'],reverse=True):
  if time.monotonic()>deadline:raise RuntimeError('Scenario DP deadline before complete policy')
  v=nodes[i];p=v['probability'];immediate=p*((v['electricity_price_dollars_per_MWh']-g['fuel_b_dollars_per_MWh'])*q-g['fuel_a_dollars_per_h']);aon=np.tile(immediate,(L,1));aoff=np.zeros(ell)
  for c in v['children_ids']:
   child=nodes[c];ranges=np.array([[np.max(on[c][age,max(0,z-down):min(N,z+up+1)]) for z in range(N)] for age in range(L)])
   for age in range(L):
    choices=ranges[min(L-1,age+1)]
    if age==L-1:choices=np.maximum(choices,off[c][0]-child['probability']*shutdown)
    aon[age]+=choices
   for age in range(ell):
    val=off[c][min(ell-1,age+1)]
    if age==ell-1:val=max(val,float(np.max(on[c][0]))-child['probability']*startup)
    aoff[age]+=val
  on[i]=aon;off[i]=aoff
 root=next(i for i in nodes if nodes[i]['parent_id'] is None);possible=list(range(N));initial=tree.get('initial_generator_status_y0',0);x0=tree.get('initial_generation_x0_MW',0)
 if initial:possible=[z for z in possible if q[z]-x0<=up+1e-8 and x0-q[z]<=down+1e-8]
 candidates=[(off[root][-1],0,ell-1,0)] if x0<=hi+1e-8 else []
 candidates.extend((on[root][-1,z],1,L-1,z) for z in possible);_,status,age,z=max(candidates);policy={};stack=[(root,status,age,z)]
 while stack:
  i,status,age,z=stack.pop();policy[i]=(int(status),int(q[z]) if status else 0);v=nodes[i]
  for c in v['children_ids']:
   cp=nodes[c]['probability'];opts=[]
   if status:
    na=min(L-1,age+1)
    for zz in range(max(0,z-down),min(N,z+up+1)):opts.append((on[c][na,zz],1,na,zz))
    if age==L-1:opts.append((off[c][0]-cp*shutdown,0,0,0))
   else:
    opts.append((off[c][min(ell-1,age+1)],0,min(ell-1,age+1),0))
    if age==ell-1:
     zz=int(np.argmax(on[c][0]));opts.append((on[c][0,zz]-cp*startup,1,0,zz))
   _,cs,ca,cz=max(opts);stack.append((c,cs,ca,cz))
 objective=0;x={};y={};u={}
 for i,(s,prod) in policy.items():
  v=nodes[i];p=v['probability'];objective+=p*((v['electricity_price_dollars_per_MWh']-g['fuel_b_dollars_per_MWh'])*prod-g['fuel_a_dollars_per_h']*s)
  if prod:x[str(i)]=prod
  if s:y[str(i)]=1
  parent=v['parent_id']
  if parent is not None:
   ps=policy[parent][0]
   if s and not ps:u[str(i)]=1;objective-=p*startup
   elif ps and not s:objective-=p*shutdown
 solution={'objective_value':objective,'x':x,'y':y,'u':u};json.dump(solution,open(a.solution_path,'w'))
 if logger:logger.log_solution(objective,solution)
if __name__=='__main__':main()
