import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time,math
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path',required=True);a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path);W=d['parameters']['W'];items=d['items'];packed=[];loads=[]
 for it in sorted(items,key=lambda v:v['weight'],reverse=True):
  for _ in range(it['demand']):
   choices=[b for b in range(len(packed)) if loads[b]+it['weight']<=W]
   if choices:b=min(choices,key=lambda b:W-loads[b]-it['weight'])
   else:b=len(packed);packed.append([]);loads.append(0)
   packed[b].append(it['item_type']);loads[b]+=it['weight']
 best=len(packed)
 def save(bins):
  nonlocal best
  if len(bins)>best:return
  best=len(bins);sol={'objective_value':best,'num_bins':best,'bins':bins};json.dump(sol,open(a.solution_path,'w'));logger.log_solution(best,sol)
 save(packed);bound=math.ceil(sum(it['weight']*it['demand'] for it in items)/W)
 if best<=bound:return
 B=best;m=cp_model.CpModel();y=[m.new_bool_var('y%d'%b) for b in range(B)];x=[[m.new_int_var(0,min(it['demand'],W//it['weight']),'x%d_%d'%(i,b)) for b in range(B)] for i,it in enumerate(items)];loadvars=[]
 for i,it in enumerate(items):m.add(sum(x[i])==it['demand'])
 for b in range(B):
  load=m.new_int_var(0,W,'load%d'%b);m.add(load==sum(it['weight']*x[i][b] for i,it in enumerate(items)));m.add(load<=W*y[b]);loadvars.append(load)
  if b:m.add(y[b]<=y[b-1]);m.add(loadvars[b]<=loadvars[b-1])
 m.add(sum(y)>=bound);m.minimize(sum(y));order=sorted(range(B),key=lambda b:loads[b],reverse=True)
 for b,old in enumerate(order):
  m.add_hint(y[b],1)
  for i,it in enumerate(items):m.add_hint(x[i][b],packed[old].count(it['item_type']))
 class Callback(cp_model.CpSolverSolutionCallback):
  def on_solution_callback(self):
   bins=[]
   for b in range(B):
    binitems=[it['item_type'] for i,it in enumerate(items) for _ in range(self.value(x[i][b]))]
    if binitems:bins.append(binitems)
   save(bins)
 s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,a.time_limit*.95-(time.monotonic()-start));s.solve(m,Callback())
if __name__=='__main__':main()
