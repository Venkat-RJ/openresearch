import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from ortools.sat.python import cp_model

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));containers={v['id']:v for v in d['containers']};trades=d['trades'];m=cp_model.CpModel();x={};periodrows={};changes=[]
 for trade in trades:
  j=trade['id'];D=trade['container_demand'];prev={}
  for t in range(trade['start_period'],trade['end_period']+1):
   ids=[i for i,c in containers.items()if c['availability_start']<=t<=c['availability_end']];row={i:m.NewBoolVar('x%d_%d_%d'%(j,t,i))for i in ids};x[j,t]=row;m.Add(sum(row.values())==D);root={i:m.NewBoolVar('r%d_%d_%d'%(j,t,i))for i in ids};q={i:m.NewIntVar(0,D,'q%d_%d_%d'%(j,t,i))for i in ids};flows={}
   for i in ids:
    m.Add(root[i]<=row[i]);m.Add(q[i]<=D*root[i]);periodrows.setdefault((t,i),[]).append(row[i])
    for k in containers[i]['adjacent_containers']:
     if k in row:
      f=m.NewIntVar(0,D,'f%d_%d_%d_%d'%(j,t,i,k));flows[i,k]=f;m.Add(f<=D*row[i]);m.Add(f<=D*row[k])
   m.Add(sum(root.values())<=trade['max_dispersion']);m.Add(sum(q.values())==D)
   for i in ids:m.Add(q[i]+sum(f for(k,l),f in flows.items()if l==i)-sum(f for(k,l),f in flows.items()if k==i)==row[i])
   if prev:
    change=m.NewBoolVar('change%d_%d'%(j,t));changes.append(change)
    for i,v in row.items():m.Add(change>=v-prev.get(i,0))
   prev=row
 for row in periodrows.values():m.Add(sum(row)<=1)
 m.Minimize(sum(changes));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,float(a.time_limit)-2-(time.monotonic()-start));status=s.Solve(m)
 if status not in(cp_model.FEASIBLE,cp_model.OPTIMAL):raise RuntimeError('No feasible connected container assignments found')
 assignments={};value=0
 for trade in trades:
  j=trade['id'];rr={};prev=None
  for t in range(trade['start_period'],trade['end_period']+1):
   ids=[i for i,v in x[j,t].items()if s.Value(v)];rr[str(t)]=ids
   if prev is not None and set(ids)!=prev:value+=1
   prev=set(ids)
  assignments[str(j)]=rr
 json.dump({'objective_value':value,'assignments':assignments},open(a.solution_path,'w'))
if __name__=='__main__':main()
