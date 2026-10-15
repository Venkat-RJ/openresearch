import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time
from ortools.sat.python import cp_model

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));products=d['products'];P=d['num_patterns'];M=d['M'];m=cp_model.CpModel();y=[];r=[];pieces=[];quant=[];knife=d['problem_type']=='cscp_with_knife_dependent_setups';closed=d['problem_type']=='closed_ended_cscp';obj=[]
 for j in range(P):
  yy=m.NewBoolVar('y'+str(j));rr=m.NewIntVar(0,M,'r'+str(j));y.append(yy);r.append(rr);m.Add(rr<=M*yy);m.Add(rr>=yy);row=[];qs=[]
  if j:m.Add(r[j-1]>=rr)
  for i,v in enumerate(products):
   aa=m.NewIntVar(0,v['N_i'],'a%d_%d'%(i,j));qq=m.NewIntVar(0,M*v['N_i'],'q%d_%d'%(i,j));m.Add(aa<=v['N_i']*yy);m.AddMultiplicationEquality(qq,[aa,rr]);row.append(aa);qs.append(qq)
  m.Add(sum(v['length']*aa for v,aa in zip(products,row))<=d['machine_length']);pieces.append(row);quant.append(qs);obj.append(1000*d['cost_setup_pattern']*yy+1000*d['cost_repetition']*rr)
  if knife:obj.append(int(round(1000*d['cost_knife_setup']))*sum(row))
 extras=[]
 for i,v in enumerate(products):
  e=m.NewIntVar(0,0 if closed else v['demand'],'extra'+str(i));extras.append(e);m.Add(sum(quant[j][i]for j in range(P))+e>=v['demand'])
 m.Add(sum(extras)<=sum(r));m.Minimize(sum(obj));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,float(a.time_limit)-2-(time.monotonic()-start));status=s.Solve(m)
 if status not in(cp_model.FEASIBLE,cp_model.OPTIMAL):raise RuntimeError('No feasible integrated cutting pattern found')
 patterns={};pq={};value=0
 for j in range(P):
  used=s.Value(y[j]);rep=s.Value(r[j]);counts=[s.Value(v)for v in pieces[j]];patterns[str(j)]={'used':bool(used),'repetitions':rep,'pieces':{str(v['id']):counts[i]for i,v in enumerate(products)}}
  if knife:patterns[str(j)]['knife_count']=sum(counts)
  value+=d['cost_setup_pattern']*used+d['cost_repetition']*rep+(d['cost_knife_setup']*sum(counts)if knife else 0)
  for i,v in enumerate(products):pq['product_%s_pattern_%d'%(v['id'],j)]=counts[i]*rep
 json.dump({'objective_value':value,'patterns':patterns,'extra_pieces':{str(v['id']):s.Value(extras[i])for i,v in enumerate(products)},'production_quantities':pq},open(a.solution_path,'w'))
if __name__=='__main__':main()
