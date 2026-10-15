import argparse,json,time
from ortools.sat.python import cp_model

def solve(d,t):
 vs=d['vertices'];E={tuple(sorted((e['i'],e['j']))):e['weight'] for e in d['edges']};m=cp_model.CpModel();x={v:m.new_bool_var(f'v{v}') for v in vs};terms=[]
 for i,v in enumerate(vs):
  for u in vs[:i]:
   e=tuple(sorted((u,v)))
   if e not in E:m.add(x[u]+x[v]<=1)
   else:
    z=m.new_bool_var(f'e{u}_{v}');m.add(z<=x[u]);m.add(z<=x[v]);m.add(z>=x[u]+x[v]-1);terms.append(E[e]*z)
 m.maximize(sum(terms));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,t-1);r=s.solve(m)
 if r not in (cp_model.OPTIMAL,cp_model.FEASIBLE):raise RuntimeError('No weighted clique incumbent')
 clique=[v for v in vs if s.value(x[v])];chosen=set(clique);obj=sum(w for (u,v),w in E.items() if u in chosen and v in chosen);return {'objective_value':obj,'clique':clique}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')
