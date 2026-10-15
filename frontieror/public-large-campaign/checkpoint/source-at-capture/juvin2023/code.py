import argparse,json,time
from ortools.sat.python import cp_model

def solve(d,t):
 jobs=d['jobs'];H=sum(max(o['processing_times'].values()) for j in jobs for o in j['operations']);m=cp_model.CpModel();machine=[[] for _ in range(d['num_machines'])];info={};ends=[];final_intervals=[]
 for j in jobs:
  previous=None
  for o in j['operations']:
   key=f'({j["job_id"]},{o["operation_id"]})';start=m.new_int_var(0,H,'s'+key);end=m.new_int_var(0,H,'e'+key);choices=[]
   for k in o['eligible_machines']:
    dur=int(o['processing_times'][str(k)]);a=m.new_bool_var(f'a{key}_{k}');interval=m.new_optional_interval_var(start,dur,end,a,f'i{key}_{k}');machine[k].append(interval);choices.append((k,a))
    if o is j['operations'][-1]:final_intervals.append(interval)
   m.add_exactly_one(a for k,a in choices)
   if previous is not None:m.add(start>=previous)
   previous=end;info[key]=(start,end,choices)
  ends.append(previous)
 for intervals in machine:m.add_no_overlap(intervals)
 m.add_no_overlap(final_intervals)
 makespan=m.new_int_var(0,H,'makespan');m.add_max_equality(makespan,ends);m.minimize(makespan);s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,t-1);r=s.solve(m)
 if r not in (cp_model.OPTIMAL,cp_model.FEASIBLE):raise RuntimeError('No flexible job-shop incumbent')
 assignment={};schedule={}
 for key,(start,end,choices) in info.items():assignment[key]=next(k for k,a in choices if s.value(a));schedule[key]=list(range(s.value(start)+1,s.value(end)+1))
 return {'objective_value':s.value(makespan)+1,'assignment':assignment,'schedule':schedule}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')
