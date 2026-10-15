import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger
pa=argparse.ArgumentParser();pa.add_argument('--instance_path',required=True);pa.add_argument('--solution_path',required=True);pa.add_argument('--time_limit',type=int,required=True);pa.add_argument('--log_path');a=pa.parse_args();d=json.load(open(a.instance_path));S=d['S'];p=d['p'];sea=sorted(d['seaside_containers'],key=lambda z:z['id']);land=d['landside_containers'];H=max([z['deadline'] for z in land]+[2*sum(z['target_slot']+p for z in sea)+d['sigma_w']+S+10]);model=cp_model.CpModel();w=[model.new_int_var(0,S,f'w{t}') for t in range(H+1)];l=[model.new_int_var(1,S+1,f'l{t}') for t in range(H+1)];model.add(w[0]==d['sigma_w']);model.add(l[0]==d['sigma_l'])
for t in range(H+1):
 model.add(w[t]<l[t])
 if t:model.add(w[t]-w[t-1]<=1);model.add(w[t-1]-w[t]<=1);model.add(l[t]-l[t-1]<=1);model.add(l[t-1]-l[t]<=1)
def at(arr,start,pos,active=None):
 for dt in range(p+1):
  idx=model.new_int_var(0,H,f'idx{len(model.proto.variables)}');model.add(idx==start+dt);v=model.new_int_var(0,S+1,f'loc{len(model.proto.variables)}');model.add_element(idx,arr,v)
  if active is None:model.add(v==pos)
  else:model.add(v==pos).only_enforce_if(active)
seadata=[];landdata=[];linterval=[];prev_end=0;prev_slot=d['sigma_w'];completion=[]
for z in sea:
 cid=z['id'];target=z['target_slot'];stage=model.new_int_var(1,S,'stage'+str(cid));coop=model.new_bool_var('coop'+str(cid));pickup=model.new_int_var(0,H-p,'pick'+str(cid));drop=model.new_int_var(0,H-p,'drop'+str(cid));model.add(pickup>=prev_end+prev_slot);model.add(drop>=pickup+p+stage);at(w,pickup,0);at(w,drop,stage);prev_end=drop+p;prev_slot=stage
 if target>S:model.add(coop==1)
 else:model.add(stage==target).only_enforce_if(coop.Not())
 lift=model.new_int_var(0,H-p,'llift'+str(cid));ldrop=model.new_int_var(0,H-p,'ldrop'+str(cid));distance=model.new_int_var(0,S,'dist'+str(cid));model.add_abs_equality(distance,stage-target);model.add(lift>=drop+p+1).only_enforce_if(coop);model.add(ldrop>=lift+p+distance).only_enforce_if(coop);at(l,lift,stage,coop);at(l,ldrop,target,coop);length=model.new_int_var(1,H,'carry'+str(cid));end=model.new_int_var(0,H,'lend'+str(cid));model.add(end==ldrop+p);model.add(length==end-lift);linterval.append(model.new_optional_interval_var(lift,length,end,coop,'liv'+str(cid)));done=model.new_int_var(0,H,'done'+str(cid));model.add(done==end).only_enforce_if(coop);model.add(done==drop+p).only_enforce_if(coop.Not());completion.append(done);seadata.append((z,drop,stage,coop,lift,ldrop))
for z in land:
 lift=model.new_int_var(0,H-p,'blift'+str(z['id']));drop=model.new_int_var(0,H-p,'bdrop'+str(z['id']));model.add(drop>=lift+p+S+1-z['source_slot']);model.add(drop+p>=z['earliest_finish_time']);model.add(drop+p<=z['deadline']);at(l,lift,z['source_slot']);at(l,drop,S+1);length=model.new_int_var(1,H,'bcarry'+str(z['id']));end=model.new_int_var(0,H,'bend'+str(z['id']));model.add(end==drop+p);model.add(length==end-lift);linterval.append(model.new_interval_var(lift,length,end,'biv'+str(z['id'])));landdata.append((z,lift,drop))
model.add_no_overlap(linterval);C=model.new_int_var(0,H,'C');model.add_max_equality(C,completion);model.minimize(C);solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.random_seed=0;solver.parameters.max_time_in_seconds=max(.1,a.time_limit-2);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
class CB(cp_model.CpSolverSolutionCallback):
 def on_solution_callback(self):
  events=[]
  for z,drop,stage,coop,lift,ldrop in seadata:
   ss=self.value(stage);ev={'container_id':z['id'],'target_slot':z['target_slot'],'seaside_drop_time':self.value(drop),'seaside_drop_slot':ss}
   if self.value(coop):ev.update(landside_lift_time=self.value(lift),landside_lift_slot=ss,landside_drop_time=self.value(ldrop))
   events.append(ev)
  les=[{'container_id':z['id'],'source_slot':z['source_slot'],'lift_time':self.value(lift),'drop_time':self.value(drop)} for z,lift,drop in landdata];s={'objective_value':self.value(C),'crane_w_positions':{str(t):self.value(v) for t,v in enumerate(w)},'crane_l_positions':{str(t):self.value(v) for t,v in enumerate(l)},'seaside_events':events,'landside_events':les};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(s['objective_value'],s)
solver.solve(model,CB())
