import time
_wall_start=time.time()
import argparse,json,random,itertools
from solution_logger import SolutionLogger
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();deadline=start+a.time_limit-.3;d=json.load(open(a.instance_path));log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if log:log.min_interval=0;log.start_time=_wall_start
patients={q['patient_id']:q for q in d['patients']};surgeons={q['surgeon_id']:q for q in d['surgeons']};hospitals={q['hospital_id']:q for q in d['hospitals']};rooms={(h['hospital_id'],r['or_id']):r for h in d['hospitals'] for r in h['ORs']};options={};alpha=d['alpha'];days=d['num_days'];rng=random.Random(0)
for p,pat in patients.items():
 opts=[]
 for s in pat['eligible_surgeons']:
  for hstr,rs in pat['eligible_ORs_by_hospital'].items():
   h=int(hstr)
   for day in surgeons[s]['operating_days']:
    if day>pat['due_date'] or day>=days:continue
    for r in rs:opts.append((s,h,day,r))
 options[p]=opts
best=float('inf');solution=None

def extend(state,p,opt):
 value,ass,rr,ss=state;pat=patients[p];s,h,day,r=opt;roomkey=(h,day,r);sk=(s,day);other=ss.get(sk)
 if other and other['hospital']!=h:return
 ri=rooms[h,r];regular=ri['daily'][str(day)]['regular_time'];maximum=ri['daily'][str(day)]['max_overtime'];availability=surgeons[s]['availability_by_day'].get(str(day),0);prep=pat['preparation_time'];clean=pat['cleaning_time'];E=pat['surgeon_specific_times'][str(s)];T=prep+E+clean
 oldroom=rr.get(roomkey,{'finish':0,'overtime':0});oldsurgeon=other or {'hospital':h,'start':0,'finish':0,'weighted':0};surgery=max(oldroom['finish']+prep,oldsurgeon['finish']);finish=surgery+E;exit=finish+clean;first=oldsurgeon['start'] if other else surgery;weighted=oldsurgeon['weighted']+alpha*E+(1-alpha)*T
 if exit>regular+maximum+1e-8 or finish-first>availability+1e-8 or weighted>availability+1e-8:return
 overtime=max(0,exit-regular);delta=(0 if roomkey in rr else ri['fixed_cost_per_hour']*regular/60)+(0 if other else hospitals[h]['surgeon_fixed_costs_per_day'].get(str(s),0))+(overtime-oldroom['overtime'])*ri['overtime_cost_per_hour']/60-(0 if pat['is_mandatory'] else max(0,pat['reward']))
 newrooms=dict(rr);newsurgeons=dict(ss);newrooms[roomkey]={'finish':exit,'overtime':overtime};newsurgeons[sk]={'hospital':h,'start':first,'finish':finish,'weighted':weighted};entry={'patient_id':p,'surgeon_id':s,'hospital_id':h,'day':day,'or_id':r,'is_mandatory':pat['is_mandatory'],'T_ps':T,'finish_time':finish,'surgery_start_time':surgery,'room_entry_time':surgery-prep,'room_exit_time':exit}
 return (value+delta,ass+[entry],newrooms,newsurgeons)

def publish(state):
 global best,solution
 value,ass,rr,ss=state
 if value>=best:return
 ors=[{'hospital_id':h,'day':day,'or_id':r,'regular_time':rooms[h,r]['daily'][str(day)]['regular_time'],'completion_time':q['finish'],'overtime':q['overtime']} for (h,day,r),q in rr.items()];surg=[{'surgeon_id':s,'hospital_id':q['hospital'],'day':day,'start_time':q['start'],'end_time':q['finish']} for (s,day),q in ss.items()];orseq=[];sseq=[]
 for u,v in itertools.combinations(ass,2):
  if u['hospital_id']!=v['hospital_id'] or u['day']!=v['day']:continue
  q={'hospital_id':u['hospital_id'],'day':u['day'],'patient_p':u['patient_id'],'patient_k':v['patient_id'],'p_after_k':int(u['surgery_start_time']>v['surgery_start_time'])}
  if u['or_id']==v['or_id']:orseq.append(dict(q,or_id=u['or_id']))
  if u['surgeon_id']==v['surgeon_id']:sseq.append(dict(q,surgeon_id=u['surgeon_id']))
 best=value;solution={'objective_value':value,'assignments':ass,'opened_ors':ors,'surgeon_assignments':surg,'or_sequences':orseq,'surgeon_sequences':sseq}
 if log:log.log_solution(value,solution)
iteration=0
while time.monotonic()<deadline:
 mandatory=[p for p,pat in patients.items() if pat['due_date']<=days];optional=[p for p,pat in patients.items() if pat['due_date']>days]
 if iteration==0:mandatory.sort(key=lambda p:(patients[p]['due_date'],len(options[p])))
 else:rng.shuffle(mandatory)
 rng.shuffle(optional);beam=[(0,[],{}, {})]
 for p in mandatory+optional:
  candidates=[]
  for state in beam:
   if p in optional:candidates.append(state)
   for opt in options[p]:
    q=extend(state,p,opt)
    if q:candidates.append(q)
   if time.monotonic()>deadline:break
  if not candidates:beam=[];break
  candidates.sort(key=lambda q:q[0]+(rng.random()*50 if iteration else 0));beam=candidates[:64]
  if time.monotonic()>deadline:break
 for state in beam:
  if set(mandatory).issubset(q['patient_id'] for q in state[1]):publish(state)
 iteration+=1
if solution is None:raise RuntimeError('No feasible surgery assignment found')
solution['runtime']=time.monotonic()-start;json.dump(solution,open(a.solution_path,'w'))
