import argparse,json,time,math
from ortools.sat.python import cp_model

def solve(d,t):
 ships=d['ships'];n=len(ships);berths=d['berths'];H=int(math.ceil(10*d['C_max']));Du=int(round(10*d['lockage_duration_Du']));MT=int(round(10*d['min_time_between_lockages_MT']));m=cp_model.CpModel();mode=[m.new_bool_var(f'u{i}') for i in range(n)];used=[m.new_bool_var(f'used{q}') for q in range(n)];completion=[m.new_int_var(0,H,f'C{q}') for q in range(n)];batch=[[m.new_bool_var(f'b{i}_{q}') for q in range(n)] for i in range(n)];shipend=[m.new_int_var(0,H,f'e{i}') for i in range(n)];qindex=[m.new_int_var(0,n-1,f'q{i}') for i in range(n)];W=int(math.floor(d['chamber_width_W']));L=int(math.floor(d['chamber_length_L']));arrival=[int(round(v['arrival_time']*10)) for v in ships];intervals={b['id']:[] for b in berths};berthchoice={};berthstart={};berthend={};terms=[]
 for q in range(n):
  m.add(completion[q]>=int(round(d['lock_start_availability_sc']*10))+Du).only_enforce_if(used[q]);m.add(sum(batch[i][q] for i in range(n))>=used[q]);xx=[];yy=[]
  if q:m.add(used[q]<=used[q-1]);m.add(completion[q]>=completion[q-1]+MT).only_enforce_if(used[q])
  for i,v in enumerate(ships):
   b=batch[i][q];m.add(b<=used[q]);m.add(completion[q]>=arrival[i]+Du).only_enforce_if(b);m.add(shipend[i]==completion[q]).only_enforce_if(b);m.add(qindex[i]==q).only_enforce_if(b);wi=math.ceil(v['width']);li=math.ceil(v['length'])
   if wi>W or li>L:m.add(b==0);continue
   xpos=m.new_int_var(0,W-wi,f'x{i}_{q}');ypos=m.new_int_var(0,L-li,f'y{i}_{q}');side=m.new_bool_var(f'side{i}_{q}');m.add(xpos==0).only_enforce_if(side);m.add(xpos==W-wi).only_enforce_if(side.Not());xx.append(m.new_optional_fixed_size_interval_var(xpos,wi,b,f'xi{i}_{q}'));yy.append(m.new_optional_fixed_size_interval_var(ypos,li,b,f'yi{i}_{q}'))
  m.add_no_overlap_2d(xx,yy)
 for i,v in enumerate(ships):
  m.add(sum(batch[i])==1-mode[i]);choices=[]
  for k in v['eligible_berths_B']:
   b=next(b for b in berths if b['id']==k);choice=m.new_bool_var(f'a{i}_{k}');start=m.new_int_var(max(arrival[i],int(round(10*b['start_availability_sk']))),int(round(10*d['berth_end_availability_ek'])),f's{i}_{k}');end=m.new_int_var(0,int(round(10*d['berth_end_availability_ek'])),f'e{i}_{k}');duration=int(round(10*d['handling_times'][i][k]));it=m.new_optional_interval_var(start,duration,end,choice,f'it{i}_{k}');intervals[k].append(it);choices.append(choice);berthchoice[i,k]=choice;berthstart[i,k]=start;berthend[i,k]=end;m.add(shipend[i]==end).only_enforce_if(choice)
  m.add(sum(choices)==mode[i]);terms.append(shipend[i]-arrival[i]+int(v['transshipment_penalty']*10)*mode[i])
 for i in range(n):
  for j in range(n):
   if arrival[i]<arrival[j]:m.add(qindex[i]<=qindex[j]).only_enforce_if([mode[i].Not(),mode[j].Not()]);m.add(shipend[i]<=shipend[j]).only_enforce_if([mode[i].Not(),mode[j].Not()])
 for ints in intervals.values():m.add_no_overlap(ints)
 m.minimize(sum(terms));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,t-1);r=s.solve(m)
 if r not in (cp_model.OPTIMAL,cp_model.FEASIBLE):raise RuntimeError('No lock/berth schedule incumbent')
 modes={str(v['id']):s.value(mode[i]) for i,v in enumerate(ships)};locks={str(v['id']):s.value(qindex[i]) for i,v in enumerate(ships) if not s.value(mode[i])};C={str(q):s.value(completion[q])/10 for q in range(n) if s.value(used[q])};times={str(v['id']):v['arrival_time'] if s.value(mode[i]) else s.value(shipend[i])/10 for i,v in enumerate(ships)};assign={};objective=sum(times[str(v['id'])]-v['arrival_time'] for v in ships)
 for b in berths:
  k=b['id'];inds=sorted([i for i in range(n) if (i,k) in berthchoice and s.value(berthchoice[i,k])],key=lambda i:s.value(berthstart[i,k]));clock=b['start_availability_sk']
  for j,i in enumerate(inds):
   order=n-len(inds)+j;v=ships[i];assign[str(v['id'])]={'berth':k,'order':order};idle=max(0,v['arrival_time']-clock);objective+=(n-order+1)*(d['handling_times'][i][k]+idle)+b['start_availability_sk']-v['arrival_time']+v['transshipment_penalty'];clock=max(clock,v['arrival_time'])+d['handling_times'][i][k]
 return {'objective_value':objective,'modes':modes,'completion_times':times,'lockage_assignments':locks,'lockage_completion_times':C,'berth_assignments':assign}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')
