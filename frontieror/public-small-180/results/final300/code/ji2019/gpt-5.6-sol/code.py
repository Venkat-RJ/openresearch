import argparse,json,time,math,functools
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+max(.1,a.time_limit-4);z=json.load(open(a.instance_path));n=z['num_ships'];m=z['num_berths'];ships=z['ships'];W=z['chamber_width_W'];L=z['chamber_length_L'];Du=z['lockage_duration_Du'];MT=z['min_time_between_lockages_MT'];sc=z['lock_start_availability_sc'];arr=[q['arrival_time'] for q in ships];handling=z['handling_times'];sk=[q['start_availability_sk'] for q in z['berths']];ek=z.get('berth_end_availability_ek',1000);logger=SolutionLogger(a.log_path,sense='minimize');best=float('inf');full=(1<<n)-1
logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
mooring_patterns={}
@functools.lru_cache(None)
def packing(mask):
 group=sorted([i for i in range(n) if mask>>i&1],key=lambda i:ships[i]['length'],reverse=True);placed={};moorings={};nodes=0
 def fit(k):
  nonlocal nodes
  nodes+=1
  if nodes>20000 or time.monotonic()>=end:return False
  if k==len(group):return True
  i=group[k];w=ships[i]['width'];l=ships[i]['length'];positions=[]
  for x,kind in [(W-w,'ml_right')]:
   if abs(x-round(x))<1e-7:
    ys=sorted({0}|{math.ceil(y+ships[j]['length']-1e-9) for j,(xx,y) in placed.items()});positions.extend((int(round(x)),y,{'kind':kind}) for y in ys)
  for j,(xx,yy) in placed.items():
   if j not in ships[i]['mooring_set_MR'] or ships[j]['length']<l:continue
   for x,kind in [(xx-w,'ml'),(xx+ships[j]['width'],'mr')]:
    if abs(x-round(x))<1e-7:
     upper=math.floor(yy+ships[j]['length']-l+1e-8);ys={yy,upper}|{math.ceil(y+ships[u]['length']-1e-9) for u,(xxx,y) in placed.items()};positions.extend((int(round(x)),y,{'kind':kind,'ship':j}) for y in sorted(ys) if yy<=y<=upper)
  for x,y,mooring in positions:
   if x<0 or y<0 or x+w>W+1e-8 or y+l>L+1e-8:continue
   if any(not(x+w<=xx+1e-8 or xx+ships[j]['width']<=x+1e-8 or y+l<=yy+1e-8 or yy+ships[j]['length']<=y+1e-8) for j,(xx,yy) in placed.items()):continue
   placed[i]=(x,y);moorings[i]=mooring
   if fit(k+1):return True
   del placed[i];del moorings[i]
  return False
 if fit(0):mooring_patterns[mask]=dict(moorings);return dict(placed)
 return None

berth=[]
for k in range(m):
 table={0:(0,[])}
 for mask in range(1,full+1):
  group=[i for i in range(n) if mask>>i&1]
  if any(k not in ships[i]['eligible_berths_B'] for i in group):continue
  for order in [sorted(group,key=lambda i:arr[i]),sorted(group,key=lambda i:handling[i][k])]:
   current=sk[k];value=0
   for j,i in enumerate(order):
    idle=max(0,arr[i]-current);weight=len(order)-j+1;value+=weight*(handling[i][k]+idle)+sk[k]-arr[i]+ships[i]['transshipment_penalty'];current+=idle+handling[i][k]
   if current<=ek and (mask not in table or value<table[mask][0]):table[mask]=(value,order)
 berth.append(table)
combined={mask:(value,[(0,mask)]) for mask,(value,order) in berth[0].items()}
for k in range(1,m):
 new={}
 for mask in range(full+1):
  sub=mask
  while True:
   other=mask^sub
   if sub in berth[k] and other in combined:
    value=berth[k][sub][0]+combined[other][0]
    if mask not in new or value<new[mask][0]:new[mask]=(value,combined[other][1]+[(k,sub)])
   if sub==0:break
   sub=(sub-1)&mask
 combined=new

def emit(trans,value,alloc,groups,times):
 global best
 if value>=best:return
 best=value;s={'objective_value':best,'modes':{str(i):int(trans>>i&1) for i in range(n)},'completion_times':{str(i):arr[i] for i in range(n)},'lockage_assignments':{},'lockage_completion_times':{},'berth_assignments':{},'placement':{},'mooring':{},'runtime':time.monotonic()-start}
 for q,group in enumerate(groups):
  s['lockage_completion_times'][str(q)]=times[q];mask=sum(1<<i for i in group)
  for i in group:s['completion_times'][str(i)]=times[q];s['lockage_assignments'][str(i)]=q;s['placement'][str(i)]=list(packing(mask)[i]);s['mooring'][str(i)]=mooring_patterns[mask][i]
 for k,mask in alloc:
  order=berth[k][mask][1]
  for j,i in enumerate(order):s['berth_assignments'][str(i)]={'berth':k,'order':n-len(order)+j}
 json.dump(s,open(a.solution_path,'w'));logger.log_solution(best,s)

def lock_solution(mask,on_incumbent):
 seq=sorted([i for i in range(n) if mask>>i&1],key=lambda i:arr[i]);inc=float('inf');answer=None
 def visit(pos,previous,value,groups,times):
  nonlocal inc,answer
  if value>=inc or time.monotonic()>=end:return
  if pos==len(seq):inc=value;answer=(groups.copy(),times.copy());on_incumbent(value,groups,times);return
  groupmask=0
  for j in range(pos,len(seq)):
   groupmask|=1<<seq[j]
   if packing(groupmask) is None:continue
   completion=max(sc+Du,arr[seq[j]]+Du,previous+MT if groups else sc+Du)
   visit(j+1,completion,value+sum(completion-arr[i] for i in seq[pos:j+1]),groups+[seq[pos:j+1]],times+[completion])
 visit(0,0,0,[],[])
 return inc,answer
for trans in range(full+1):
 if time.monotonic()>=end:break
 if trans not in combined:continue
 bv,alloc=combined[trans];lock_solution(full^trans,lambda value,groups,times:emit(trans,bv+value,alloc,groups,times))
