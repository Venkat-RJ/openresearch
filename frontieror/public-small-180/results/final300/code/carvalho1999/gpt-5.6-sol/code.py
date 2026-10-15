import argparse,json,time,random,math,os,itertools,collections
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();deadline=time.monotonic()+max(.1,a.time_limit-2)
 d=json.load(open(a.instance_path));W=d['bin_capacity'];items=d['items'];rng=random.Random(0);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def pack(seq):
  bins=[];buckets=[[] for _ in range(W+1)];mask=0
  for x in seq:
   available=mask>>x
   if available:
    remain=x+(available&-available).bit_length()-1;j=buckets[remain].pop()
    if not buckets[remain]:mask&=~(1<<remain)
    bins[j].append(x);r=remain-x;buckets[r].append(j);mask|=1<<r
   else:
    bins.append([x]);r=W-x;buckets[r].append(len(bins)-1);mask|=1<<r
  return bins
 best=pack(sorted(items,reverse=True));lb=math.ceil(sum(items)/W)
 def save():
  s={'objective_value':len(best),'num_bins':len(best),'bin_assignments':[{'items':b,'total_size':sum(b)} for b in best]};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(len(best),s)
 save()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 counts=collections.Counter(items);widths=sorted(counts);patterns=[]
 if sum(items)==lb*W and len(items)==3*lb:
  for i,x in enumerate(widths):
   for j in range(i,len(widths)):
    z=W-x-widths[j]
    if z<widths[j] or z not in counts:continue
    pat=(x,widths[j],z)
    if all(collections.Counter(pat)[v]<=counts[v] for v in pat):patterns.append(pat)
 if patterns and len(patterns)<1900:
  model=gp.Model();model.Params.Seed=0;model.Params.Threads=1;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;y=model.addVars(len(patterns),vtype=GRB.INTEGER,lb=0)
  for w in widths:model.addConstr(gp.quicksum(patterns[r].count(w)*y[r] for r in range(len(patterns)) if w in patterns[r])==counts[w])
  model.setObjective(y.sum())
  def cb(md,where):
   nonlocal best
   if where==GRB.Callback.MIPSOL:
    bins=[list(pat) for r,pat in enumerate(patterns) for _ in range(round(md.cbGetSolution(y[r])))];best=bins;save()
  model.Params.TimeLimit=max(.1,(deadline-time.monotonic())*.85);model.optimize(cb)
  if model.SolCount:
   best=[list(pat) for r,pat in enumerate(patterns) for _ in range(round(y[r].X))];save()
   if len(best)==lb:return
 it=0
 while time.monotonic()<deadline and len(best)>lb:
  count=min(len(best),max(8,min(100,len(best)//4)))
  if it%2:chosen=rng.sample(range(len(best)),count)
  else:chosen=sorted(range(len(best)),key=lambda j:sum(best[j]))[:count]
  cs=set(chosen);seq=[x for j in chosen for x in best[j]]
  if it%3==0:seq.sort(reverse=True)
  elif it%3==1:seq.sort(key=lambda x:x*(.85+.3*rng.random()),reverse=True)
  else:rng.shuffle(seq)
  new=pack(seq)
  if len(new)<len(chosen):best=[b for j,b in enumerate(best) if j not in cs]+new;save()
  it+=1
 save()
if __name__=='__main__':main()
