import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,bisect,math,collections,random
from ortools.sat.python import cp_model

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'):p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args();deadline=time.monotonic()+max(.1,float(a.time_limit)-2);d=json.load(open(a.instance_path));old='bin_capacity'in d;cap=d['bin_capacity']if old else d['c'];items=d['items']if old else [w for w,q in zip(d['weights'],d['demands']) for _ in range(q)];items=sorted(items,reverse=True);bins=[];free=[]
 for w in items:
  pos=bisect.bisect_left(free,(w,-1))
  if pos==len(free):j=len(bins);bins.append([]);r=cap
  else:r,j=free.pop(pos)
  bins[j].append(w);bisect.insort(free,(r-w,j))
 n=len(items);upper=len(bins);lower=math.ceil(sum(items)/cap)
 if n>120 and min(items)>=cap/4 and max(items)<=cap/2:
  demand=collections.Counter(items);patterns={tuple(sorted(row))for row in bins};rng=random.Random(1999);construction_end=time.monotonic()+min(12,max(.1,(deadline-time.monotonic())*.35))
  for attempt in range(16):
   counts=demand.copy();sizes=sorted(counts);candidate=[]
   while sizes and time.monotonic()<construction_end:
    w=sizes[-1];counts[w]-=1
    if not counts[w]:sizes.pop()
    row=[w];remaining=cap-w;l=0;r=len(sizes)-1;choices=[];best_pair=-1
    while l<=r:
     x,y=sizes[l],sizes[r]
     if x+y>remaining:r-=1;continue
     if l==r and counts[x]<2:break
     total=x+y
     if total>=best_pair-12:
      if total>best_pair:best_pair=total;choices=[pair for pair in choices if pair[0]>=best_pair-12]
      choices.append((total,x,y))
     l+=1
    if choices:
     _,x,y=max(choices)if attempt==0 else rng.choice(choices);row.extend([x,y]);remaining-=x+y
     for v in (x,y):
      counts[v]-=1
      if not counts[v]:sizes.remove(v)
    while sizes:
     index=bisect.bisect_right(sizes,remaining)-1
     if index<0:break
     v=sizes[index];row.append(v);remaining-=v;counts[v]-=1
     if not counts[v]:sizes.pop(index)
    candidate.append(row)
   if sizes:break
   patterns.update(tuple(sorted(row))for row in candidate)
   if len(candidate)<len(bins):bins=candidate
  if time.monotonic()<deadline:
   rows=sorted(patterns);m=cp_model.CpModel();variables=[m.NewIntVar(0,n,'pattern%d'%i)for i in range(len(rows))];incumbent=collections.Counter(tuple(sorted(row))for row in bins);by_size={v:[]for v in demand}
   for row,var in zip(rows,variables):
    for v,q in collections.Counter(row).items():by_size[v].append(q*var)
    m.AddHint(var,incumbent[row])
   for v,q in demand.items():m.Add(sum(by_size[v])==q)
   m.Add(sum(variables)>=lower);m.Minimize(sum(variables));solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=max(.01,deadline-time.monotonic());status=solver.Solve(m)
   if status in(cp_model.FEASIBLE,cp_model.OPTIMAL):
    candidate=[list(row)for row,var in zip(rows,variables)for _ in range(solver.Value(var))]
    if len(candidate)<len(bins):bins=candidate
  upper=len(bins)
 if n<=120 and upper>lower and time.monotonic()<deadline:
  m=cp_model.CpModel();used=[m.NewBoolVar('u'+str(b))for b in range(upper)];x=[[m.NewBoolVar('x%d_%d'%(i,b))for b in range(min(i+1,upper))]for i in range(n)]
  for row in x:m.Add(sum(row)==1)
  for b in range(upper):
   m.Add(sum(items[i]*x[i][b] for i in range(n)if b<len(x[i]))<=cap*used[b])
   if b:m.Add(used[b]<=used[b-1])
  m.Add(sum(used)>=lower);m.Minimize(sum(used));s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,deadline-time.monotonic());status=s.Solve(m)
  if status in(cp_model.FEASIBLE,cp_model.OPTIMAL):
   candidate=[[items[i]for i in range(n)if b<len(x[i])and s.Value(x[i][b])]for b in range(upper)];candidate=[b for b in candidate if b]
   if len(candidate)<len(bins):bins=candidate
 sol={'objective_value':len(bins),'num_bins':len(bins),'bin_assignments':[{'items':b,'total_size':sum(b)}for b in bins]if old else bins};json.dump(sol,open(a.solution_path,'w'))
if __name__=='__main__':main()
