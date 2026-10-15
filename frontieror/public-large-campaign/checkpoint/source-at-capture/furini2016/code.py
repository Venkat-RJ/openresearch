import argparse,json,time,random,collections

def solve(d,t):
 items=d['items'];deadline=time.monotonic()+max(.1,t-1);rng=random.Random(12);best=(0,{})
 for run in range(1000000):
  if time.monotonic()>deadline:break
  remain={v['id']:v['copies'] for v in items};plates=[(d['panel']['length'],d['panel']['width'])];taken=collections.Counter();profit=0
  while plates and time.monotonic()<deadline:
   L,W=plates.pop();candidates=[v for v in items if remain[v['id']]>0 and v['length']<=L and v['width']<=W]
   if not candidates:continue
   v=max(candidates,key=lambda v:v['profit']/(v['length']*v['width'])*(1 if run==0 else rng.uniform(.5,1.5)));l=v['length'];w=v['width'];taken[v['id']]+=1;remain[v['id']]-=1;profit+=v['profit']
   if rng.randrange(2):plates.extend([(L-l,W),(l,W-w)])
   else:plates.extend([(L,W-w),(L-l,w)])
   plates=[p for p in plates if p[0]>0 and p[1]>0]
  if profit>best[0]:best=(profit,dict(taken))
 return {'objective_value':best[0],'items_selected':[{'item_id':v['id'],'plate_dims':[v['length'],v['width']],'copies':best[1][v['id']]} for v in items if best[1].get(v['id'],0)]}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')
