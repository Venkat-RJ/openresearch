import argparse,json,time,functools,collections
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+max(.1,a.time_limit-3)
z=json.load(open(a.instance_path));items=z['items'];L=z['panel']['length'];W=z['panel']['width'];m=len(items);limits=tuple(it['copies'] for it in items);zero=(0,)*m
logger=SolutionLogger(a.log_path,sense='maximize');logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start);incumbent=-1

def emit(root):
 global incumbent
 value=root[0]
 if value<=incumbent:return
 plates=[{'id':0,'length':L,'width':W,'in_J_bar':False}];mapping={(L,W):0};cuts=collections.Counter();selected=collections.Counter()
 def plate(l,w):
  if (l,w) not in mapping:
   mapping[l,w]=len(plates);plates.append({'id':len(plates),'length':l,'width':w,'in_J_bar':False})
  return mapping[l,w]
 def build(l,w,plan):
  pid=plate(l,w);choice=plan[2]
  if choice is None:return
  if choice[0]=='item':selected[(choice[1],pid)]+=1;plates[pid]['in_J_bar']=True;return
  orient,q=choice;dims=((q,w),(l-q,w)) if orient=='v' else ((l,q),(l,w-q));c1=plate(*dims[0]);c2=plate(*dims[1]);cuts[(pid,orient,q,c1,c2)]+=1
  for child in dims:build(*child,dp(*child))
 build(L,W,root)
 s={'objective_value':value,'items_selected':[{'item_id':items[i]['id'],'plate_id':pid,'plate_dims':[items[i]['length'],items[i]['width']],'copies':copies} for (i,pid),copies in selected.items()],'cuts_used':[{'parent_plate_id':pid,'orientation':o,'position':q,'child1_plate_id':c1,'child2_plate_id':c2,'count':count} for (pid,o,q,c1,c2),count in cuts.items()],'plates':plates,'runtime':time.monotonic()-start}
 json.dump(s,open(a.solution_path,'w'));logger.log_solution(value,s);incumbent=value

@functools.lru_cache(None)
def dp(l,w):
 best=(0,zero,None)
 for i,it in enumerate(items):
  if (l,w)==(it['length'],it['width']) and limits[i]>0 and it['profit']>best[0]:
   counts=list(zero);counts[i]=1;best=(it['profit'],tuple(counts),('item',i))
 if (l,w)==(L,W):emit(best)
 if time.monotonic()>=end:return best
 for orient,dim in [('v',l),('h',w)]:
  positions=range(1,dim//2+1) if max(L,W)<=150 else sorted({it['length' if orient=='v' else 'width'] for it in items if it['length' if orient=='v' else 'width']<dim})
  for q in positions:
   if time.monotonic()>=end:return best
   x=dp(q,w) if orient=='v' else dp(l,q);y=dp(l-q,w) if orient=='v' else dp(l,w-q);value=x[0]+y[0]
   if value<=best[0]:continue
   counts=tuple(u+v for u,v in zip(x[1],y[1]))
   if all(u<=v for u,v in zip(counts,limits)):
    best=(value,counts,(orient,q))
    if (l,w)==(L,W):emit(best)
 return best
emit(dp(L,W))
