"""Render scientific figures from the frozen completed-evidence snapshot only."""
import csv, hashlib, json, pathlib, platform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];DATA=ROOT/'data';OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
P=json.loads((DATA/'public.json').read_text());S=json.loads((DATA/'synthetic.json').read_text());SCH=json.loads((DATA/'schedules.json').read_text())
C={'v1':'#7B61A8','v2':'#0072B2','LNS':'#D55E00','egrasp':'#009E73','frozen-v2':'#0072B2','reference':'#666666'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':9,'xtick.labelsize':9,'ytick.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':300,'axes.axisbelow':True})
FIGS=[];CASES=[f'large-{i}' for i in range(1,6)]

def select(a,budget,seed,case):
 rows=[r for r in P if r['algorithm']==a and r['budget_s']==budget and r['seed']==seed and r['instance']==case]
 assert len(rows)==1,(a,budget,seed,case);return rows[0]

def save(fig,id,title,caption,rows,extra=None):
 for ax in fig.axes:
  if ax.axison:ax.tick_params(length=3)
 for ext in ('svg','pdf','png'):fig.savefig(OUT/f'{id}.{ext}',bbox_inches='tight',facecolor='white',metadata={'Creator':'Reproducible matplotlib script'} if ext=='pdf' else None)
 plt.close(fig)
 record={'id':id,'title':title,'caption':caption,'experiments':sorted({r['experiment_id'] for r in rows if 'experiment_id' in r}),'source_records':rows,'files':{ext:str((OUT/f'{id}.{ext}').relative_to(ROOT)) for ext in ('svg','pdf','png')}}
 if extra:record.update(extra)
 FIGS.append(record)

def pair_panel(ax,rows_a,rows_b,labela,labelb,title):
 y=np.arange(len(rows_a));a=np.array([r['objective'] for r in rows_a]);b=np.array([r['objective'] for r in rows_b])
 for i in range(len(a)):
  assert rows_a[i]['input_sha256']==rows_b[i]['input_sha256'],'unmatched inputs'
  ax.plot([a[i],b[i]],[i,i],color='#BCC3CA',lw=2,zorder=1)
 ax.scatter(a,y,c=C[labela],marker='o',s=43,label=labela,zorder=3)
 ax.scatter(b,y,c=C[labelb],marker='s',s=36,label=labelb,zorder=3)
 ax.set(yticks=y,yticklabels=[r['instance'] for r in rows_a],xlabel='Total weighted tardiness',title=title)
 ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(loc='best');ax.margins(x=.13,y=.2)

fig,axes=plt.subplots(1,2,figsize=(10.5,3.6),layout='constrained');rows=[]
for seed,ax in enumerate(axes):
 a=[select('v1',300,seed,c) for c in CASES];b=[select('v2',300,seed,c) for c in CASES];rows+=a+b
 pair_panel(ax,a,b,'v1','v2',f'Seed {seed} | 300 s requested')
save(fig,'01_v1_v2_matched','Algorithm development: v1 versus v2','Five familiar public cases at each of two seeds; identical instance hashes and requested 300-second solver budgets, one worker. Lower is better. v2 has a lower objective in all ten matched case-seed comparisons. These are successive implementations developed using public cases, not an ablation or independent holdout. Native host runs were not CPU-isolated; elapsed times are in the source table.',rows)

fig,axes=plt.subplots(1,2,figsize=(10.5,3.6),layout='constrained');rows=[]
for seed,ax in enumerate(axes):
 a=[select('v2',300,seed,c) for c in CASES];b=[select('LNS',300,seed,c) for c in CASES];rows+=a+b
 pair_panel(ax,a,b,'v2','LNS',f'Seed {seed} | 300 s requested')
save(fig,'02_v2_lns_300','v2 versus CP-SAT LNS at 300 seconds','Matched public inputs, nominal budgets and seed labels. The independent CP-SAT destroy-and-repair implementation wins large-4 at seed 0 and large-5 at seed 1; v2 wins the other eight pairs. Matching seed labels does not synchronize randomness between implementations. Both schedule validators pass. No population confidence interval is inferred from these reused five cases.',rows)
fig,ax=plt.subplots(figsize=(6.8,3.6),layout='constrained');a=[select('v2',1200,0,c) for c in CASES];b=[select('LNS',1200,0,c) for c in CASES]
pair_panel(ax,a,b,'v2','LNS','Seed 0 | 1,200 s requested')
save(fig,'03_v2_lns_1200','v2 versus CP-SAT LNS at 1,200 seconds','Only the completed seed-0 batch is included in this frozen visual release. LNS wins large-5 (25,633 versus 26,113); v2 wins the other four cases. The seed-1 LNS batch was incomplete at the evidence cutoff and is not read or shown. Nominal one-worker budgets match, but host computation is not isolated.',a+b)

fig,(ax,ax2)=plt.subplots(1,2,figsize=(11,4.1),layout='constrained');y=np.arange(5);rows=[]
refs=np.array([select('v2',1200,0,c)['reference'] for c in CASES]);ax.scatter(refs,y,marker='x',s=65,c=C['reference'],label='Published Gurobi incumbent')
for seed,marker,off in [(0,'o',-.12),(1,'s',.12)]:
 rs=[select('v2',1200,seed,c) for c in CASES];rows+=rs;v=np.array([r['objective'] for r in rs]);ax.scatter(v,y+off,c=C['v2'],marker=marker,s=40,label=f'v2 seed {seed}')
 ax2.scatter(100*(refs-v)/refs,y+off,c=C['v2'],marker=marker,s=40,label=f'v2 seed {seed}')
for a in (ax,ax2):a.set(yticks=y,yticklabels=CASES);a.invert_yaxis();a.grid(axis='x',alpha=.2)
ax.set(xlabel='Total weighted tardiness',title='Recorded incumbent versus v2 endpoints');ax.legend(fontsize=8)
ax2.set(xlabel='Reduction relative to reference (%)',title='Reference is not a certified optimum');ax2.axvline(0,color='#888',lw=.7);ax2.legend()
save(fig,'04_published_references','Comparison with published Gurobi incumbents','One FrontierOR task, bierwirth2017, and its five familiar large public cases. v2 endpoints used 1,200-second requested budgets; the published references used approximately 3,600 seconds on unmatched hardware. This is an objective-quality comparison, not a speedup or optimality-gap plot. large-1 reference 30,879.999981 is displayed rounded to 30,880. The seed-1 v2 series combines three validated completed cases before a host reboot and two separately resumed cases; experiment boundaries are retained in the data.',rows,{'additional_sources':['data/references.json','data/sources/evidence/seed1-combined-audit-1791440428687560000/summary.json']})

def synthetic_figure(study,id,title):
 rows=[r for r in S if r['study']==study];names=list(dict.fromkeys(r['instance'] for r in rows));y=np.arange(len(names))
 fig,(ax,im)=plt.subplots(1,2,figsize=(12,6.3),gridspec_kw={'width_ratios':[1.5,1]},layout='constrained')
 for i,name in enumerate(names):
  a=next(r for r in rows if r['instance']==name and r['algorithm']=='frozen-v2');b=next(r for r in rows if r['instance']==name and r['algorithm']=='egrasp')
  if b['objective'] is not None:
   ax.plot([a['objective'],b['objective']],[i,i],color='#BAC2C9',lw=1.8)
   ax.scatter(b['objective'],i+.09,s=31,marker='s',color=C['egrasp'])
   den=b['objective'];pct=100*(den-a['objective'])/den if den else None
   if pct is None:im.text(.02,i,'Both objectives = 0',transform=im.get_yaxis_transform(),va='center',fontsize=9)
   else:im.scatter(pct,i,c=C['v2'] if pct>0 else C['egrasp'],s=35)
  else:
   ax.annotate('EGRASP: deadline failure',(a['objective'],i),xytext=(35,0),textcoords='offset points',va='center',fontsize=8,color='#9C3B35')
   im.text(.02,i,'Undefined: comparator failed',transform=im.get_yaxis_transform(),va='center',fontsize=8,color='#9C3B35')
  ax.scatter(a['objective'],i-.09,s=31,color=C['v2'])
 labels=[n.replace('dev2-','').replace('dev-','').replace('tf','TF ').replace('3_2','1.5').replace('7_4','1.75') for n in names]
 ax.set(yticks=y,yticklabels=labels,xlabel='Total weighted tardiness',title=title);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
 ax.scatter([],[],c=C['v2'],label='v2');ax.scatter([],[],c=C['egrasp'],marker='s',label='EGRASP');ax.legend(loc='lower right')
 im.set(yticks=y,yticklabels=[],xlabel='(EGRASP − v2) / EGRASP (%)',title='Paired endpoint difference');im.invert_yaxis();im.axvline(0,c='#777',lw=.7);im.grid(axis='x',alpha=.2)
 for a in (ax,im):a.set_ylim(len(names)-.4,-.6)
 for j in [1.5,3.5,5.5,7.5,9.5]:
  for a in (ax,im):a.axhline(j,c='#E1E4E8',lw=.7)
 if study=='development_1':
  ax.set_xlim(-800,42000);im.set_xlim(-5,24)
  cap='First custom development distribution: TF 2, 3, 4 crossed with uniform/priority weights, two cases per stratum. One execution per solver. All twelve pairs shown: two v2 numeric wins, one EGRASP numeric win, eight zero ties, one valid v2 result paired with an EGRASP deadline failure. Failure has no numeric objective or improvement percentage. Internal budgets differed (v2 300 s, EGRASP 298 s); external deadline 300 s. Zero ties are retained and are not omitted from any count.'
 else:
  im.set_xlim(0,18)
  cap='Second, separately designed development distribution: exact TF 1.5, 1.75, 2 with release + ceil(TF × job processing sum), uniform/priority weights and two cases per stratum. Both implementations requested 290 s within 300 s external deadlines. All 24 schedules were valid and in time. v2 had lower objectives in all twelve pairs; arithmetic mean per-case reduction 9.73%. This distribution was selected after the first study, so the studies are not pooled and these are not confirmatory results. One execution per solver; no inferential error bars.'
 cap+=' Both studies use a historical author-provided EGRASP binary in Linux x86 emulation, one CPU quota and 512 MiB. EGRASP randomness is undocumented; host contention is a limitation.'
 save(fig,id,title,cap,rows)
synthetic_figure('development_1','05_development_first','First development study: TF 2, 3, 4')
synthetic_figure('development_2','06_development_harder','Second development study: TF 1.5, 1.75, 2')

# Descriptive paired effects, keeping seeds separate and all public cases visible.
fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained');rows=[]
for ax,budget,seed in zip(axes,[300,300,1200],[0,1,0]):
 a=[select('v2',budget,seed,c) for c in CASES];b=[select('LNS',budget,seed,c) for c in CASES];rows+=a+b
 pct=[100*(y['objective']-x['objective'])/y['objective'] for x,y in zip(a,b)]
 ax.barh(CASES,pct,color=[C['v2'] if v>0 else C['LNS'] for v in pct]);ax.invert_yaxis();ax.axvline(0,color='#555',lw=.8);ax.set(title=f'{budget:,} s | seed {seed}',xlabel='(LNS − v2) / LNS (%)');ax.set_xlim(-6,14);ax.grid(axis='x',alpha=.2)
save(fig,'07_public_paired_effects','Signed paired differences retain both methods’ wins','Positive values favor v2; negative values favor the independent LNS implementation. Each bar is one observed case-seed pair, not an average across repeats. Same completed evidence as Figures 2 and 3. No confidence intervals because these reused cases and two seeds do not define the preregistered confirmatory sample.',rows)

fig,ax=plt.subplots(figsize=(9,4.0),layout='constrained');groups=[];rows=[]
for name,other,budget,seeds in [('v2 vs v1 | 300 s','v1',300,[0,1]),('v2 vs LNS | 300 s','LNS',300,[0,1]),('v2 vs LNS | 1,200 s','LNS',1200,[0])]:
 counts=[0]*5
 for seed in seeds:
  for case in CASES:
   a=select('v2',budget,seed,case);b=select(other,budget,seed,case);rows += [a,b];counts[0 if a['objective']<b['objective'] else 1 if a['objective']==b['objective'] else 2]+=1
 groups.append((name,counts))
for study,title in [('development_1','Synthetic development 1'),('development_2','Synthetic development 2')]:
 rr=[r for r in S if r['study']==study];rows+=rr;counts=[0]*5
 for name in dict.fromkeys(r['instance'] for r in rr):
  a=next(r for r in rr if r['instance']==name and r['algorithm']=='frozen-v2');b=next(r for r in rr if r['instance']==name and r['algorithm']=='egrasp')
  if b['objective'] is None:counts[3]+=1
  elif a['objective'] is None:counts[4]+=1
  else:counts[0 if a['objective']<b['objective'] else 1 if a['objective']==b['objective'] else 2]+=1
 groups.append((title,counts))
colors=[C['v2'],'#AAAAAA',C['LNS'],'#E5B75A','#9B3842'];labels=['v2 lower','Tie','Comparator lower','v2 valid, comparator failed','v2 failed']
left=np.zeros(len(groups))
for j,(color,label) in enumerate(zip(colors,labels)):
 counts=np.array([g[1][j] for g in groups]);width=100*counts/np.array([sum(g[1]) for g in groups]);ax.barh(np.arange(len(groups)),width,left=left,color=color,label=label)
 for i,(n,w,l) in enumerate(zip(counts,width,left)):
  if n:ax.text(l+w/2,i,str(n),ha='center',va='center',color='white' if j in (0,2,4) else '#222',fontsize=10)
 left+=width
ax.set(yticks=np.arange(len(groups)),yticklabels=[f'{g[0]} (n={sum(g[1])})' for g in groups],xlabel='Share of observed pairs (%)',xlim=(0,100),title='Descriptive outcomes; counts printed inside bars');ax.invert_yaxis();ax.legend(loc='upper center',bbox_to_anchor=(.4,-.2),ncol=2)
save(fig,'08_win_tie_loss_failure','Observed wins, ties, losses and failures','Counts and percentages summarize exactly the plotted completed comparisons. n denotes case-seed pairs for public studies and cases for the synthetic studies. Repeated seeds on one public instance are not independent instances. The first synthetic deadline failure remains a distinct category rather than a numeric win. Startup/build failures of earlier candidates are documented in the research diagram and inventory, outside these matched successful-implementation cohorts.',rows)

fig,axes=plt.subplots(1,5,figsize=(13,3.5),sharey=True,layout='constrained');rows=[]
for ax,case in zip(axes,CASES):
 for alg,offset,marker in [('v2',-12,'o'),('LNS',12,'s')]:
  rr=[select(alg,t,0,case) for t in [300,1200]];rows+=rr
  ax.scatter([r['budget_s']+offset for r in rr],[r['objective'] for r in rr],color=C[alg],marker=marker,label=alg,s=38)
 ax.set(title=case,xticks=[300,1200],xticklabels=['300','1,200'],xlim=(180,1320),xlabel='Requested budget (s)');ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Final total weighted tardiness');axes[-1].legend()
save(fig,'09_budget_endpoints','Independent final objectives at two budgets','Each point is the final objective of a distinct seed-0 run. Points are deliberately not joined: they are not timestamped incumbents from one search trajectory. All five matched public cases are retained. A larger runtime budget can change stochastic search decisions and does not guarantee monotonic outcomes between independently executed runs.',rows)

# Saved schedule analysis: occupation bars are exact input durations at saved starts.
completion_rows=[]
for case in ['large-1','large-5']:
 pair=[SCH[f'{alg}_{case}'] for alg in ['v2','LNS']];assert pair[0]['instance']==pair[1]['instance'];instance=pair[0]['instance']
 fig,axes=plt.subplots(2,1,figsize=(14,7),sharex=True,layout='constrained');makespans=[];stats=[]
 for ax,bundle,alg in zip(axes,pair,['v2','LNS']):
  horizon=max(j['completion_time'] for j in bundle['solution']['schedule']);makespans.append(horizon);busy=np.zeros(instance['num_machines'])
  for sj,job in zip(sorted(bundle['solution']['schedule'],key=lambda x:x['job_id']),sorted(instance['jobs'],key=lambda x:x['job_id'])):
   assert sj['job_id']==job['job_id'];tardy=max(0,sj['completion_time']-job['due_date']);color='#DA8C62' if tardy>0 else '#78A9C4'
   completion_rows.append(dict(instance=case,algorithm=alg,job_id=job['job_id'],release=job['release_date'],due=job['due_date'],completion=sj['completion_time'],weight=job['weight'],tardiness=tardy,weighted_tardiness=tardy*job['weight'],makespan=horizon,experiment_id=bundle['record']['experiment_id']))
   for k,(op,actual) in enumerate(zip(job['operations'],sj['operations'])):
    m=op['machine'];start=actual['start_time'];duration=op['processing_time'];busy[m]+=duration
    ax.broken_barh([(start,duration)],(m-.38,.76),facecolors=color,edgecolors='white',linewidth=.35)
    if duration>=27:ax.text(start+duration/2,m,str(job['job_id']),ha='center',va='center',fontsize=6,color='#152A35',clip_on=True)
  stats.append(100*busy/horizon)
  ax.set(yticks=range(10),yticklabels=[f'M{i}' for i in range(10)],ylabel='Machine',title=f"{alg} | {case} | seed 0 | 1,200 s | objective {bundle['record']['objective']:,}",ylim=(9.7,-.7));ax.grid(axis='x',alpha=.2)
 axes[-1].set_xlabel('Schedule time (instance units, not computation seconds)');axes[0].legend(handles=[Patch(color='#78A9C4',label='Job completed by its due date'),Patch(color='#DA8C62',label='Job completed late')],loc='upper left',bbox_to_anchor=(0,1.22),ncol=2,fontsize=8)
 save(fig,f'10_gantt_{case}',f'Validated machine schedules: {case}','Exact operation intervals reconstructed from saved validated schedules and input processing times. Bar labels are zero-based job IDs where width permits; every operation remains in the source data. Bar color denotes whether its job ultimately finished late, not whether that operation alone violated a deadline. Shared time axis; machine IDs are zero-based. These are schedule timelines, not optimization convergence. Both runs use seed 0 and 1,200-second requested budgets. '+('v2 wins this instance.' if case=='large-1' else 'LNS wins this instance.'),[b['record'] for b in pair])
 fig,(ax,util)=plt.subplots(1,2,figsize=(11,8),gridspec_kw={'width_ratios':[2,1]},layout='constrained');rr=[r for r in completion_rows if r['instance']==case]
 for alg,off,marker in [('v2',-.12,'o'),('LNS',.12,'s')]:
  aa=sorted([r for r in rr if r['algorithm']==alg],key=lambda x:x['job_id']);yy=np.arange(len(aa))+off
  ax.scatter([r['completion'] for r in aa],yy,s=21,marker=marker,color=C[alg],label=alg,zorder=3)
  for y,r in zip(yy,aa):
   if r['tardiness']>0:ax.plot([r['due'],r['completion']],[y,y],color=C[alg],alpha=.45,lw=1)
 due=[j['due_date'] for j in sorted(instance['jobs'],key=lambda j:j['job_id'])];ax.scatter(due,np.arange(30),marker='|',s=65,color='#333',label='Due date')
 ax.set(yticks=range(30),yticklabels=range(30),ylabel='Job ID',xlabel='Schedule time (instance units)',title=f'{case}: due dates and completion');ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right')
 yy=np.arange(10);util.barh(yy-.17,stats[0],height=.33,color=C['v2'],label=f'v2, horizon {makespans[0]}');util.barh(yy+.17,stats[1],height=.33,color=C['LNS'],label=f'LNS, horizon {makespans[1]}');util.set(yticks=yy,yticklabels=[f'M{i}' for i in yy],xlim=(0,100),xlabel='Machine busy time / own makespan (%)',title='Schedule occupancy');util.invert_yaxis();util.legend(fontsize=8,loc='lower left')
 save(fig,f'11_completion_occupancy_{case}',f'Job completion and machine occupancy: {case}','Each vertical tick is the actual job due date. Dots/squares show saved completion times; segments show positive tardiness. Objective contributions are weight × max(0, completion − due), exported per job in data/job-completions.csv. Occupancy divides each machine’s fixed processing workload by that algorithm’s own schedule makespan; it is not CPU utilization, and higher occupancy is not necessarily lower weighted tardiness. Same schedules as the corresponding Gantt figure.',[b['record'] for b in pair])
with (DATA/'job-completions.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(completion_rows[0]));w.writeheader();w.writerows(completion_rows)

# Measured categorical stage summaries, not an ablation and not a trajectory.
fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,layout='constrained');rows=[]
for ax,budget in zip(axes,[300,1200]):
 for i,case in enumerate(CASES):
  r=select('LNS',budget,0,case);rows.append(r);m=r['metadata'];v=[m['dispatch_objective'],m['pre_lns_objective'],r['objective']]
  ax.scatter(np.arange(3)+(i-2)*.07,v,s=30,label=case)
 ax.set(xticks=range(3),xticklabels=['Dispatch','Pre-LNS search','Final'],title=f'Seed 0 | {budget:,} s requested',xlabel='Recorded stage (no timestamp)');ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Total weighted tardiness');axes[1].legend(fontsize=8)
save(fig,'12_lns_recorded_stages','Recorded LNS stage endpoints','Three objective values recorded by the completed LNS runs: initial dispatch, the pre-LNS local search endpoint, and final schedule objective. Categorical points have no time interpretation. These are within-run stage summaries, not controlled ablations: no component was removed under otherwise matched conditions. Intermediate stages did not save validated schedules in this evidence bundle; they are solver-reported diagnostics, while final points are independently validated.',rows)

fig,ax=plt.subplots(figsize=(12,6));ax.axis('off');ax.set_xlim(0,12);ax.set_ylim(0,6)
def box(x,y,w,h,text,color='#EDF2F5',size=9):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=color,edgecolor='#6F808C',lw=.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#526472','lw':1.2})
box(.15,4.3,1.9,.85,'Sagent\nresearch and propose',size=8.5);box(2.55,4.3,1.9,.85,'Sagent\nimplement candidate',size=8.5);box(4.95,4.3,1.9,.85,'Trackinizer\nregister experiment',size=8.5);box(7.35,4.3,1.9,.85,'Evaluation harness\nrun and validate',size=8.5);box(9.75,4.3,1.9,.85,'Trackinizer\noutcome + conclusion\nread back',size=8.5)
for x in [2.1,4.5,6.9,9.3]:arrow((x,4.73),(x+.4,4.73))
ax.annotate('',xy=(1.1,4.2),xytext=(10.7,4.2),arrowprops={'arrowstyle':'->','connectionstyle':'arc3,rad=-.25','color':'#526472'})
ax.text(6,3.5,'Recorded state informs the next investigation',ha='center',fontsize=9)
box(.3,2,2.0,.9,'v1\n2 build failures\nthen valid C++',size=9);box(3,2,2,.9,'v2\nincremental search\nand evolving elites',size=9);box(5.7,2,2,.9,'Independent LNS\nstartup failure\nthen repaired',size=9);box(8.4,2,3.1,.9,'Matched public-case tests\n300 s and 1,200 s\nresults retained',size=9)
for x in [2.4,5.1,7.8]:arrow((x,2.45),(x+.5,2.45))
box(.3,.3,11.2,.85,'Codex: integration, frozen comparison harness, source/data checks, independent audits,\nsynthetic development studies, evidence preservation and publication figures',color='#E5EDF7',size=10)
ax.text(6,5.65,'Observed research workflow and contributions',ha='center',fontsize=14)
inv=json.loads((DATA/'experiment-inventory.json').read_text())
save(fig,'13_research_evolution','Research workflow and recorded variants','Top: the custom integrated research loop, not a claim that the default public components enforce persistence automatically. Middle: selected implemented candidate stages in chronological order, including two v1 build failures and the initial LNS startup failure. Arrows show recorded progression, not a causal ablation or proof of novelty. Sagent proposed and implemented research variants; Codex supplied substantial infrastructure, experimental controls and independent validation. Trackinizer experiment and conclusion readbacks are preserved in data/trackinizer. The active confirmatory evaluation is not an outcome in this diagram.',[{'experiment_id':x['experiment_id']} for x in inv])

# All observations shown, avoiding smooth density estimates on tiny samples.
fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout='constrained');rows=[]
for ax,study,title in zip(axes,['development_1','development_2'],['First distribution','Second distribution']):
 rr=[r for r in S if r['study']==study];rows+=rr;strata=list(dict.fromkeys(r['stratum'] for r in rr))
 for j,stratum in enumerate(strata):
  for alg,off,marker in [('frozen-v2',-.12,'o'),('egrasp',.12,'s')]:
   values=[r['objective'] for r in rr if r['stratum']==stratum and r['algorithm']==alg and r['objective'] is not None]
   ax.scatter([j+off+(k-(len(values)-1)/2)*.065 for k in range(len(values))],values,marker=marker,s=30,color=C[alg])
 ax.set(xticks=range(6),xticklabels=[x.replace('tf','').replace('uniform','U').replace('priority','P') for x in strata],xlabel='TF and weight stratum (U: uniform; P: priority)',ylabel='Total weighted tardiness',title=title);ax.grid(axis='y',alpha=.2)
 if study=='development_1':ax.text(.02,.95,'One missing EGRASP objective: deadline failure\nEight paired zero ties retained',transform=ax.transAxes,va='top',fontsize=8)
 ax.scatter([],[],c=C['v2'],label='v2');ax.scatter([],[],c=C['egrasp'],marker='s',label='EGRASP');ax.legend(loc='upper right',fontsize=8)
save(fig,'14_difficulty_strata','Observed objective distributions by difficulty stratum','Individual endpoint observations, two instances per stratum per solver. No kernel density, interpolation or inferential interval is fitted. Exact zero values are retained, and the first study’s failed execution has no objective rather than a fabricated penalty. Distributions remain separate because the second design was selected after reviewing the first. Axis scales differ and objectives also reflect instance weights.',rows)

(ROOT/'captions.json').write_text(json.dumps(FIGS,indent=2)+'\n')
files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ('.svg','.pdf','.png')}
(ROOT/'render-manifest.json').write_text(json.dumps({'python':platform.python_version(),'matplotlib':matplotlib.__version__,'numpy':np.__version__,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'data_manifest_sha256':hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),'figures':files},indent=2)+'\n')
print(f'Rendered {len(FIGS)} figures in SVG, PDF and PNG')
