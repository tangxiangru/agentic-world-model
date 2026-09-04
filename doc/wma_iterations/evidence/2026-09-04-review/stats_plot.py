import json,statistics,collections,datetime,math,os
from pathlib import Path
os.environ['MPLCONFIGDIR']='/tmp/wma-mpl'
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('/tmp/wma-deep-analysis');cs=json.loads((p/'complete-cells.json').read_text())
groups={}
for c in cs:groups.setdefault(c['batch'],[]).append(c)
def batch(suffix):return next(b for b in groups if b.endswith(suffix))
bl=batch('r02-v02-baseline-x4-v2');ct=batch('r02-ctl-x4-v2')
comparisons={}
for b in [bl,batch('r02-a-format-floor-x4-v2'),batch('r02-ab-format-floor-width-x4-v2'),batch('r02-c-probe-before-fail-x4-v2'),batch('r02-d-checkpoint-precondition-x4-v2')]:
 ref=ct if b==bl else bl;x=np.array([c['accuracy'] for c in groups[b]]);y=np.array([c['accuracy'] for c in groups[ref]])
 se=np.sqrt(np.var(x,ddof=1)/len(x)+np.var(y,ddof=1)/len(y));df=se**4/((np.var(x,ddof=1)/len(x))**2/(len(x)-1)+(np.var(y,ddof=1)/len(y))**2/(len(y)-1));delta=np.mean(x)-np.mean(y);half=stats.t.ppf(.975,df)*se
 comparisons[b]={'reference':ref,'delta_pp':100*delta,'welch_ci95_pp':[100*(delta-half),100*(delta+half)],'df':df,'n':[len(x),len(y)]}
cycles={};solve_h=0
for b,cc in groups.items():
 durations=[];states=collections.Counter();nrequest=0;ncard=sum(c['cards'] for c in cc)
 for c in cc:
  root=Path(c['result_dir']);q=root/'task/.wma';t=(root/'time_taken.txt').read_text().strip().split(':');solve_h+=int(t[0])+int(t[1])/60+float(t[2])/3600
  for f in (q/'processed').glob('*.json'):
   req=json.loads(f.read_text());rp=q/'responses'/f.name
   if not rp.exists():continue
   rsp=json.loads(rp.read_text());nrequest+=1;states[rsp.get('state')]+=1
   if req.get('created_at') and rsp.get('completed_at'):
    sec=(datetime.datetime.fromisoformat(rsp['completed_at'].replace('Z','+00:00'))-datetime.datetime.fromisoformat(req['created_at'])).total_seconds();durations.append(sec/60)
 cycles[b]={'cells':len(cc),'cards':ncard,'request_response_pairs':nrequest,'states':dict(states),'lifecycle_total_min':sum(durations),'lifecycle_mean_min_per_cell':sum(durations)/len(cc),'lifecycle_mean_min_per_request':statistics.mean(durations) if durations else None}
(p/'comparisons.json').write_text(json.dumps(comparisons,indent=2));(p/'cycles.json').write_text(json.dumps(cycles,indent=2))
print('comparisons',json.dumps(comparisons,indent=2));print('cycles',json.dumps({b:r for b,r in cycles.items() if 'r02-' in b},indent=2));print('solve_hours64',solve_h)
fig,axs=plt.subplots(1,2,figsize=(12,5.8),gridspec_kw={'width_ratios':[1,2.6]},sharey=True)
blue='#2459A9';orange='#C77534'
sets=[[(batch('r01-ctl-x8-v1'),'Control',orange),(batch('r01-wma-x8-v1'),'WMA v0.2',blue)],[(ct,'Control',orange),(bl,'WMA v0.2',blue),(batch('r02-a-format-floor-x4-v2'),'A',blue),(batch('r02-ab-format-floor-width-x4-v2'),'A+B',blue),(batch('r02-c-probe-before-fail-x4-v2'),'C',blue),(batch('r02-d-checkpoint-precondition-x4-v2'),'D*',blue)]]
for ax,items,title in zip(axs,sets,['Original asynchronous cohort','Blocking review: completed cells only']):
 for i,(b,label,color) in enumerate(items):
  vals=np.array([c['accuracy']*100 for c in groups[b]]);jitter=np.linspace(-.13,.13,len(vals));ax.scatter(i+jitter,vals,color=color,s=38,alpha=.72,zorder=3)
  ax.errorbar(i,vals.mean(),yerr=vals.std(ddof=1),fmt='D',color='#1C2739',capsize=6,markersize=6,elinewidth=1.7,zorder=4)
  ax.text(i,62.4,f'{vals.mean():.2f}%\nn={len(vals)}',ha='center',va='bottom',fontsize=9,color='#344054')
 ax.set_xticks(range(len(items)),[x[1] for x in items]);ax.set_title(title,fontsize=12,pad=14);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.set_ylim(61.5,86)
axs[0].set_ylabel('Official GSM8K accuracy (%)',fontsize=11)
fig.suptitle('WMA evolution: score variation is larger than most observed arm differences',fontsize=14,y=.98)
fig.text(.06,.02,'Dots = individual runs. Diamond = mean; bars = sample SD (not confidence intervals). D*: 3/4 completed.\nSnapshot: 2026-09-04 05:52 UTC. Cohorts must be analyzed separately; access-scope gates remain unresolved.',fontsize=9,color='#475467')
fig.tight_layout(rect=[0,.11,1,.94]);fig.savefig(p/'wma-scores.png',dpi=180);fig.savefig(p/'wma-scores.svg')
