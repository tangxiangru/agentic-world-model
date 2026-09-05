from pathlib import Path
import csv,json,os,statistics
os.environ['MPLCONFIGDIR']='/tmp/wma-distribution-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'2026-09-04-sft-efficiency/efficiency.json'
data=json.loads(SOURCE.read_text())
BG='#f8fafc';INK='#172d43';MUTED='#536a83';BLUE='#648dbf';GREEN='#459c8c';GOLD='#d4923f';RED='#ae524b'
rows=[
 ('R1 core','v0.2','R1 original / WMA v0.2'),
 ('R1 core','Control','R1 original / no WMA'),
 ('R1 B-v2','v0.2','R1 extension B-v2 / WMA'),
 ('R1 B-v2','Control','R1 extension B-v2 / no WMA'),
 ('R1 C-v2','v0.2','R1 extension C-v2 / WMA'),
 ('R1 C-v2','Control','R1 extension C-v2 / no WMA'),
 ('R1 B-v3','v0.2','R1 extension B-v3 / WMA'),
 ('R1 B-v3','Control','R1 extension B-v3 / no WMA'),
 ('R2 blocking','v0.2','R2 blocking / WMA v0.2'),
 ('R2 blocking','Control','R2 blocking / no WMA'),
 ('R2 blocking','A','A / format-floor'),
 ('R2 blocking','A+B','A+B / format-floor + width'),
 ('R2 blocking','C','C / probe-before-fail'),
 ('R2 blocking','D','D / checkpoint plan *'),
]
exports=[]
for co,v,label in rows:
 cs=[c for c in data['cells'] if c['cohort']==co and c['variant']==v]
 scores=[c['accuracy']*100 for c in cs]
 exports.append(dict(cohort=co,variant=v,label=label,n=len(cs),mean_pct=statistics.mean(scores),min_pct=min(scores),max_pct=max(scores),mean_completed_fits=statistics.mean(c['completed_supervised'] for c in cs),cells=[c['cell'] for c in cs],scores_pct=scores))
assert sum(r['n'] for r in exports)==64
(HERE/'cohorts.json').write_text(json.dumps(exports,indent=2)+'\n')
with (HERE/'cohorts.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(exports[0]));w.writeheader();w.writerows(exports)

def plot(selected,stem,compact=False):
 count=sum(r['n'] for r in selected)
 h=9.1 if compact else 13.6
 fig=plt.figure(figsize=(19,h),facecolor=BG)
 ax=fig.add_axes([.345,.21 if compact else .165,.43,.64 if compact else .72],facecolor=BG)
 yy=[];y=0
 for i,r in enumerate(selected):
  if i and r['cohort']!=selected[i-1]['cohort']:y+=.25
  yy.append(y);y+=1
 ax.set_ylim(y-.55,-.55)
 ax.set_xlim(63,84)
 ax.set_xticks([65,70,75,80])
 ax.grid(axis='x',color='#dce3ec',lw=1)
 ax.set_axisbelow(True)
 for spine in ax.spines.values():spine.set_visible(False)
 ax.tick_params(axis='both',length=0,labelsize=15,pad=12)
 ax.set_yticks(yy,[r['label'] for r in selected],fontsize=15)
 ax.set_xlabel('Official GSM8K accuracy (%)',fontsize=17,labelpad=16,color=INK)
 for i,(r,y0) in enumerate(zip(selected,yy)):
  color=GREEN if r['variant']=='Control' else BLUE if r['variant']=='v0.2' else GOLD
  values=r['scores_pct'];count_r=len(values)
  ax.hlines(y0,min(values),max(values),color=color,alpha=.30,lw=3,zorder=1)
  order=np.argsort(values)
  jitter=np.zeros(count_r)
  sequence=[-.085,.075,-.035,.12,-.12,.025,.10,-.06]
  for rank,idx in enumerate(order):jitter[idx]=sequence[rank%len(sequence)] if count_r>1 else 0
  for idx,(cell,score) in enumerate(zip(r['cells'],values)):
   flagged=cell=='w09r03'
   ax.scatter(score,y0+jitter[idx],s=62 if compact else 55,color=color if not flagged else BG,edgecolor=RED if flagged else BG,linewidth=1.7 if flagged else .8,zorder=3)
  ax.scatter(r['mean_pct'],y0,marker='D',s=105,color=INK,zorder=4)
  ax.text(1.02,y0,f"{r['mean_pct']:.2f}%",transform=ax.get_yaxis_transform(),va='center',fontsize=15,color=INK,clip_on=False)
  ax.text(1.17,y0,f"n={r['n']}",transform=ax.get_yaxis_transform(),va='center',fontsize=15,color=INK,clip_on=False)
  ax.text(1.32,y0,f"{r['mean_completed_fits']:.2f}",transform=ax.get_yaxis_transform(),va='center',fontsize=15,color=INK,clip_on=False)
  # Mark the best candidate trajectory, keeping leader line short and in its row band.
  if r['variant']=='C':
   j=r['cells'].index('w08r04');s=values[j]
   ax.annotate(f'w08r04: {s:.2f}%',xy=(s,y0+jitter[j]),xytext=(s-3.2,y0-.34),fontsize=12,color='#945713',arrowprops=dict(arrowstyle='-',color='#b77927',lw=1.3),ha='left')
 # Table headers share the data axis y transform.
 for xx,label in [(1.02,'Mean'),(1.17,'Runs'),(1.32,'Mean fits')]:
  ax.text(xx,1.025,label,transform=ax.transAxes,ha='left',va='bottom',fontsize=12,color=MUTED)
 if compact:
  title=f'WMA evolve: key comparisons ({count} of 64 completed runs)'
 else:
  title='WMA evolve: 64 PTB-validated, original judge-clean runs'
 fig.text(.03,.955,title,fontsize=25,weight='bold',color=INK,va='top')
 fig.text(.03,.906 if compact else .919,'Gemma-3-4B-PT | Opus-5 high / 1M | 10-hour session budget | Snapshot: 2026-09-04 05:52 UTC',fontsize=15,color=MUTED,va='top')
 foot=[
  'Dots = individual trajectories; diamond = cohort mean; line = observed range. Cohorts are not causal contrasts.',
  'Mean fits = completed SFT + RFT schedules; final-save failures count, explicit smokes do not. R1 counts are audited lower bounds.',
  '* D: 3/4 completed. Red outline marks w09r03 (manual data-access finding); original validator/judge status is retained.',
 ]
 if compact:foot.append('Shown: original R1 8+8, blocking baseline/control 4+4, and A/A+B/C/D (15). Historical extensions (25) are in the all-runs chart.')
 else:foot.append('Historical extensions include unmatched/sensitivity runs. E/F/G/H and incomplete/withdrawn attempts are not in this frozen result set.')
 fig.text(.03,.115 if compact else .071,'\n'.join(foot),fontsize=11.5,color=MUTED,va='top',linespacing=1.4)
 for ext in ['png','svg','pdf']:
  fig.savefig(HERE/(stem+'.'+ext),dpi=180,facecolor=BG)
 plt.close(fig)

plot(exports,'wma-score-distribution-all64')
plot([r for r in exports if r['cohort'] in ['R1 core','R2 blocking']],'wma-score-distribution-key39',compact=True)
print(HERE/'wma-score-distribution-all64.png')
