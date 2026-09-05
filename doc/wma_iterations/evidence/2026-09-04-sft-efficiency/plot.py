import json
import math
import os
import statistics
from pathlib import Path

os.environ['MPLCONFIGDIR'] = '/tmp/wma-efficiency-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

P = Path(__file__).resolve().parent
D = json.loads((P / 'efficiency.json').read_text())
COLORS = {'Control':'#C06B32','v0.2':'#2867B0','A':'#8295AB','A+B':'#8C62AC','C':'#268674','D':'#B58D24'}
RULERS = {150:'#765BB1',500:'#2682A2',1319:'#379364'}


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)


fig, axs = plt.subplots(2,2,figsize=(13.5,10.5))
for j, co in enumerate(['R1 core','R2 blocking']):
    cc = [c for c in D['cells'] if c['cohort']==co and c['variant'] in ['Control','v0.2']]
    ax = axs[0,j]
    for variant, offset in [('Control',-.07),('v0.2',.07)]:
        group=[c for c in cc if c['variant']==variant]
        xx=[c['completed_supervised']+offset for c in group]; yy=[100*c['accuracy'] for c in group]
        ax.scatter(xx,yy,color=COLORS[variant],s=58,alpha=.8,label=f'{variant} (n={len(group)})')
        for c,x,y in zip(group,xx,yy):
            ax.annotate(c['cell'],(x,y),xytext=(4,3),textcoords='offset points',fontsize=7,color=COLORS[variant])
        ax.scatter([np.mean([c["completed_supervised"] for c in group])],[np.mean(yy)],marker='X',s=145,color=COLORS[variant],edgecolor='white',zorder=4)
    ax.set_title(co+': final official score vs training count',fontsize=12)
    ax.set_xlabel('Completed supervised fits (SFT + RFT)')
    ax.set_ylabel('Final official GSM8K accuracy (%)')
    ax.set_xticks(range(1,6));ax.set_xlim(1.5,5.5);ax.set_ylim(64,84);clean(ax);ax.legend(loc='lower right',fontsize=9)
    ax=axs[1,j]
    for variant in ['Control','v0.2']:
        for c in cc:
            if c['variant']!=variant:continue
            ss=c['completed_supervised_stages'][1:]
            pts=[(s['k'],s['best_observed'].get('150')) for s in ss]
            pts=[(x,y) for x,y in pts if y is not None]
            if pts:ax.plot([x for x,y in pts],[100*y for x,y in pts],color=COLORS[variant],alpha=.18,lw=1)
        rows=[r for r in D['curves'] if r['cohort']==co and r['variant']==variant and r['axis']=='completed_supervised' and r['eval_n']==150 and r['n']]
        ax.plot([r['k'] for r in rows],[100*r['mean'] for r in rows],'-o',color=COLORS[variant],lw=2.5,label=variant)
        for r in rows:
            ax.annotate('n='+str(r['n']),(r['k'],100*r['mean']),xytext=(0,10 if variant=='Control' else -15),textcoords='offset points',ha='center',fontsize=8,color=COLORS[variant])
    ax.set_title(co+': best recorded n=150 score by fit count',fontsize=12)
    ax.set_xlabel('Completed supervised fits (SFT + RFT)')
    ax.set_ylabel('Best observed accuracy (%)')
    ax.set_xticks(range(1,6));ax.set_xlim(.75,5.3);ax.set_ylim(38,86);clean(ax);ax.legend(fontsize=9,loc='lower right')
fig.suptitle('Does WMA reach better scores with fewer completed training runs?',fontsize=16,y=.99)
fig.text(.055,.025,'Top: each dot is one trajectory (slight horizontal jitter); X = group mean. Bottom: same-n descriptive envelopes, not fixed-budget causal effects.\nR1 excludes c01r05 from n=150 curves (it used n=200/600). Curves stop when a trajectory ends; n changes at higher counts.\nRFT fits and completed schedules with failed final saves count. Explicit smokes do not. Training sizes and evaluation settings vary.',fontsize=9,color='#4A5568')
fig.tight_layout(rect=[0,.105,1,.965]);fig.savefig(P/'efficiency-comparison.png',dpi=180);fig.savefig(P/'efficiency-comparison.svg');plt.close(fig)

groups={}
for c in D['cells']:groups.setdefault(c['cohort'],[]).append(c)
with PdfPages(P/'all-trajectories.pdf') as pdf:
    for co, cc in groups.items():
        cc=sorted(cc,key=lambda c:(c['variant'],c['cell']))
        for page,start in enumerate(range(0,len(cc),16)):
            chunk=cc[start:start+16]
            fig,axes=plt.subplots(4,4,figsize=(15,12),squeeze=False)
            for ax,c in zip(axes.flat,chunk):
                for n,col in RULERS.items():
                    stages=c['completed_supervised_stages'];pts=[(s['k'],s['best_observed'].get(str(n)),n in s['fresh_ns']) for s in stages]
                    pts=[p for p in pts if p[1] is not None]
                    if not pts:continue
                    ax.step([x for x,y,f in pts],[y*100 for x,y,f in pts],where='post',color=col,lw=1.4)
                    for x,y,f in pts:
                        ax.scatter([x],[100*y],s=24,facecolor=col if f else 'white',edgecolor=col,zorder=3)
                        if x>0 and f:ax.annotate(f'{100*y:.1f}',(x,100*y),xytext=(1,4),textcoords='offset points',fontsize=6,color=col)
                ax.scatter([c['completed_supervised']],[c['accuracy']*100],marker='D',color='#17243A',s=28,zorder=4)
                ax.set_title(f"{c['cell']}  {c['variant']} | final {100*c['accuracy']:.2f}%\n{c['completed_sft']} SFT + {c['completed_rft']} RFT; partial >= {c['aborted_full_intent_known']}",fontsize=9)
                ax.set_xticks(range(c['completed_supervised']+1));ax.set_xlim(-.2,c['completed_supervised']+.35);ax.set_ylim(0,90);clean(ax)
                ax.tick_params(labelsize=7)
            for ax in list(axes.flat)[len(chunk):]:ax.set_visible(False)
            fig.suptitle(co+' | every trajectory, aligned on completed supervised fits',fontsize=16)
            fig.text(.06,.02,'Purple=n150; blue=n500; green=n1319; black diamond=final official score. Hollow circle=old observation carried forward.\nEach ruler is separate. No score is reassigned back to the training run that created its model. X counts SFT+RFT; smokes excluded.\nScores are observed maxima, not necessarily the selected incumbent. Other evaluation sizes are available in the CSV / interactive viewer.',fontsize=9)
            fig.tight_layout(rect=[0,.075,1,.95]);pdf.savefig(fig)
            if co=='R2 blocking':fig.savefig(P/f'r2-trajectories-{page+1}.png',dpi=140)
            plt.close(fig)

# Every cell has a readable row, including per-ruler stage values and all intermediate card events.
lines=['**每条轨迹：完整训练次数与分数演化**','',
       '数据冻结：2026-09-04 05:52:48 UTC，64条完成轨迹。主要横轴=完成正式优化schedule的SFT+RFT次数；包含最终保存失败。',
       '表中序列为完成第1、2、…次训练后、下一次完整训练前，已经记录的最高同题量分数。不同题量分列；†表示该阶段没有新增该题量读数，仅沿用旧观测。',
       '最终官方分与中间评估独立列出。不得把n150高分当最终1319题分；不得从存活到高k的少数轨迹推原始随机对照效果。','']
def chain(c,n):
    vals=[]
    for s in c['completed_supervised_stages'][1:]:
        v=s['best_observed'].get(str(n));vals.append('—' if v is None else f'{100*v:.2f}'+('†' if n not in s['fresh_ns'] else ''))
    return ' → '.join(vals)
for co,cc in groups.items():
    lines += [f'**{co}**','', '| traj | 组 | SFT+RFT | 未完成正式尝试≥ | n150演化 % | n500演化 % | n1319演化 % | 最终官方 % |', '|---|---|---:|---:|---|---|---|---:|']
    for c in sorted(cc,key=lambda x:(x['variant'],x['cell'])):
        lines.append(f"| [{c['cell']}]({c['trace_path']}) | {c['variant']} | {c['completed_sft']}+{c['completed_rft']}={c['completed_supervised']} | {c['aborted_full_intent_known']} | {chain(c,150)} | {chain(c,500)} | {chain(c,1319)} | {100*c['accuracy']:.2f} |")
    lines += ['']
lines+=['**逐卡原始分数**','', '这是观察顺序；同一次训练的多个checkpoint、不同decoder与重复读数不会多计训练次数。','']
for c in sorted(D['cells'],key=lambda c:(c['cohort'],c['variant'],c['cell'])):
    lines += [f"**{c['cell']} / {c['cohort']} / {c['variant']}**",'', '| card | 类别 / 执行状态 | 本卡完整训练数 | 累计完整训练数 | 原始accuracy（题量） |', '|---|---|---:|---:|---|']
    for e in c['events']:
        ss='; '.join(f"{100*m['score']:.2f}% (n={m['n']})" for m in e['measurements']) or '—'
        lines.append(f"| [{e['card']}]({e['source']}) | {e['family']} / {e['execution']} | {e['completed_fits']} | {e['completed_supervised']} | {ss} |")
    lines+=['']
(P/'all-trajectories.md').write_text('\n'.join(lines)+'\n')
print('wrote comparison figure, all-trajectories.pdf, all-trajectories.md')
