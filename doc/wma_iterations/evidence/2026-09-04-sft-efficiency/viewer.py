import json
from pathlib import Path

P=Path(__file__).resolve().parent
d=json.loads((P/'efficiency.json').read_text())
data=[]
for c in d['cells']:
    data.append({k:c[k] for k in ['cell','cohort','variant','accuracy','completed_supervised','completed_sft','completed_rft','aborted_full_intent_known','training_card_wall_h','manual_heldout_exposure','events','completed_supervised_stages','completed_sft_stages','trace_path']})
html=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WMA：按完整训练次数比较效率</title>
<style>
:root{font-family:system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;color:#1b2a3d;background:#f3f6fa}body{max-width:1320px;margin:0 auto;padding:28px}h1{font-size:27px;margin:0 0 12px}h2{font-size:19px;margin:0 0 14px}.muted{color:#53657b;font-size:13px;line-height:1.7}select{padding:8px;border:1px solid #ccd6e2;border-radius:6px;background:white;color:#18283b;max-width:100%}.bar{display:flex;flex-wrap:wrap;gap:16px;align-items:center;margin:20px 0}.panel{background:white;border:1px solid #dae2ec;border-radius:10px;padding:20px;margin:18px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}svg{width:100%;height:auto}.scroll{overflow:auto;max-height:520px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #e1e7ef;text-align:left;white-space:nowrap}th{background:#f5f8fc;position:sticky;top:0}tr.pick{background:#edf4ff}tr[data-cell]{cursor:pointer}tr[data-cell]:hover{background:#f3f7fd}.tag{font-weight:650}.blue{color:#2867b0}.orange{color:#c06b32}.stat{font-size:15px;line-height:1.8}.warn{color:#926512}a{color:#2867b0}.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:13px}@media(max-width:900px){.grid{grid-template-columns:1fr}body{padding:16px}}
</style>
<h1>WMA 能否用更少次 SFT 获得更高分？</h1>
<div class="muted">64 条完成轨迹 · 数据冻结 2026-09-04 05:52:48 UTC · 点击轨迹或散点查看逐卡变化。<br>完整训练指正式优化 schedule 已跑完；最终保存失败仍计数，smoke 与中途失败单列。RFT 也是监督拟合，因此默认计入；可切换原卡 SFT 标签口径。</div>
<div class="bar"><label>批次 <select id="cohort"></select></label><label>横轴 <select id="axis"><option value="completed_supervised">完整监督训练（SFT＋RFT）</option><option value="completed_sft">仅原卡标为 SFT</option></select></label><label>曲线题量 <select id="ruler"><option value="150">n = 150</option><option value="500">n = 500</option><option value="1319">n = 1319</option></select></label></div>
<div class="panel"><div id="summary" class="stat"></div></div>
<div class="grid"><div class="panel"><h2>最终官方分数 vs 完整训练次数</h2><svg id="scatter" viewBox="0 0 600 340"></svg><div class="muted">每点一条轨迹，纵轴均为最终官方1319题分数。散点横坐标为整数次数；均值不构成因果效应。</div></div><div class="panel"><h2 id="curveTitle">选中轨迹的分数演化</h2><svg id="curve" viewBox="0 0 600 340"></svg><div class="muted">圆点=本卡原始读数；折线=按当前题量已测最高分；黑菱形=最终官方分，仅n1319显示。后测的旧checkpoint不倒填到早期。变更解码、并发和重复选模均可能影响分数。</div></div></div>
<div class="panel"><h2>每条轨迹</h2><div class="legend" id="legend"></div><div class="scroll"><table><thead><tr><th>traj</th><th>组</th><th>SFT＋RFT</th><th>未完成正式尝试 ≥</th><th>已测最高分演化（按题量）</th><th>最终官方</th></tr></thead><tbody id="cells"></tbody></table></div><div class="muted">† 表示该训练阶段没有新增所选题量的读数，只沿用此前观测；— 表示缺测。停止后的轨迹不向更高次数外推。</div></div>
<div class="panel"><h2 id="detailTitle">逐卡证据</h2><div id="detailMeta" class="muted"></div><div class="scroll"><table><thead><tr><th>card</th><th>类型 / 状态</th><th>本卡完整训练</th><th>累计完整训练</th><th>原始分数（题量）</th><th>决定</th></tr></thead><tbody id="events"></tbody></table></div><p class="muted" id="provenance"></p></div>
<div class="panel muted">读图边界：训练次数不等于相同GPU小时、数据量或token预算；多次训练可能属于更难的轨迹。R1旧版与R2阻塞版分别比较。高次数只剩部分轨迹，不能当作原对照组。历史次数经卡片普查和重点重试核验，是已确认下界。w09r03有独立测试数据访问审计标记；曲线保留用于追溯，不作为候选晋升依据。完整原始表见同目录 <a href="all-trajectories.md">all-trajectories.md</a>、<a href="trajectories.csv">CSV</a>、<a href="all-trajectories.pdf">全部曲线PDF</a>。</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
const colors={'Control':'#C06B32','v0.2':'#2867B0','A':'#8295AB','A+B':'#8C62AC','C':'#268674','D':'#B58D24'};
let selected='w10r01';const ns='http://www.w3.org/2000/svg';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=x=>(100*x).toFixed(2)+'%';const avg=xs=>xs.reduce((a,b)=>a+b,0)/xs.length;
const cohorts=[...new Set(DATA.map(c=>c.cohort))];$('cohort').innerHTML=cohorts.map(x=>`<option>${esc(x)}</option>`).join('');$('cohort').value='R2 blocking';
function el(svg,tag,attrs,text){let e=document.createElementNS(ns,tag);for(let k in attrs)e.setAttribute(k,attrs[k]);if(text!==undefined)e.textContent=text;svg.appendChild(e);return e}
function axes(svg,xmax,ymin,ymax,xlabel){svg.innerHTML='';let x=k=>55+k*(510/Math.max(1,xmax)),y=v=>285-(v-ymin)*240/(ymax-ymin);for(let v=Math.ceil(ymin/10)*10;v<=ymax;v+=10){el(svg,'line',{x1:55,x2:570,y1:y(v),y2:y(v),stroke:'#e4eaf1'});el(svg,'text',{x:45,y:y(v)+4,'text-anchor':'end','font-size':11,fill:'#57677b'},v+'%')};for(let k=0;k<=xmax;k++){el(svg,'text',{x:x(k),y:305,'text-anchor':'middle','font-size':11,fill:'#57677b'},k)}el(svg,'text',{x:315,y:333,'text-anchor':'middle','font-size':12,fill:'#34465f'},xlabel);return {x,y}}
function choose(id){selected=id;render()}
function render(){let co=$('cohort').value,axis=$('axis').value,n=+$('ruler').value,cs=DATA.filter(c=>c.cohort===co).sort((a,b)=>a.variant.localeCompare(b.variant)||a.cell.localeCompare(b.cell));if(!cs.some(c=>c.cell===selected))selected=cs[0].cell;let c=cs.find(c=>c.cell===selected),variants=[...new Set(cs.map(c=>c.variant))];
 $('legend').innerHTML=variants.map(v=>`<span style="color:${colors[v]}">● ${esc(v)}</span>`).join('');
 $('summary').innerHTML=variants.map(v=>{let g=cs.filter(c=>c.variant===v);return `<span style="color:${colors[v]}"><b>${esc(v)}</b></span>：${g.length} 条，平均 ${avg(g.map(c=>c[axis])).toFixed(3)} 次，最终均分 <b>${pct(avg(g.map(c=>c.accuracy)))}</b>`}).join('<br>');
 const maxX=Math.max(...cs.map(c=>c[axis]));let s=$('scatter'),a=axes(s,maxX+1,60,85,'完整训练次数');
 for(let t of cs){let pt=el(s,'circle',{cx:a.x(t[axis]),cy:a.y(t.accuracy*100),r:t.cell===selected?7:5,fill:colors[t.variant],stroke:t.cell===selected?'#16283e':'white','stroke-width':t.cell===selected?2:1,style:'cursor:pointer'});let title=document.createElementNS(ns,'title');title.textContent=`${t.cell} / ${t.variant}: ${t[axis]} 次，${pct(t.accuracy)}`;pt.appendChild(title);pt.addEventListener('click',()=>choose(t.cell))}
 $('curveTitle').textContent=`${c.cell} / ${c.variant} · n=${n}`;s=$('curve');a=axes(s,c[axis]+1,0,90,axis==='completed_supervised'?'SFT＋RFT 完整次数':'仅 SFT 次数（RFT另有成本）');
 let ss=c[axis+'_stages'],pts=ss.filter(st=>st.best_observed[n]!==undefined),path=pts.map((st,i)=>(i?'L':'M')+a.x(st.k)+','+a.y(st.best_observed[n]*100)).join(' ');el(s,'path',{d:path,fill:'none',stroke:colors[c.variant],'stroke-width':2});
 for(let e of c.events){for(let m of e.measurements){if(m.n!==n||m.evaluation_scope!=='declared_official')continue;let circle=el(s,'circle',{cx:a.x(e[axis]),cy:a.y(m.score*100),r:4,fill:colors[c.variant],opacity:.65});let tt=document.createElementNS(ns,'title');tt.textContent=`${e.card} ${e.family}: ${pct(m.score)}, n=${n}`;circle.appendChild(tt)}}
 if(n===1319){let x=a.x(c[axis]),y=a.y(c.accuracy*100);el(s,'path',{d:`M${x},${y-7}L${x+7},${y}L${x},${y+7}L${x-7},${y}Z`,fill:'#152940'})}
 $('cells').innerHTML=cs.map(t=>{let chain=t[axis+'_stages'].slice(1).map(st=>st.best_observed[n]===undefined?'—':(st.best_observed[n]*100).toFixed(2)+(st.fresh_ns.includes(n)?'':'†')).join(' → ');return `<tr data-cell="${esc(t.cell)}" class="${t.cell===selected?'pick':''}"><td class="tag">${esc(t.cell)}${t.manual_heldout_exposure?' ⚑':''}</td><td style="color:${colors[t.variant]}">${esc(t.variant)}</td><td>${t.completed_sft}+${t.completed_rft}=${t.completed_supervised}</td><td>${t.aborted_full_intent_known}</td><td>${chain}</td><td>${pct(t.accuracy)}</td></tr>`}).join('');$('cells').querySelectorAll('tr').forEach(tr=>tr.addEventListener('click',()=>choose(tr.dataset.cell)));
 $('detailTitle').textContent=c.cell+' · 逐卡原始分数';$('detailMeta').textContent=`训练卡记录wall合计 ${c.training_card_wall_h.toFixed(2)} h（不是纯GPU时间）。同一卡内多次完整重训合并计数，精确审计说明见CSV/JSON。`;
 $('events').innerHTML=c.events.map(e=>`<tr><td>${esc(e.card)}</td><td>${esc(e.family)} / ${esc(e.execution)}</td><td>${e.completed_fits}${e.audit_note?' *':''}</td><td>${e.completed_supervised}</td><td>${e.measurements.map(m=>`${pct(m.score)} (n=${m.n})`).join('; ')||'—'}</td><td>${esc(e.decision)}</td></tr>`).join('');$('provenance').textContent='原始轨迹：'+c.trace_path;
}
['cohort','axis','ruler'].forEach(id=>$(id).addEventListener('change',render));render();
</script></html>'''
(P/'viewer.html').write_text(html.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')))
print('wrote viewer.html')
