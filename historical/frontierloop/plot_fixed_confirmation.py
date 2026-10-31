"""Existing N76 data only: descriptive two-readout mean/range resource planes."""
from pathlib import Path
from io import BytesIO
import numpy as np
from .paths import ROOT
from .io import load,save,write,sha256
from .fixed_confirmation_execution import LOCAL,capacity

OUT=ROOT/'analysis/N76-figures-v2'
NAMES={'ACTIVE_GROUP_FIXED_LIBRARY':'Active group (fixed)','B4_FIXED_SELECTED_ENSEMBLE':'B4 bank (fixed)',
    'C5_C_LUCJ_COMPRESSED_L1':'LUCJ L1','C5_C_LUCJ_COMPRESSED_L2':'LUCJ L2','C5_C_LUCJ_COMPRESSED_L4':'LUCJ L4',
    'SQDRIFT_N5_FIXED30':'SqDRIFT N5','SQDRIFT_N20_FIXED30':'SqDRIFT N20'}
COLORS=['#D55E00','#CC79A7','#0072B2','#009E73','#56B4E9','#E69F00','#444444']
CASES={'NAPH20-BLOCK-BOYS':'Naphthalene / 20q','N2-32-R1p2-P2':'N2 1.2 A / 32q','N2-32-R2p4-P2':'N2 2.4 A / 32q','FE2S2-40-P2':'Fe2S2 / 40q (SCI reference unverified)'}

def nondominated(points):
    """Indices on an x/y minimization plane, not a total-cost success rule."""
    return [i for i,(x,y) in enumerate(points) if not any(a<=x and b<=y and (a<x or b<y) for j,(a,b) in enumerate(points) if j!=i)]

def collect(p):
    rows=[]
    for case in CASES:
        source=[r for r in p['rows'] if r['case']==case and r['context']=='R32' and r['status']=='COMPLETED']
        hci=[r['energy_Ha'] for r in source if r['method']=='CLASSICAL_HF_HCI'];assert len(hci)==2 and hci[0]==hci[1]
        for method in NAMES:
            rr=[r for r in source if r['method']==method]
            if not rr:continue
            assert len(rr)==2 and all(r['independent_parameter_rank'] is None for r in rr)
            gaps=[1000*(r['energy_Ha']-hci[0]) for r in rr]
            rows.append(dict(case=case,method=method,energy_gap_to_HCI_mHa=gaps,energy_gap_mean_mHa=float(np.mean(gaps)),
                Recovery_pct=[r['Recovery_pct'] for r in rr],Recovery_mean_pct=float(np.mean([r['Recovery_pct'] for r in rr])),
                actual_R=[r['actual_R'] for r in rr],actual_D=[r['actual_D'] for r in rr],
                **{k:rr[0][k] for k in ['CNOT','CNOT_Depth','Total_Depth']},
                campaign_model_samples=rr[0]['campaign_model_samples_including_pilot']))
    assert len(rows)==26 and all(r['CNOT']>0 for r in rows)
    return rows

def run():
    capacity();file=ROOT/'analysis/N76-fixed-confirmation-summary-v2.json';p=load(file);rows=collect(p)
    if (OUT/'manifest.json').exists():raise RuntimeError('Existing figures; no replace')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator,NullFormatter,FuncFormatter
    def clean_log_ticks(ax):
        ax.xaxis.set_major_locator(LogLocator(base=10,subs=(1,5),numticks=6))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    keys=['CNOT','CNOT_Depth','Total_Depth'];labels=['Logical CNOT count','Logical CNOT depth','Logical total depth']
    fig,axes=plt.subplots(4,3,figsize=(13,12),layout='constrained');planes=[]
    for i,case in enumerate(CASES):
        rr=[r for r in rows if r['case']==case]
        for j,(key,label) in enumerate(zip(keys,labels)):
            ax=axes[i,j];points=[(r[key],r['energy_gap_mean_mHa']) for r in rr];front=nondominated(points)
            for r in rr:
                idx=list(NAMES).index(r['method']);y=r['energy_gap_mean_mHa'];lo,hi=min(r['energy_gap_to_HCI_mHa']),max(r['energy_gap_to_HCI_mHa'])
                ax.errorbar(r[key],y,yerr=np.array([[y-lo],[hi-y]]),fmt='o',color=COLORS[idx],capsize=3,markersize=5,linewidth=1)
                if any(x!=32 for x in r['actual_R']):ax.annotate('R'+('/'.join(map(str,r['actual_R'])) if len(set(r['actual_R']))>1 else str(r['actual_R'][0])),(r[key],y),xytext=(4,4),textcoords='offset points',fontsize=6)
            ff=sorted((points[k] for k in front))
            if ff:ax.plot([x for x,y in ff],[y for x,y in ff],':',color='#888888',linewidth=1,zorder=0)
            planes.append(dict(case=case,x_metric=key,frontier_methods=[rr[k]['method'] for k in front],scope='DESCRIPTIVE_MEAN_ENERGY_RESOURCE_PLANE_NOT_TOTAL_WORK_FRONTIER'))
            ax.axhline(0,color='black',linestyle='--',linewidth=.8);ax.set_xscale('log');ax.set_yscale('symlog',linthresh=1);ax.set_ylim(bottom=-.05)
            clean_log_ticks(ax);ax.set_xlabel(label+' (log scale)');ax.set_title(CASES[case],fontsize=10)
            if j==0:ax.set_ylabel('Energy above HF-HCI / mHa\n(lower is better; symlog)')
            ax.grid(alpha=.18)
    handles=[Line2D([],[],marker='o',linestyle='',color=c,label=NAMES[m]) for m,c in zip(NAMES,COLORS)]
    fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('Fixed-source confirmation: R cap 32, two new readouts per quantum source\nBars: observed min-max, NOT confidence intervals; dotted line: descriptive mean-plane frontier',fontsize=11)
    artifacts={}
    for ext in ['png','svg']:
        buffer=BytesIO();fig.savefig(buffer,format=ext,dpi=180);blob=buffer.getvalue();capacity(len(blob));name='energy-resource-planes.'+ext;write(OUT/name,blob);artifacts[name]=sha256(blob)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,case in zip(axes.flat,CASES):
        rr=[r for r in rows if r['case']==case]
        for r in rr:
            idx=list(NAMES).index(r['method']);y=r['Recovery_mean_pct'];lo,hi=min(r['Recovery_pct']),max(r['Recovery_pct'])
            ax.errorbar(r['CNOT'],y,yerr=np.array([[y-lo],[hi-y]]),fmt='o',color=COLORS[idx],capsize=3,markersize=5)
        hci=next(r for r in p['rows'] if r['case']==case and r['context']=='R32' and r['method']=='CLASSICAL_HF_HCI')
        ax.axhline(hci['Recovery_pct'],color='black',linestyle='--',linewidth=.8,label='HF-HCI reference')
        ax.set_xscale('log');clean_log_ticks(ax);ax.set_xlabel('Logical CNOT count (log scale)');ax.set_ylabel('Recovery / % (higher is better)');ax.set_title(CASES[case]);ax.grid(alpha=.18)
    fig.legend(handles=handles+[Line2D([],[],color='black',linestyle='--',label='Classical HF-HCI')],loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('Same stored reference and R cap 32; Fe Recovery is provisional\nClassical HF-HCI is a horizontal reference, NOT a zero-CNOT quantum state',fontsize=11)
    for ext in ['png','svg']:
        buffer=BytesIO();fig.savefig(buffer,format=ext,dpi=180);blob=buffer.getvalue();capacity(len(blob));name='recovery-cnot.'+ext;write(OUT/name,blob);artifacts[name]=sha256(blob)
    plt.close(fig)
    save(OUT/'manifest.json',dict(input_file=str(file),input_SHA256=sha256(file.read_bytes()),plotter_SHA256=sha256(Path(__file__).read_bytes()),
        rows=rows,planes=planes,artifacts=artifacts,known_missing_source_cells=2,actual_new_model_samples=0,actual_new_CI_roots=0,
        unknown_independent_parameters_not_plotted=True,actual_dimensions_may_differ=True,new_source_search_authorized=False,scientific_goal_achieved=False))
    caption='''# N76：取得済みデータのfrontier図（追加実験なし）

入力は修正版v2の全320欄のみ。新しいShot・CI・GPU処理は0。2反復の平均と観測min–maxを表示し、エラーバーを信頼区間と呼ばない。

- `energy-resource-planes.png` / `.svg`：4条件×CNOT・CX Depth・Total Depth。縦軸は同じ最終CI条件の古典HF-HCIからのenergy差mHaで、真の基底状態誤差ではない。破線0は古典参照であり、0 CNOTの量子状態ではない。点線は量子sourceの平均energy–資源平面の非支配点をつなぐ参考線。完全work frontierではない。
- `recovery-cnot.png` / `.svg`：同じ保存参照に対するRecoveryとCNOT。FeのSCI参照は未検証で、Recoveryも暫定値。

全点はR上限32。同じ上限でも実R/Dは異なる。energy図でR32以外の点を注記した。ナフタレンLUCJ L2/L4の欠測は点として捏造しない。campaignの量子換算費用は各200,000でB4旧構築pilotを一回含むが、完全cold classical workは揃っていない。独立Parametersはunknownなので数値の軸を作らない。

source固定後の新しい読出し2反復であり、独立したsource構築の再現ではない。保存LUCJ fitの未収束段階、最高原著予算未再現、cross-framework compiler未完の制約は最終結論のまま。この図を理由に最強論文への勝利を認定しない。
'''
    write(OUT/'README_JA.md',caption.encode());print('N76_EXISTING_DATA_ONLY_FIGURES_WRITTEN',artifacts,flush=True)

if __name__=='__main__':run()
