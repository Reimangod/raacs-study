"""Headless scientific scatter: same-source circuit and CI effects separated."""
import csv
import os
from .paths import ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .io import load
from .analyze import reference


def main():
    assert load(ROOT/'analysis/G5-final-report-audit.json')['joined_rows']==16
    summary=load(ROOT/'analysis/G5-whole-N20-summary.json')
    old=list(csv.DictReader((ROOT/'analysis/R2-common-CI-all.csv').open(encoding='utf-8-sig')))
    t2=list(csv.DictReader((ROOT/'analysis/T2-pooled-ranking-all.csv').open(encoding='utf-8-sig')))
    joined=list(csv.DictReader((ROOT/'analysis/G5-energy-resource-joined.csv').open(encoding='utf-8-sig')))
    labels={'NAPH20-BLOCK-BOYS':'Naphthalene, block Boys (20q)',
        'N2-32-R1p2-P2':'N2, 1.2 A (32q)','N2-32-R2p4-P2':'N2, 2.4 A (32q)',
        'FE2S2-40-P2':'Fe2S2 (40q)'}
    fig,axes=plt.subplots(2,2,figsize=(10,7.5))
    for ax,source in zip(axes.flat,summary['cases']):
        case=source['case'];_,ref=reference(case)
        def err(e):return 1000*(float(e)-ref)
        others=[r for r in old if r['case']==case and r['status']=='COMPLETED' and r['CNOT'] and float(r['CNOT'])>0]
        for family,marker,color in [('LUCJ','s','#777777'),('SQDRIFT','^','#6f8bbd'),('RAACS','x','#a87976')]:
            group=[r for r in others if family in r['method']]
            ax.scatter([float(r['CNOT']) for r in group],[err(r['energy_Ha']) for r in group],
                label='Observed '+family,marker=marker,color=color,s=30,alpha=.75)
        rest=[r for r in others if not any(k in r['method'] for k in ['LUCJ','SQDRIFT','RAACS'])]
        ax.scatter([float(r['CNOT']) for r in rest],[err(r['energy_Ha']) for r in rest],
            label='Other checked samplers',marker='.',color='#444444',s=25,alpha=.7)
        fair_global=[]
        for r in t2:
            if r['case']==case and r['status']=='COMPLETED' and r['arm']=='GLOBAL_CI_MARGINAL':
                prior=next((p for p in others if p['method']==r['method']),None)
                if prior: fair_global.append((float(prior['CNOT']),err(r['energy_Ha'])))
        ax.scatter([x for x,y in fair_global],[y for x,y in fair_global],label='Controls with same Global CI',
            marker='D',facecolors='none',edgecolors='#85579c',s=38,linewidths=.8)
        rows={r['method']:r for r in joined if r['case']==case}
        names=['N20_ORIGINAL_NATIVE_OLD_R2_FREQUENCY','N20_COMPRESSED_SAME_OLD_R2_FREQUENCY',
               'N20_STRONG_COMPILER_SAME_GLOBAL','N20_COMPRESSED_SAME_GLOBAL']
        for name,color,marker,label in zip(names,['#344e86','#d68c37','#666666','#248457'],['o','o','D','*'],
                ['N20 native + frequency','Compressed + same frequency','Strong compiler + Global','Compressed + Global']):
            row=rows[name];ax.scatter(float(row['CNOT']),err(row['energy_Ha']),c=color,marker=marker,
                s=90 if marker=='*' else 45,label=label,zorder=5,edgecolors='black',linewidths=.4)
        for start,end in [(names[0],names[1]),(names[2],names[3])]:
            a,b=rows[start],rows[end]
            ax.annotate('',xy=(float(b['CNOT']),err(b['energy_Ha'])),xytext=(float(a['CNOT']),err(a['energy_Ha'])),
                arrowprops=dict(arrowstyle='->',color='#248457',lw=1.1))
        # Show a classical energy reference as a horizontal line, not a fictitious0-CX preparation.
        classic_all=[r for r in old if r['case']==case and r['status']=='COMPLETED' and 'CISD' in r['method']]
        classic_all += [r for r in t2 if r['case']==case and r['status']=='COMPLETED' and 'CISD' in r['method']]
        if classic_all:
            best=min(classic_all,key=lambda r:float(r['energy_Ha']))
            ax.axhline(err(best['energy_Ha']),ls=':',color='#8f477c',lw=1,label='Observed CISD readout energy')
        ax.set_xscale('log');ax.set_yscale('log');ax.set_title(labels[case],fontsize=11)
        ax.set_xlabel('Logical CNOT (shot-weighted mean)');ax.set_ylabel('CI energy minus reference (mHa)')
        ax.grid(True,which='major',alpha=.18)
    handles,legend=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='lower center',ncol=3,fontsize=8,frameon=False,bbox_to_anchor=(.5,.005))
    fig.suptitle('Fixed sources: compiler savings and CI selection are different effects',fontsize=12)
    fig.text(.5,.105,'100k model shots; R cap128 (actual dimensions vary); finite S2 penalty; reused observations; no hardware claim',
        ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.13,1,.95))
    for suffix in ['png','svg']:fig.savefig(ROOT/('analysis/G5-energy-CNOT-context.'+suffix),dpi=250)
    plt.close(fig)
    print('G5_SCIENTIFIC_FIGURE_SAVED_PNG_SVG')


if __name__=='__main__':main()
