"""Source-specific H figures including incomplete iterations and raw batches."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .io import ROOT,save_json
from .analysis_products import write_csv


def run():
    out=ROOT/'analysis-v1';batches=[];figures=[]
    for case in('P3-N2-ratio1p00','P3-N2-ratio2p00','P3-FE2S2-40'):
        folder=ROOT/'hardware-postprocessing-v2'/case/'output'
        result=folder/'result.json'
        if not result.exists():continue
        v=json.loads(result.read_text());m=json.loads((ROOT/'hardware-physical-v1'/case/'physical.json').read_text())
        ref=json.loads((ROOT/'hardware-physical-v1'/case/'evaluation-reference.json').read_text())
        no=json.loads((folder/'no-CR.json').read_text())if(folder/'no-CR.json').exists()else{}
        before=no.get('CI');den=m['hf_energy_Ha']-ref['reference_energy_Ha'];history=v['iterations'];x=[r['iteration']+1 for r in history]
        for path in sorted(folder.glob('CI-*-*.json')):
            c=json.loads(path.read_text())
            batches.append({'case_id':case,'problem_id':v['problem_id'],'iteration':c['iteration']+1,'batch_zero_based':c['batch'],
                'energy_Ha':c['energy_Ha'],'actual_CI_dimension':c['actual_d'],'spin_string_count':c['spin_strings'],
                'CI_converged':c['converged'],'residual_norm':c['residual_norm'],'spin_square':c['spin_square'],
                'matvec_calls':c['matvec_calls'],'wall_seconds':c['wall_seconds'],
                'belongs_to_complete_10_batch_iteration':c['iteration']in{r['iteration']for r in history},
                'correlation_recovery_vs_reported_reference_pct':100*(m['hf_energy_Ha']-c['energy_Ha'])/den if den>0 else None,
                'new_QPU_shots':0,'source':str(path.relative_to(ROOT))})
        fig,axes=plt.subplots(2,2,figsize=(11,7.8),layout='constrained')
        if before:
            E=before['energy_Ha'];axes[0,0].axhline(1000*(E-m['hf_energy_Ha']),ls='--',color='gray',label='No CR: all correct-sector strings')
            if den>0:axes[0,1].axhline(100*(m['hf_energy_Ha']-E)/den,ls='--',color='gray',label='No CR')
            axes[1,0].axhline(before['actual_d'],ls='--',color='gray',label='No CR dimension')
        else:axes[0,0].text(.03,.97,'No-CR result unavailable/resource capped',transform=axes[0,0].transAxes,va='top',fontsize=8)
        if history:
            mean=[1000*(r['mean_energy_Ha']-m['hf_energy_Ha'])for r in history]
            lower=[1000*(r['mean_energy_Ha']-r['minimum_energy_Ha'])for r in history]
            upper=[1000*(r['maximum_energy_Ha']-r['mean_energy_Ha'])for r in history]
            axes[0,0].errorbar(x,mean,yerr=[lower,upper],fmt='o-',capsize=3,label='CR: complete 10-batch mean/range')
            if den>0:axes[0,1].plot(x,[100*(m['hf_energy_Ha']-r['mean_energy_Ha'])/den for r in history],'o-',label='Versus source reference')
            for r in history:axes[1,0].scatter([r['iteration']+1]*len(r['actual_d']),r['actual_d'],s=10,alpha=.6,color='#1b6ca8')
            axes[1,1].plot(x,[r['CPU_seconds']/60 for r in history],'o-',label='Cumulative worker CPU minutes')
        partial=[b for b in batches if b['case_id']==case and not b['belongs_to_complete_10_batch_iteration']]
        if partial:
            px=[b['iteration']for b in partial]
            axes[0,0].scatter(px,[1000*(b['energy_Ha']-m['hf_energy_Ha'])for b in partial],s=20,color='gray',marker='x',label='Incomplete iteration: individual batches')
            if den>0:axes[0,1].scatter(px,[b['correlation_recovery_vs_reported_reference_pct']for b in partial],s=20,color='gray',marker='x',label='Incomplete iteration: individual batches')
            axes[1,0].scatter(px,[b['actual_CI_dimension']for b in partial],s=20,color='gray',marker='x',label='Incomplete iteration: individual batches')
        terminal_x=len(list(folder.glob('CI-*-*.json')))/10
        axes[1,1].scatter([terminal_x],[v['CPU_seconds']/60],marker='x',color='#b7472a',label='Paid CPU including partial final iteration')
        axes[1,1].annotate('Terminal cost',xy=(terminal_x,v['CPU_seconds']/60),xytext=(4,5),textcoords='offset points',fontsize=7)
        axes[0,0].set_ylabel('CI energy - HF (mHa); lower is better')
        axes[0,1].set_ylabel('Correlation recovery (%) vs reported reference')
        axes[1,0].set_ylabel('Actual Cartesian CI dimension (batch)')
        axes[1,0].set_yscale('log');axes[1,1].set_ylabel('CPU minutes')
        for ax in axes.flat:
            ax.set_xlim(-.2,10.5);ax.set_xticks(range(0,11,2));ax.set_xlabel('CR iteration (0 = before CR; grey = incomplete)');ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
        axes[1,1].set_xlabel('Completed CI batches / 10 (partial work retained)')
        title=case+' | current-runtime public-weight reanalysis; NOT primary comparison\n'+str(len(history))+'/10 complete CR iterations; '+str(len(list(folder.glob('CI-*-*.json'))))+'/100 completed batches. Whiskers = batch range, not confidence interval.'
        fig.suptitle(title,fontsize=10)
        name=case+'-public-CR-capacity-cost'
        for ext in('png','svg'):fig.savefig(out/(name+'.'+ext),dpi=200)
        plt.close(fig);figures.append(name+'.png')
    write_csv(out/'H-all-observed-CI-batches-including-partial-iterations.csv',batches)
    save_json(out/'hardware-figures-status.json',{'figures':figures,'all_observed_batches':len(batches),
        'new_QPU_shots':0,'not_original_runtime_reproduction':True,'whole_experiment_complete':False})
    print(json.dumps({'figures':len(figures),'observed_batches_including_partial':len(batches)}),flush=True)


if __name__=='__main__':run()
