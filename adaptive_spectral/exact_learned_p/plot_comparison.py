"""Export all four matched losses, global and per-matrix exponent histories."""
import argparse
import csv
import json
from pathlib import Path

from comparison import METHODS, read_json, read_metrics, verify_comparable


def write_csv(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    verify_comparable(root)
    output=root/'comparison'
    output.mkdir(exist_ok=True)
    figure,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    pfigure,paxes=plt.subplots(figsize=(9,4),constrained_layout=True)
    curves,summaries,history,final_rows=[],[],[],[]
    for name,label,mode,_,color in METHODS:
        if read_json(root/name/'progress.json').get('state') != 'complete':
            raise ValueError(f'{name}: run is not complete')
        metrics=read_metrics(root/name/'metrics.jsonl')
        train=[row for row in metrics if row['event']=='step']
        validation=[row for row in metrics if row['event']=='validation']
        if not train or not validation:
            raise ValueError(f'{name}: missing curves')
        for row in train:
            if not isinstance(row['p_used'],dict):
                row['p_used']={'global':row['p_used']}
                row['p_before']={'global':row['p_before']}
                row['p_mean']=row['p_min']=row['p_max']=row['p_used']['global']
                row['hypergradient_p']={'global':row['hypergradient_p']}
        signatures=([r['train_tokens'] for r in train],[r['step'] for r in validation])
        if name=='muon':
            reference_signatures=signatures
        elif signatures!=reference_signatures:
            raise ValueError(f'{name}: unmatched token exposure or validation steps')
        for column,xkey,scale in [(0,'step',1),(1,'elapsed_seconds',3600)]:
            axes[0,column].plot([r[xkey]/scale for r in validation],[r['loss'] for r in validation],
                                label=label,color=color,marker='o',markersize=3)
            axes[1,column].plot([r[xkey]/scale for r in train],[r['train_loss'] for r in train],
                                label=label,color=color,alpha=0.8,linewidth=1)
        paxes.plot([r['step'] for r in train],[r['p_mean'] for r in train],label=label,color=color)
        if name=='matrix_learned':
            paxes.fill_between([r['step'] for r in train],[r['p_min'] for r in train],
                               [r['p_max'] for r in train],color=color,alpha=0.2,label='Matrix p min–max')
            names=list(train[-1]['p_used'])
            heatfigure,heataxes=plt.subplots(figsize=(12,max(4,len(names)*0.18)),constrained_layout=True)
            pixels=[[row['p_used'][parameter] for row in train] for parameter in names]
            image=heataxes.imshow(pixels,aspect='auto',interpolation='nearest',vmin=0,vmax=1,cmap='viridis')
            tick_indices=sorted(set([0,len(train)//2,len(train)-1]))
            heataxes.set_xticks(tick_indices,[str(train[i]['step']) for i in tick_indices])
            heataxes.set_yticks(range(len(names)),names,fontsize=6 if len(names)>20 else 9)
            heataxes.set(xlabel='Optimizer step',title='Independently learned exponent per matrix')
            heatfigure.colorbar(image,ax=heataxes,label='p')
            heatfigure.savefig(output/'matrix_p_heatmap.png',dpi=160)
        for row in train:
            curves.append(dict(method=name,split='train',step=row['step'],elapsed_seconds=row['elapsed_seconds'],
                               response_tokens=row['train_tokens'],loss=row['train_loss'],
                               p_mean=row['p_mean'],p_min=row['p_min'],p_max=row['p_max']))
            for parameter,p in row['p_used'].items():
                history.append(dict(method=name,step=row['step'],parameter=parameter,p=p,
                                    p_before=row['p_before'][parameter],
                                    delayed_hypergradient=(row['hypergradient_p'] or {}).get(parameter,'')))
        for row in validation:
            curves.append(dict(method=name,split='validation',step=row['step'],elapsed_seconds=row['elapsed_seconds'],
                               response_tokens=row['train_tokens'],loss=row['loss'],p_mean='',p_min='',p_max=''))
        for parameter,p in train[-1]['p_used'].items():
            final_rows.append(dict(method=name,parameter=parameter,p=p))
        best=min(validation,key=lambda r:r['loss'])
        summaries.append(dict(method=name,label=label,initial_validation_loss=validation[0]['loss'],
                              final_validation_loss=validation[-1]['loss'],best_validation_loss=best['loss'],
                              best_validation_step=best['step'],final_train_loss=train[-1]['train_loss'],
                              final_p_mean=train[-1]['p_mean'],final_p_min=train[-1]['p_min'],final_p_max=train[-1]['p_max'],
                              training_seconds=train[-1]['elapsed_seconds'],train_tokens=train[-1]['train_tokens'],
                              p_updates=train[-1]['p_updates']))
    for row,label in [(0,'Selection response-token CE'),(1,'Training batch response-token CE')]:
        for column in (0,1):
            axes[row,column].set_ylabel(label)
            axes[row,column].set_xlabel('Optimizer step' if column==0 else 'Elapsed training time (hours)')
            axes[row,column].grid(alpha=0.25)
    axes[0,0].legend(fontsize=8)
    figure.suptitle('Exact SVD: fixed p, global learned p, and per-matrix learned p')
    for extension in ('png','svg'):
        figure.savefig(output/f'loss_curves.{extension}',dpi=160)
    paxes.set(xlabel='Optimizer step',ylabel='Mean exponent (shade: matrix min–max)',ylim=(-0.03,1.03))
    paxes.grid(alpha=0.25)
    paxes.legend(fontsize=8)
    pfigure.savefig(output/'p_trajectory.png',dpi=160)
    plt.close('all')
    for filename,rows in [('curves.csv',curves),('summary.csv',summaries),
                          ('matrix_p_history.csv',history),('final_matrix_p.csv',final_rows)]:
        write_csv(output/filename,rows)
    (output/'summary.json').write_text(json.dumps(summaries,indent=2))
    for row in summaries:
        print(f"{row['label']}: final validation CE={row['final_validation_loss']:.5f}; p mean={row['final_p_mean']:.5f}")
    print(f'Plots and tables: {output}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    plot(parser.parse_args().output)
