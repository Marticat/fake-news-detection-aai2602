from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS=['#64748b','#2b6cb0','#0e9388','#d08a35','#845ec2']


def make_charts(run_dir,df,meta):
    folder=Path(run_dir)/'figures'
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    suffix=' | SYNTHETIC TEST DATA' if meta['dataset_kind']=='synthetic_demo' else ''
    rows=meta['metrics']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    names=[r['model'].upper() for r in rows]
    axes[0].bar(names,[r['test']['macro_f1'] for r in rows],color=COLORS[:len(rows)])
    axes[0].set(ylim=(0,1.1),ylabel='Test macro-F1',title='Classification quality')
    axes[1].bar(names,[r['fit_seconds'] for r in rows],color=COLORS[:len(rows)])
    axes[1].set(ylabel='Seconds',title='Representation + classifier fit')
    fig.suptitle('Representation comparison'+suffix,fontsize=14)
    fig.savefig(folder/'comparison.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    counts=df.label.value_counts().reindex([0,1],fill_value=0)
    axes[0].bar(['Real (0)','Fake (1)'],counts.values,color=['#2b6cb0','#d08a35'])
    axes[0].set(ylabel='Articles',title='Cleaned class balance')
    for label,color in [(0,'#2b6cb0'),(1,'#d08a35')]:
        axes[1].hist(df.loc[df.label==label,'word_count'],bins=30,alpha=.6,label=['Real','Fake'][label],color=color)
    axes[1].set(xlabel='Body word count',ylabel='Articles',title='Article length')
    axes[1].legend()
    fig.suptitle('Dataset overview'+suffix,fontsize=14)
    fig.savefig(folder/'overview.png',dpi=180)
    plt.close(fig)
    n=len(rows)
    fig,axes=plt.subplots(1,n,figsize=(3.1*n,3.5),layout='constrained')
    for ax,row in zip(np.atleast_1d(axes),rows):
        cm=np.array(row['test']['confusion_matrix'])
        ax.imshow(cm,cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else '#172b4d',fontsize=14)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Real','Fake'],yticklabels=['Real','Fake'],xlabel='Predicted',ylabel='Actual',title=row['model'].upper())
    fig.suptitle('Test confusion matrices'+suffix,fontsize=13)
    fig.savefig(folder/'confusion.png',dpi=180)
    plt.close(fig)
