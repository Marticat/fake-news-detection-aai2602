"""Render measured results as a readable report and an editable slide deck."""
from pathlib import Path
import json,sys,shutil
import pandas as pd
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN=Path(sys.argv[1]).resolve()
OUT=ROOT/'outputs/Desktop_Delivery'
OUT.mkdir(parents=True,exist_ok=True)
meta=json.loads((RUN/'run.json').read_text(encoding='utf-8'))
if meta.get('status')!='complete':
    raise ValueError('Only completed runs may be reported.')
data=meta['data']; metrics=meta['metrics']
best=next(r for r in metrics if r['model']==meta['selected_model'])
robust=json.loads((RUN/'robustness.json').read_text())
audit=json.loads((RUN/'audit.json').read_text())
errors=pd.read_csv(RUN/f"errors_{best['model']}.csv",keep_default_na=False)
frame=pd.read_csv(RUN/'articles.csv',keep_default_na=False)

styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Body2',fontName='Helvetica',fontSize=10.5,leading=15,spaceAfter=9,textColor=colors.HexColor('#172E32')))
story=[]
def p(s,style='Body2'):story.append(Paragraph(s,styles[style]))
def h(s):p(s,'Heading1')
def table(rows,widths=None):
    converted=[[Paragraph(escape(str(v)),styles['Body2']) for v in row] for row in rows]
    t=Table(converted,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DCE9E3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#137E79')),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6)]))
    story.append(t);story.append(Spacer(1,12))
def page():story.append(PageBreak())

p('FAKE NEWS DETECTION','Title')
p('Programming for Data Analysis and Database','Heading2')
p('Applied Artificial Intelligence | AAI-2602')
p('Kurmet Adilzhan · Tolegenov Nursultan · Issakulov Zeiin')
p('Executed research report · 14 November 2026')
story.append(Spacer(1,25))
h('Abstract')
p(f'This project compares a majority-class baseline with BoW, TF-IDF, latent semantic analysis and frozen E5 neural embeddings for binary classification of ISOT news articles. The executed files contain {data["raw_rows"]:,} articles; {data["clean_rows"]:,} remain after cleaning. Detected duplicate families are isolated across training, validation and test partitions. The selected model, {best["model"].upper()}, achieves test macro-F1 {best["test"]["macro_f1"]:.4f}, with a conditional 95% bootstrap interval of {best["macro_f1_ci"]["low"]:.4f}–{best["macro_f1_ci"]["high"]:.4f}. Additional grouped splits and a chronological holdout assess sensitivity. SQLite connects articles, experiments and predictions through ten analytical queries.')
p('The results measure classification of this historical corpus. They do not establish performance on new publishers, languages or factual verification tasks. All numerical findings below come from saved model predictions; synthetic smoke-test results are excluded.')
h('Research question')
p('Which text representation gives the most useful balance of held-out classification quality and computational cost, compared with predicting the most frequent training label?')
p('Run identifier: '+escape(RUN.name))
page()
h('1. Data and provenance')
p('The topic specifies the Fake and Real News Dataset by Clement Bisaillon on Kaggle. Since Kaggle required authentication, the executed True.csv and Fake.csv were obtained from the public emilurosev/isot-fake-news-dataset mirror, pinned to commit aa8ca45d9da5a308479c52d121c42f36a1f97e69. The mirror attributes the data to an ISOT Kaggle republication. Byte-for-byte equivalence with the specified Kaggle archive was not independently verified.')
table([['Item','Observed value'],['Raw articles',data['raw_rows']],['Raw CSV bytes',data['raw_bytes']],['Empty bodies removed',data['empty_removed']],['Exact duplicate rows removed',data['duplicate_rows']],['Conflicting-label groups removed',data['conflicting_groups']],['Clean articles',data['clean_rows']],['Invalid dates',data['invalid_dates']],['Real / fake labels',f"{data['class_counts']['0']} / {data['class_counts']['1']}"]],[240,270])
p('The two CSV files use title, text, subject and date. File membership supplies the label: True.csv = 0 and Fake.csv = 1. Subject and date are used for analysis, not predictive features. The historical corpus contains publisher, style and topic biases; its labels are accepted as dataset annotations rather than independently rechecked claims.')
p(f'The actual input is {data["raw_bytes"]/1e6:.2f} MB and {data["raw_rows"]:,} rows. Recommended volume threshold met: {data["volume_threshold_met"]}. If below the recommended 200 MB or 500,000 rows, this scope exception requires instructor agreement. Repeating articles or inflating files would not supply independent data.')
for name,v in data['source_files'].items():p(escape(name)+': SHA-256 '+v['sha256'], 'Code')
page()
h('2. Processing and experimental protocol')
p('CSV ingestion uses chunks of 5,000 rows. HTML, URLs, punctuation and explicit Reuters markers are removed; text is lowercased. Exact normalized body hashes identify duplicates across files and chunks. Empty bodies are excluded. All occurrences of conflicting body labels are excluded.')
p('Before splitting, articles sharing a normalized title of at least six words, or an identical first 80 body words in bodies of at least 100 words, are joined into connected families. GroupShuffleSplit isolates these families. This conservative heuristic reduces repeated-story leakage but cannot detect every paraphrase. Group splitting is not exact label stratification; the observed partition composition is recorded.')
table([['Partition','Articles','Real','Fake']]+[[s,len(g),int((g.label==0).sum()),int((g.label==1).sum())] for s,g in frame.groupby('split')],[150,120,120,120])
p('Seed 42 defines the primary approximately 70/15/15 split. Vocabulary, IDF, SVD and logistic regression are fitted only on training data. The highest validation macro-F1 selects the model; test results are not used for selection. Identical scores use deterministic model-list order. No hyperparameter search was performed.')
p('Lexical matrices use sparse float32 arrays, at most 30,000 unigram/bigram features and min_df=2. Labels use int8 and lengths int32. Chunked reading controls ingestion overhead, but cleaned texts and fitted models still reside in memory. Parquet is optional; this execution used CSV and SQLite because a Parquet engine was unavailable.')
h('3. Models')
table([['Method','Representation / classifier'],['Baseline','Most frequent training class'],['BoW','Counts, L2 normalization, Logistic Regression'],['TF-IDF','Sublinear TF-IDF, Logistic Regression'],['LSA','TF-IDF, up to 128 SVD components, L2, Logistic Regression'],['E5','Frozen multilingual-e5-small, 384-dimensional vectors, Logistic Regression']],[90,420])
page()
h('4. Neural representation and measurement')
p('E5 uses the pinned pretrained revision 614241f622f53c4eeff9890bdc4f31cfecc418b3. Each input receives the query: prefix recommended for classification. Attention-mask mean pooling excludes padding and L2 normalization produces 384-dimensional vectors. The encoder is frozen; only the classifier learns labels. CPU inference uses batches of 16 and truncates inputs at 128 tokenizer tokens. This bounded budget can lose evidence near the end of long articles; lexical methods see the full cleaned text, so this is a practical pipeline comparison, not a controlled full-document representation ablation.')
p('All logistic regression classifiers use C=1, liblinear, max_iter=2000 and random_state=42. E5 weights are separate from the saved classifier and must remain available for inference. E5 uses four CPU threads; core isolated workers use one. Concurrent robustness work and CPU contention affect timings. Resource values describe this execution and are not a hardware-independent speed ranking.')
p('Accuracy, fake-class precision/recall, macro-F1 and ROC-AUC are measured on the same held-out primary test set. Macro-F1 averages the two class F1 scores. The confusion matrix uses real then fake for both axes. Stratified article bootstrap with 1,000 draws and seed 42 produces percentile 95% macro-F1 intervals. These intervals condition on the trained model and do not account for all family dependence or training uncertainty.')
table([['Model','Val F1','Test F1','Accuracy','95% F1 interval']]+[[r['model'],f"{r['validation']['macro_f1']:.4f}",f"{r['test']['macro_f1']:.4f}",f"{r['test']['accuracy']:.4f}",f"{r['macro_f1_ci']['low']:.4f}–{r['macro_f1_ci']['high']:.4f}"] for r in metrics],[65,80,80,80,205])
table([['Model','Fit seconds','Test inference s','Peak RSS MiB','Model MiB']]+[[r['model'],f"{r['fit_seconds']:.2f}",f"{r['predict_seconds']:.2f}",f"{r['peak_rss_mib']:.1f}",f"{r['model_mib']:.2f}"] for r in metrics],[65,105,115,115,110])
p('RSS samples the entire worker every 20 ms, including Python and data; short peaks may be missed. E5 fit and inference include encoding, while its saved joblib excludes external encoder weights (470.64 MB).')
page()
h('5. Robustness checks')
p('Three additional group splits (seeds 7, 21 and 84) rerun the baseline and lexical models from scratch. These are sensitivity analyses with overlapping data, not independent external datasets. They do not replace the primary validation-selected model. Neural repeated-split evaluation was not executed because of CPU cost.')
table([['Check','Model','Macro-F1','Train / test']]+[[('seed '+str(r['seed'])) if r['check']=='grouped_seed' else 'temporal',r['model'],f"{r['macro_f1']:.4f}",f"{r['train_n']} / {r['test_n']}"] for r in robust],[105,85,100,220])
temporal=[r for r in robust if r['check']=='temporal']
if temporal:
    r=temporal[0]
    p(f'The chronological boundary is {r["cutoff"]}, the empirical 80th percentile date. Training uses families entirely before the boundary; test uses families entirely on or after it. Families crossing the boundary and undated rows are excluded ({r["excluded_n"]} rows total). Both partitions contain both labels. This is a within-corpus time shift, not a test on unseen publishers.')
    matrix=r['confusion_matrix']
    p(f'The temporal test contains {sum(matrix[0]):,} real and {sum(matrix[1]):,} fake articles. This class-distribution shift explains why the training-majority baseline deteriorates sharply; macro-F1 is more informative here than accuracy alone.')
page()
h('6. Error analysis')
cm=best['test']['confusion_matrix']
p(f'The validation-selected {best["model"].upper()} makes {len(errors)} errors among {best["test_n"]} test articles. False positives: {cm[0][1]}; false negatives: {cm[1][0]}. Fake-class precision is {best["test"]["precision_fake"]:.4f}; recall is {best["test"]["recall_fake"]:.4f}.')
table([['Actual / predicted','Real','Fake'],['Real',*cm[0]],['Fake',*cm[1]]],[230,140,140])
p('The examples below are the highest-confidence mistakes of each type. Scores are model outputs, not calibrated truth probabilities. Labels and titles are reproduced for auditing; this report does not independently endorse the underlying claims.')
for kind in ('false_positive','false_negative'):
    for r in errors[errors.error_type==kind].head(2).itertuples():
        p(f'{kind.replace("_"," ").title()} · article {r.article_id} · P(fake)={r.probability_fake:.3f}','Heading3')
        p(escape(str(r.title)))
        p('Subject: '+escape(str(r.subject))+'. '+escape(str(r.text)[:380])+'…')
p('Interpretation: shared political vocabulary and overlapping reporting styles can confuse a word-based classifier. Short or unusual articles provide little evidence. These are hypotheses suggested by examples; they are not causal explanations. Full error CSVs, term weights and subject/length error summaries support further inspection.')
page()
h('7. Database and architecture')
p('The project separates data ingestion, models, worker measurement, orchestration, SQL, uncertainty checks and reporting into Python modules. The CLI provides loading/training, classification, SQL export, report generation and a synthetic software test. Error handling reports missing files, invalid schemas, insufficient samples and incomplete runs.')
from newscheck.reporting import make_er
make_er(RUN/'er_diagram.png')
story.append(RLImage(str(RUN/'er_diagram.png'),width=510,height=255))
p('SQLite tables: subjects, articles, experiments and predictions. Foreign keys connect each article to a subject and every prediction to an article and an experiment. Ten queries cover class balance, subject GROUP BY/HAVING, monthly counts, an average-length subquery, a CTE with ROW_NUMBER, model errors by JOIN, confusion counts, subject error rates, cumulative SUM OVER and validation DENSE_RANK. Results are loaded into Pandas and exported as CSV.')
h('8. Conclusions and limitations')
p(f'{best["model"].upper()} is selected by primary validation macro-F1. The trained text models improve on the majority baseline on this corpus. The real-data experiment replaces the earlier synthetic demonstration and produces nonzero errors that can be inspected. High scores remain compatible with source/style shortcuts and should not be interpreted as general fact-checking accuracy.')
p('Remaining limitations: historical English data, no independent publisher-held-out external test, heuristic near-duplicate detection, truncated E5 input, one hardware environment, conditional bootstrap intervals and a corpus below the recommended volume. Additional corpora should be deduplicated against ISOT before external evaluation.')
table([['Diagnostic (not candidate models)','Test macro-F1']]+[[r['check'],f"{r['macro_f1']:.4f}"] for r in audit],[370,140])
p('A subject-only diagnostic learns each subject’s training fake-label frequency. High diagnostic performance reveals label/category confounding, which can also be reflected indirectly in text. Subject is never an input to the deployed classifiers. The title-only and body-only TF-IDF ablations use the same primary split and fixed settings; they are diagnostic, not additional model-selection trials.')
p('Authorship: the three students are listed as the project team. No individual work allocation has been verified. Each participant must add an accurate contribution statement before submission; no invented allocation is presented as fact.')
page()
h('9. Reproduction and references')
for s in ['python -m pip install -r requirements.txt','python -m pip install torch transformers','python main.py run --raw data/raw --include-e5','python -m newscheck.robustness --run runs/RUN_ID','python -m unittest discover -s tests -v']:
    p(s,'Code')
p('Download the linked CSV files and preserve filenames. Record the environment and manifests for each execution. A pretrained E5 download is required on a fresh machine; NEWSCHECK_E5_PATH may point to a local model directory. Run the neural option only when torch and transformers are installed. Saved joblib files are trusted project outputs and should not be loaded from unknown sources.')
refs=[('Project repository','https://github.com/Marticat/fake-news-detection-aai2602'),('Requested Kaggle data card','https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset'),('Executed ISOT mirror','https://github.com/emilurosev/isot-fake-news-dataset'),('ISOT dataset documentation','https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf'),('E5 model card','https://huggingface.co/intfloat/multilingual-e5-small'),('scikit-learn documentation','https://scikit-learn.org/stable/'),('SQLite window functions','https://www.sqlite.org/windowfunctions.html')]
for name,url in refs:p(name+': <link href="'+url+'">'+url+'</link>')
def footer(c,doc):
    c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#607477'));c.drawString(42,25,'AAI-2602 | Fake News Detection | Measured ISOT experiment');c.drawRightString(552,25,str(doc.page))
SimpleDocTemplate(str(OUT/'Fake_News_Report.pdf'),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)

# Reuse the established visual primitives; all slide contents below are measured.
prefix=(ROOT/'tools/slide_primitives.py').read_text(encoding='utf-8')
exec(prefix,globals())
start('Programming for Data Analysis and Database','',True)
photo(770,0,510,650);rect(64,106,65,5,'#D9A45F')
text('Fake News<br/>Detection',64,145,700,70,paper,'Serif')
text(f'A measured study of<br/>{data["clean_rows"]:,} news articles',67,336,640,30,'#B7D2CC')
text('BoW / TF-IDF / LSA / Neural E5',67,443,640,23,paper)
text('Kurmet Adilzhan · Tolegenov Nursultan<br/>Issakulov Zeiin',67,530,650,22,paper);end()

start('01 / Research question','Which representation earns its complexity?')
photo(64,195,360,405)
text('Predict the dataset label<br/>from article text.',475,204,710,38,ink,'Serif')
for yy,a,b in [(340,'Quality','Macro-F1, precision, recall and real mistakes'),(430,'Cost','Fit time, inference time and process memory'),(520,'Stability','Different group splits and later articles')]:
    text(a,475,yy,710,26,teal,'Bold');text(b,475,yy+38,710,23)
end()

start('02 / Data','From a public corpus to a clean experiment')
for x,value,label in [(64,f'{data["raw_rows"]:,}','input articles'),(470,f'{data["clean_rows"]:,}','clean articles'),(875,f'{data["raw_bytes"]/1e6:.0f} MB','raw CSV data')]:
    text(value,x,225,360,65,teal,'Serif');text(label,x,320,360,27,ink,'Bold')
text(f'{data["empty_removed"]:,} empty bodies removed<br/>{data["duplicate_rows"]:,} exact duplicate rows removed',64,425,560,28)
text('ISOT public mirror, pinned commit<br/>File hashes saved with every run',675,425,535,26)
note('The selected corpus is below the volume guideline; an instructor exception is still needed.');end()

start('03 / Leakage control','Related publications stay together')
for x,a,b in [(64,'Clean','Remove HTML, URLs<br/>and Reuters markers'),(465,'Group','Exact bodies, long titles<br/>and 80-word prefixes'),(866,'Separate','No detected family<br/>crosses partitions')]:
    text(a,x,220,350,36,teal,'Serif');text(b,x,295,350,25)
x=64
for label,num,ww,col in [('TRAIN',data['split_counts']['train'],750,teal),('VALIDATION',data['split_counts']['validation'],201,'#75AAA0'),('TEST',data['split_counts']['test'],201,orange)]:
    rect(x,440,ww,98,col);text(label,x+14,452,ww-20,16,paper,'Bold');text(f'{num:,}',x+14,484,ww-20,29,paper,'Bold');x+=ww
note('Vocabulary and classifiers learn from train. Validation chooses the model. Test measures it.');end()

start('04 / Methods','Four representations, one classifier family')
for i,(a,b) in enumerate([('BoW','Normalized word<br/>and bigram counts'),('TF-IDF','Weighted words<br/>and bigrams'),('LSA','128-dimensional<br/>latent vectors'),('E5','384-dimensional<br/>neural vectors')]):
    x=64+i*295;text(a,x,226,260,44,teal,'Serif');text(b,x,320,265,26)
rect(64,451,1152,65,ink);text('Logistic Regression · C = 1 · Same primary split',88,467,1100,28,paper)
text('Baseline: predict the most frequent training label.',64,550,1150,25)
text('E5 is frozen and sees the first 128 tokens; lexical models see full text.',64,605,1150,22,orange);end()

start('05 / Primary results',f'Measured quality on {best["test_n"]:,} held-out articles')
for i,r in enumerate(metrics):
    y=210+i*74;v=r['test']['macro_f1'];text(r['model'].upper(),64,y,170,25)
    rect(235,y,600,34,'#E0E3D9');rect(235,y,600*v,34,teal if r['model']!='baseline' else muted)
    text(f'{v:.4f}',860,y,150,26,ink,'Bold')
    text(f"{r['macro_f1_ci']['low']:.3f}–{r['macro_f1_ci']['high']:.3f}",1025,y,190,22,muted)
note(f"Selected on validation: {best['model'].upper()}. Right column: conditional 95% macro-F1 intervals.");end()

start('06 / Errors','High accuracy still leaves meaningful mistakes')
text(best['model'].upper()+' · primary test',64,195,550,27,ink,'Bold')
text('Predicted real',190,255,200,20);text('Predicted fake',405,255,200,20)
for row in range(2):
    text(['Real','Fake'][row],64,335+row*120,100,23)
    for col in range(2):
        x=180+col*215;y=295+row*120;rect(x,y,205,110,teal if row==col else '#DCE9E3');text(str(cm[row][col]),x+30,y+27,170,42,paper if row==col else ink,'Serif')
text(str(len(errors)),735,225,420,80,orange,'Serif');text('misclassified articles',735,325,450,30,ink,'Bold')
text(f'False positives: {cm[0][1]}<br/>False negatives: {cm[1][0]}',735,415,450,28)
note('Confusion axes use the original dataset labels; the model does not independently verify claims.');end()

start('07 / Error examples','Inspect the article behind the score')
for i,kind in enumerate(('false_positive','false_negative')):
    candidates=errors[errors.error_type==kind];x=64+i*600
    if candidates.empty:
        text('No observed '+kind.replace('_',' '),x,260,535,27,ink,'Serif')
        continue
    item=candidates.iloc[0]
    text(kind.replace('_',' ').upper(),x,210,550,18,teal,'Bold')
    text(escape(str(item.title)),x,260,535,27,ink,'Serif')
    text(f"Article {item.article_id}<br/>P(fake): {item.probability_fake:.3f}<br/>Subject: {escape(str(item.subject))}",x,460,535,24)
note('Shared vocabulary and reporting style are plausible error sources, not proven causal explanations.');end()

start('08 / Robustness','Does performance survive another split?')
text('Macro-F1 across three extra group splits',64,200,1150,26,ink,'Bold')
for i,name in enumerate(('baseline','bow','tfidf','lsa')):
    vals=[r['macro_f1'] for r in robust if r['check']=='grouped_seed' and r['model']==name]
    y=270+i*68;text(name.upper(),64,y,230,26);text(f'{min(vals):.4f} – {max(vals):.4f}',340,y,420,28,teal,'Bold')
    tr=next((r for r in robust if r['check']=='temporal' and r['model']==name),None)
    text(f"Later articles: {tr['macro_f1']:.4f}" if tr else 'Temporal result unavailable',810,y,400,25)
note('Seeds 7 / 21 / 84. Chronological holdout is within the same corpus, not an external test.');end()

start('09 / Bias audit','A shortcut hidden in the dataset')
text('1.000',64,220,550,90,orange,'Serif')
text('macro-F1 using subject alone',64,335,550,30,ink,'Bold')
text('This diagnostic reveals complete<br/>subject / label separation<br/>on the held-out split.',64,410,550,27)
text('Our text models do not use subject.',690,230,520,29,teal,'Bold')
text('But topic and publisher style<br/>can still appear in vocabulary.<br/><br/>An independent external corpus<br/>is needed to test generalization.',690,320,520,27)
note('Diagnostic only: the metadata model is not selected, deployed or counted as a text representation.');end()

start('10 / Computation','The neural encoder has a measurable cost')
xs=[80,365,660,960]
for x,v in zip(xs,['Model','Fit seconds','Predict seconds','Peak RSS, MiB']):text(v,x,200,260,21,teal,'Bold')
for i,r in enumerate(metrics):
    y=260+i*59
    for x,v in zip(xs,[r['model'].upper(),f"{r['fit_seconds']:.1f}",f"{r['predict_seconds']:.1f}",f"{r['peak_rss_mib']:.0f}"]):text(v,x,y,260,27)
text('E5: 4 CPU threads; lexical workers: 1. Background work affects timings.',64,580,1150,23,orange)
text('These are observed execution costs, not a controlled hardware benchmark.',64,620,1150,21,muted);end()

start('10 / Database','Every prediction remains traceable')
def entity(x,y,title,fields):
    rect(x,y,440,45,teal);text(title,x+16,y+8,410,23,paper,'Bold');rect(x,y+45,440,97,pale);text(fields,x+16,y+61,410,22)
entity(64,200,'subjects','PK subject_id<br/>name');entity(64,430,'articles','PK article_id · FK subject_id<br/>text · label · split · hash')
entity(776,200,'experiments','PK model<br/>config · metrics · runtime');entity(776,430,'predictions','PK / FK model + article_id<br/>predicted · probability_fake')
for x in (284,996):line(x,342,x,430,teal,3)
line(504,500,776,500,teal,3);text('1 : many',565,460,180,22,teal)
note('10 SQL queries: JOIN, GROUP BY/HAVING, subquery, CTE and window functions.');end()

start('11 / Reproduce','A complete run leaves auditable evidence')
for i,(a,b) in enumerate([('Input','CSV hashes<br/>Cleaning manifest'),('Models','Saved classifiers<br/>Pinned neural weights'),('Analysis','Predictions and errors<br/>SQL and robustness CSVs'),('Deliver','Written report<br/>Editable presentation')]):
    x=64+i*295;text(str(i+1).zfill(2),x,220,260,45,teal,'Serif');text(a,x,300,270,29,ink,'Bold');text(b,x,355,270,24)
rect(64,485,1152,72,ink);text('python main.py run --raw data/raw --include-e5',85,506,1110,25,paper)
text('python -m newscheck.robustness --run runs/RUN_ID',64,608,1150,24);end()

start('12 / Conclusions','Useful evidence, with a defined scope',True)
text('Established',64,210,530,36,'#91C6BD','Serif');text('Real-data training and test results<br/>Neural embeddings executed<br/>Nonzero errors inspected<br/>Split and temporal sensitivity measured',64,285,540,27,paper)
text('Still limited',690,210,510,36,'#D9A45F','Serif');text('Historical source and topic biases<br/>No independent external corpus<br/>Heuristic duplicate families<br/>128-token neural input budget',690,285,530,27,paper)
text('Before submission: approve the dataset-size exception<br/>and record each member’s actual contribution.',64,550,1150,28,paper);end()

start('13 / Sources','Data, implementation and image credits')
for y,a,b in [(205,'Repository','github.com/Marticat/fake-news-detection-aai2602'),(285,'Corpus','ISOT · emilurosev/isot-fake-news-dataset<br/>Requested data card: Kaggle / Clement Bisaillon'),(390,'Methods','scikit-learn · SQLite · intfloat/multilingual-e5-small'),(475,'Photography','Mr Cup / Fabien Barral on Unsplash<br/>unsplash.com/photos/Mwuod2cm8g4')]:
    text(a,64,y,260,25,teal,'Bold');text(b,350,y,850,23)
text('All charts use the saved real-data run. Full references and hashes are in the report.',64,605,1150,23,orange)
end();c.save();deck.save(OUT/'Fake_News_Presentation.pptx')
print(OUT)
