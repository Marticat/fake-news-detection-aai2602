"""Generate the report and 16:9 slide deck from one measured run."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image
from reportlab.pdfgen import canvas

NAVY=colors.HexColor('#142b45')
TEAL=colors.HexColor('#0b8b80')
ORANGE=colors.HexColor('#b76618')
LIGHT=colors.HexColor('#edf3f7')
AUTHORS=['Kurmet Adilzhan','Tolegenov Nursultan','Issakulov Zeiin']
REFERENCES=[
 ('Kaggle. Fake and Real News Dataset (data card).','https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset'),
 ('University of Victoria, ISOT. Fake News Dataset README.','https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf'),
 ('scikit-learn. Feature extraction.','https://scikit-learn.org/stable/modules/feature_extraction.html'),
 ('scikit-learn. Common pitfalls and recommended practices.','https://scikit-learn.org/stable/common_pitfalls.html'),
 ('Sentence Transformers. Semantic textual similarity.','https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html'),
 ('SQLite. Window functions.','https://www.sqlite.org/windowfunctions.html')]


def make_er(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(11,5.5))
    ax.set(xlim=(0,11),ylim=(0,6))
    ax.axis('off')
    boxes=[(.3,3.4,'subjects',['PK subject_id','UNIQUE name']),
           (6.2,3.4,'articles',['PK article_id / FK subject_id','title, text, clean_text, date','label, UNIQUE text_hash','split, word_count']),
           (.3,.3,'experiments',['PK model','config_json, validation_f1, test_f1','fit_seconds, predict_seconds','peak_rss_mib, model_mib']),
           (6.2,.3,'predictions',['PK (model, article_id)','FK model / FK article_id','predicted, probability_fake'])]
    for x,y,title,lines in boxes:
        ax.add_patch(FancyBboxPatch((x,y),4.5,2.15,boxstyle='round,pad=0.08',facecolor='#f0f5f8',edgecolor='#74879a'))
        ax.text(x+.2,y+1.78,title,fontsize=15,fontweight='bold',color='#142b45')
        for i,line in enumerate(lines):
            ax.text(x+.2,y+1.35-i*.32,line,fontsize=10,color='#243c50')
    for start,end,label,labelpos in [((4.9,4.4),(6.1,4.4),'1 : N',(5.5,4.65)),((4.9,1.35),(6.1,1.35),'1 : N',(5.5,1.6)),((8.45,3.25),(8.45,2.6),'1 : N',(9,2.9))]:
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#0b8b80',lw=2))
        ax.text(*labelpos,label,ha='center',fontsize=11,color='#0b8b80')
    fig.tight_layout()
    fig.savefig(path,dpi=180,bbox_inches='tight')
    plt.close(fig)


def styles():
    s=getSampleStyleSheet()
    s.add(ParagraphStyle(name='Body2',fontName='Helvetica',fontSize=10,leading=15,textColor=NAVY,spaceAfter=10))
    s.add(ParagraphStyle(name='Small2',fontName='Helvetica',fontSize=8,leading=11,textColor=NAVY,spaceAfter=7))
    s.add(ParagraphStyle(name='Title2',fontName='Helvetica-Bold',fontSize=32,leading=37,textColor=NAVY,spaceAfter=18))
    s.add(ParagraphStyle(name='Head2',fontName='Helvetica-Bold',fontSize=19,leading=24,textColor=NAVY,spaceAfter=15))
    s.add(ParagraphStyle(name='Sub2',fontName='Helvetica-Bold',fontSize=12,leading=16,textColor=TEAL,spaceAfter=8))
    return s


def build_documents(run_dir):
    run=Path(run_dir)
    meta=json.loads((run/'run.json').read_text(encoding='utf-8'))
    if meta['status']!='complete':
        raise ValueError('Documents require a completed run.')
    make_er(run/'figures'/'er_diagram.png')
    build_report(run,meta)
    build_slides(run,meta)
    print(f'PDF report and slides: {run}',flush=True)


def build_report(run,meta):
    s=styles()
    story=[]
    demo=meta['dataset_kind']=='synthetic_demo'
    status='SYNTHETIC SOFTWARE VALIDATION - NOT KAGGLE RESEARCH RESULTS' if demo else 'EMPIRICAL RUN - LOCAL ISOT CSV INPUT'
    data=meta['data']
    rows=meta['metrics']
    best=next(r for r in rows if r['model']==meta['selected_model'])
    def p(text,style='Body2'):
        story.append(Paragraph(text,s[style]))
    def heading(text):
        p(text,'Head2')
    def page():
        story.append(PageBreak())
    def table(values,widths):
        cells=[[Paragraph(escape(str(x)),s['Small2']) for x in row] for row in values]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),1,TEAL),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dce5eb')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.append(t)
        story.append(Spacer(1,12))
    def picture(name,w=490,h=190):
        story.append(Image(str(run/'figures'/name),width=w,height=h,kind='proportional'))
        story.append(Spacer(1,10))
    p('ASTANA IT UNIVERSITY  /  AAI-2602','Sub2')
    story.append(Spacer(1,24))
    p('Fake News<br/>Detection','Title2')
    p('A reproducible comparison of text representations<br/>with Python and SQLite','Head2')
    p('Programming for Data Analysis and Database<br/>Applied Artificial Intelligence - Master\'s programme')
    p('<br/>'.join(AUTHORS))
    p(status,'Sub2')
    p('Abstract','Sub2')
    p('This project integrates text preparation, relational storage, machine learning and error analysis in a modular Python application. A majority-class baseline is compared with normalized Bag of Words, TF-IDF and dense LSA embeddings using a common logistic regression classifier. A frozen MiniLM encoder is available as an additional experiment. Results, SQL exports and documents are generated from a single run manifest.')
    if demo:
        p('The delivered execution uses 240 explicitly synthetic articles to validate software behavior. Its scores do not estimate performance on genuine news. Kaggle files were unavailable during preparation; the empirical experiment and final scientific conclusions remain pending. Running the same pipeline on the real CSV files regenerates this report with measured results.')
    else:
        p(f'This run retained {data["clean_rows"]:,} articles and selected {best["model"].upper()} by validation macro-F1. Its held-out test macro-F1 is {best["test"]["macro_f1"]:.4f}. This is an in-dataset evaluation; external factual reliability has not been established.')
    p(f'Run: {escape(run.name)}<br/>Created (UTC): {escape(meta["created_utc"])}','Small2')
    page()
    heading('1. Data and preparation')
    p('The intended source is the Kaggle Fake and Real News Dataset, derived from ISOT [1,2]. Files contain title, text, subject and date. Labels come from the file identity: True.csv = 0, Fake.csv = 1. Published dataset descriptions differ by version; this project counts the actual local files and records SHA-256 hashes.')
    table([['Property','Measured value'],['Dataset kind',meta['dataset_kind']],['Raw / retained rows',f'{data["raw_rows"]:,} / {data["clean_rows"]:,}'],['Raw CSV size',f'{data["raw_bytes"]/1e6:.3f} MB (decimal)'],['Empty bodies / repeated rows',f'{data["empty_removed"]} / {data["duplicate_rows"]}'],['Conflicting duplicate groups',data['conflicting_groups']],['Unparsed retained dates',data['invalid_dates']],['Train / validation / test',' / '.join(str(data['split_counts'].get(k,0)) for k in ['train','validation','test'])],['Volume guideline met',str(data['volume_threshold_met'])]],[220,270])
    p('Cleaning decodes HTML entities, removes HTML tags, links, punctuation and Reuters markers, and lowercases text. Digits are retained. Normalized bodies are hashed; repeated bodies are removed before splitting. Any group with contradictory labels is excluded. Titles are combined with cleaned bodies for classification.')
    p('CSV loading uses 5,000-row chunks. Class labels use int8 and word counts int32; feature matrices remain sparse float32. Parquet is written when pyarrow is available. Deduplication records and the full cleaned training corpus remain in RAM, so this is memory-aware batch processing, not fully out-of-core learning.')
    p('The assignment recommends 500,000 rows OR 200 MB. The measured flag above uses uncompressed CSV bytes. If both thresholds are unmet, an instructor-approved exception is needed; duplicates must not be added to inflate volume.','Small2')
    picture('overview.png',490,180)
    page()
    heading('2. Experimental design')
    p('The real-data run uses an approximately 70/15/15 group split with seed 42. Long normalized titles and 80-word body prefixes define duplicate families. Synthetic tests use stratification. The vocabulary, IDF weights and SVD projection are fitted on train only [3,4]. The validation set selects the method; test labels are used only for final reporting. Subject, date and source are not classifier features.')
    table([['Method','Representation','Estimator'],['Baseline','No text features','Most-frequent class'],['BoW','1-2-grams; counts; L2 normalization','Logistic Regression'],['TF-IDF','1-2-grams; sublinear TF; L2','Logistic Regression'],['LSA embeddings','TF-IDF -> SVD, up to 128 dimensions -> L2','Logistic Regression'],['MiniLM (optional)','Frozen all-MiniLM-L6-v2; mean of token chunks','Logistic Regression']],[85,280,125])
    p('The offline vectorizers use max_features=30,000 and min_df=2. All logistic regression models use C=1, liblinear, max_iter=2,000 and seed=42. These settings are fixed in advance rather than optimized against test results. LSA is a learned linear dense embedding, not a pretrained neural representation.')
    p('For the optional MiniLM experiment, each document is tokenized and divided into chunks that fit the encoder. Chunk embeddings are averaged and normalized; only the logistic classifier is trained [5]. The first run requires downloading weights. This run includes: '+', '.join(r['model'].upper() for r in rows)+'.')
    p('E5 uses frozen multilingual-e5-small, a query prefix, masked mean pooling and L2 normalization. The first 128 tokens yield 384-dimensional vectors; lexical methods use full text. External encoder weights are required for inference.')
    p('Evaluation criteria','Sub2')
    p('Accuracy is the fraction of correct predictions. Fake-class precision = TP/(TP+FP); recall = TP/(TP+FN). F1 = 2PR/(P+R), and macro-F1 averages the F1 values of the two classes. ROC-AUC measures ranking across thresholds. Undefined precision/recall values are set to zero.')
    p('Core workers use one numerical computation thread. E5 uses four CPU threads and a 128-token budget. Concurrent work may affect timings. Fit time includes representation fitting and classifier training; inference time includes test transformation. RSS is sampled every 20 ms and includes Python, data and libraries. It can miss short peaks. Timing is a single run, not a statistical benchmark. MiniLM first-load/download costs are included in fit time; external encoder weights are excluded from joblib size.')
    page()
    heading('3. Relational model and SQL')
    picture('er_diagram.png',490,245)
    p('SQLite enforces foreign keys on creation. Subject names and article hashes are unique; a prediction is uniquely identified by (model, article_id). Each output folder contains a separate database, so model names are unique within that experimental run.')
    table([['Query','Purpose / SQL feature'],['01 - 03','Class balance, subject distribution, monthly counts; GROUP BY, JOIN, HAVING'],['04','Articles longer than the corpus average; scalar subquery'],['05','Top three longest articles per subject; CTE + ROW_NUMBER'],['06 - 08','Model errors, confusion counts, error rate per subject; JOIN + CTE'],['09','Cumulative article volume by month; SUM OVER'],['10','Ranking by validation macro-F1; DENSE_RANK']],[100,390])
    p('All ten SELECT queries are executed with pandas.read_sql_query and exported as CSV. Full queries and CREATE TABLE statements are in sql/queries.sql and sql/schema.sql. ROW_NUMBER gives a sequential rank within each subject; DENSE_RANK assigns equal ranks to tied validation scores [6].')
    p('Example interpretation: query 08 joins predictions to articles and subjects, aggregates mistakes, and computes errors/n. This reveals category-specific failures that an overall accuracy can hide. Groups with few examples should be interpreted cautiously.')
    page()
    heading('4. Measured comparison')
    if demo:
        p('SYNTHETIC CHECK ONLY. The examples deliberately have obvious class-specific wording. These numbers validate execution, storage and plotting; they cannot support claims about detecting real misinformation.','Sub2')
    table([['Model','Val F1','Test F1','Accuracy','Fit (s)','RSS (MiB)']]+[[r['model'],f'{r["validation"]["macro_f1"]:.4f}',f'{r["test"]["macro_f1"]:.4f}',f'{r["test"]["accuracy"]:.4f}',f'{r["fit_seconds"]:.3f}',f'{r["peak_rss_mib"]:.1f}'] for r in rows],[80,70,70,80,90,100])
    picture('comparison.png',490,190)
    p(f'Selected method: {best["model"].upper()}, based on validation macro-F1 = {best["validation"]["macro_f1"]:.4f}. Ties use the predefined model order. Test results never determine the selection.')
    p(f'For the selected method, fake-class precision = {best["test"]["precision_fake"]:.4f}, recall = {best["test"]["recall_fake"]:.4f}, ROC-AUC = {best["test"]["roc_auc"]:.4f}; test inference = {best["predict_ms_per_article"]:.3f} ms/article, and the saved classifier file is {best["model_mib"]:.3f} MiB.')
    if all('macro_f1_ci' in r for r in rows):
        p('Conditional 95% macro-F1 bootstrap intervals (1,000 stratified article resamples): '+ '; '.join(f"{r['model']}: {r['macro_f1_ci']['low']:.4f}-{r['macro_f1_ci']['high']:.4f}" for r in rows))
    p('The choice of representation should consider error type, runtime, resources and robustness together. A higher score on this split does not prove that a method generalizes to new publishers or future events. The raw per-method JSON files retain additional metrics and confusion matrices.')
    page()
    heading('5. Error analysis')
    picture('confusion.png',490,150)
    p('Rows are actual labels and columns predicted labels. A false positive incorrectly flags a real article; a false negative lets a fake article pass. The consequences differ, so both counts matter.')
    cm=best['test']['confusion_matrix']
    table([['Selected method','TN','FP','FN','TP'],[best['model'],cm[0][0],cm[0][1],cm[1][0],cm[1][1]]],[170,80,80,80,80])
    error_path=run/f'errors_{best["model"]}.csv'
    errors=pd.read_csv(error_path)
    if errors.empty:
        p('No mistakes were observed for the selected method on this test partition. This is not evidence of perfect real-world reliability. In a synthetic run it mainly reflects the deliberately easy templates. Other models may still have errors; inspect their separate CSV files.')
    else:
        p('Examples below are the highest-confidence observed mistakes. Their causes are hypotheses to investigate manually, not conclusions inferred solely from confidence.')
        for r in errors.head(3).itertuples():
            p(f'<b>Article {r.article_id} / {escape(r.error_type)}</b>: {escape(str(r.title)[:170])}<br/>Actual={r.label}, predicted={r.predicted}, fake score={r.probability_fake:.3f}.<br/>{escape(str(r.text)[:240])}...','Small2')
    p('Reusable analysis outputs','Sub2')
    p('errors_*.csv stores mistakes, labels, scores and text; length_errors_*.csv reports error rates for four length ranges. SQL query 08 measures category error rates. terms_bow.csv and terms_tfidf.csv list strongly weighted features; coefficients describe model association, not causal evidence.')
    p('Manual review should check source-specific phrases, sensational language, quotations, mixed factual/opinion content and unusually short text. Removing Reuters markers alone cannot remove all publisher signals. Detected title/prefix families are already isolated. Run newscheck.robustness for repeated grouped splits and a later-period holdout; an independent external-source test remains future work.')
    page()
    heading('6. Reproducibility, limitations and team')
    p('The command python main.py run --raw data/raw creates a fresh run directory, saves file hashes, records package versions, fits all offline methods and generates every table and figure. The --include-e5 flag adds frozen neural embeddings; --include-minilm remains an alternative. Failed runs retain status=failed and are rejected for inference/reporting.')
    p('Software validation covers required CSV columns, blank bodies, duplicates across chunks, conflicting labels, deterministic splits, train-only vocabulary, SQL foreign keys, all analytical queries and end-to-end training/prediction. A synthetic fixture is used for software tests only; the actual test output is recorded separately in the delivered validation log.')
    p('Limits of inference','Sub2')
    for text in meta['limitations']:
        p('- '+escape(text))
    if demo:
        p('Outstanding empirical work: obtain the official files; execute the real-data command; inspect the resulting errors and volume flag; regenerate the documents and replace this demonstration report before submission. No Kaggle quality claim is supported by this delivery.')
    p('Proposed division of team responsibilities','Sub2')
    table([['Team member','Responsibility for preparation and defence'],[AUTHORS[0],'Application integration, interface and demonstration'],[AUTHORS[1],'Database design, SQL and data-quality analysis'],[AUTHORS[2],'Model comparison, metrics and error interpretation']],[170,320])
    p('This is a proposed allocation, not a verified record of work already performed. Replace it with the actual contributions before submission. All members should reproduce the experiment and understand the main modules.','Small2')
    page()
    heading('7. Conclusions and references')
    if demo:
        p('The implemented system provides a reproducible workflow connecting Python text processing, relational analytics and model comparison. The synthetic execution demonstrates that these components run together. A substantive conclusion about the most suitable representation for real news must await the Kaggle experiment.')
    else:
        p(f'On this fixed split, validation selected {best["model"].upper()}; its test macro-F1 was {best["test"]["macro_f1"]:.4f}. The result supports this method as an in-dataset choice under the recorded configuration. It does not establish factual correctness, cross-source performance or reliability on current news.')
    p('Requirements coverage','Sub2')
    p('The project implements more than five functions, lists/dictionaries, branching, loops, file persistence, exceptions, a console menu and multiple modules. It compares a baseline with three representations, measures quality/time/memory, and provides a normalized SQLite database, ER diagram, ten SQL queries and Pandas exports. Dataset volume is measured explicitly; the recommendation may require an approved exception.')
    p('References','Sub2')
    for i,(title,url) in enumerate(REFERENCES,1):
        p(f'[{i}] {escape(title)}<br/><link href="{escape(url)}" color="#0b8b80">{escape(url)}</link>','Small2')
    p('Source descriptions and software documentation accessed during project preparation, November 2026. Dataset counts in this report come from local file processing, not copied descriptions.','Small2')
    def footer(c,doc):
        c.setStrokeColor(TEAL)
        c.line(48,38,A4[0]-48,38)
        c.setFont('Helvetica',8)
        c.setFillColor(NAVY)
        c.drawString(48,25,'FAKE NEWS DETECTION | AAI-2602'+(' | SYNTHETIC DEMO' if demo else ''))
        c.drawRightString(A4[0]-48,25,str(doc.page))
    SimpleDocTemplate(str(run/'Report.pdf'),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=44,bottomMargin=52,title='Fake News Detection - Project Report',author=', '.join(AUTHORS)).build(story,onFirstPage=footer,onLaterPages=footer)


def build_slides(run,meta):
    width,height=960,540
    c=canvas.Canvas(str(run/'Presentation.pdf'),pagesize=(width,height))
    c.setTitle('Fake News Detection - AAI-2602')
    c.setAuthor(', '.join(AUTHORS))
    demo=meta['dataset_kind']=='synthetic_demo'
    rows=meta['metrics']; data=meta['data']
    n=0
    def text(value,x,y,w=850,size=21,bold=False,color=NAVY):
        style=ParagraphStyle('slide',fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,leading=size*1.3,textColor=color)
        para=Paragraph(value,style)
        _,h=para.wrap(w,height)
        para.drawOn(c,x,height-y-h)
        return h
    def start(title):
        nonlocal n
        n+=1
        c.setFillColor(colors.white);c.rect(0,0,width,height,fill=1,stroke=0)
        c.setFillColor(TEAL);c.rect(48,height-44,56,5,fill=1,stroke=0)
        text(title,48,65,size=30,bold=True)
        c.setFont('Helvetica',9);c.setFillColor(NAVY)
        c.drawString(48,22,'AAI-2602 | '+('SYNTHETIC DEMO - NOT KAGGLE RESULTS' if demo else 'KAGGLE CSV EXPERIMENT'))
        c.drawRightString(912,22,f'{n:02d}')
    def lines(items,y=145,size=22):
        for item in items:
            used=text(item,52,y,850,size=size)
            y+=used+22
    def image(name,x=48,y=140,w=864,h=330):
        c.drawImage(str(run/'figures'/name),x,height-y-h,width=w,height=h,preserveAspectRatio=True,anchor='c',mask='auto')
    def end():
        c.showPage()
    start('Fake News Detection')
    text('Comparing text representations<br/>with Python and SQLite',52,145,size=32)
    text('Kurmet Adilzhan  /  Tolegenov Nursultan<br/>Issakulov Zeiin',52,290,size=21)
    text('Programming for Data Analysis and Database<br/>Applied Artificial Intelligence',52,380,size=18,color=TEAL)
    end()
    start('What the project tests')
    lines(['Can text representations improve over a majority-class baseline?',
           'Compare quality, runtime and process memory on the same split.',
           'Connect predictions to articles through SQL for error analysis.',
           'A text classifier identifies patterns; it does not verify facts.'])
    end()
    start('Data and current execution')
    lines([f'{data["raw_rows"]:,} input rows -> {data["clean_rows"]:,} retained articles.',
           f'Input size: {data["raw_bytes"]/1e6:.3f} MB. Volume guideline met: {data["volume_threshold_met"]}.',
           'True.csv = real (0); Fake.csv = fake (1).',
           'This run uses synthetic software-test data. Real Kaggle evaluation is pending.' if demo else 'Counts, file hashes and cleaning decisions are recorded in the manifest.'],size=23)
    end()
    start('Remove leakage before training')
    lines(['Normalize text and remove explicit Reuters markers.',
           'Deduplicate normalized bodies; exclude conflicting labels.',
           'Real data: grouped train / validation / test, approximately 70% / 15% / 15%.',
           'Fit vocabulary, IDF and SVD on train only. Keep metadata out of features.'],size=22)
    end()
    start('Four representations, one comparison')
    lines(['Baseline: always predict the most frequent training class.',
           'BoW + LR: normalized word and bigram counts.',
           'TF-IDF + LR: reduce the weight of common terms.',
           'LSA + LR: dense vectors from train-only TF-IDF/SVD.',
           'Optional MiniLM + LR: frozen pretrained encoder, averaged token chunks.'],y=135,size=21)
    end()
    start('Relational data model')
    image('er_diagram.png',y=120,h=345)
    end()
    start('Ten SQL queries explain the results')
    lines(['GROUP BY / HAVING: class balance and category counts.',
           'JOIN: connect predictions, articles and subjects.',
           'CTE + ROW_NUMBER: longest articles within each category.',
           'SUM OVER and DENSE_RANK: cumulative volume and validation ranking.',
           'Every query is exported to Pandas and CSV.'],y=130,size=21)
    end()
    start('Measured quality and fit time')
    image('comparison.png',y=125,h=310)
    text('Synthetic scores test the pipeline only; they do not estimate real-world accuracy.' if demo else 'Selection uses validation macro-F1; the chart reports held-out test macro-F1.',52,450,size=16,color=ORANGE)
    end()
    start('Inspect mistakes, not only accuracy')
    image('confusion.png',y=135,h=230)
    text('False positive: real -> fake. False negative: fake -> real.<br/>Review error examples, category rates and length buckets.',52,390,size=21)
    end()
    start('Reproducible and memory-aware')
    lines(['Chunked CSV input; int8 labels; sparse float32 features.',
           'Fresh process per model; elapsed time and sampled RSS.',
           'File hashes, seed, versions and split labels saved.',
           'Cleaned text remains in RAM: training is not fully out-of-core.'],size=23)
    end()
    start('Team preparation and demonstration')
    lines(['Kurmet Adilzhan: integration, interface and demonstration.',
           'Tolegenov Nursultan: database, SQL and data quality.',
           'Issakulov Zeiin: models, metrics and error interpretation.',
           'Proposed allocation: replace with actual contributions before submission.'],size=22)
    end()
    start('Conclusion and next evidence')
    lines(['Python processing, SQLite and model comparison run as one workflow.',
           'Kaggle evaluation remains pending; synthetic results are not scientific findings.' if demo else 'The selected model is '+meta['selected_model'].upper()+'. Its performance is limited to this dataset and split.',
           'External-source and temporal tests are needed to assess generalization.',
           'Sources: Kaggle/ISOT, scikit-learn, Sentence Transformers and SQLite. Full links are in the report.'],size=22)
    end()
    c.save()
