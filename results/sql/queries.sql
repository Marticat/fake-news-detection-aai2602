-- 01_class_balance
SELECT label, COUNT(*) AS n, ROUND(AVG(word_count),1) AS mean_words FROM articles GROUP BY label;

-- 02_subjects
SELECT s.name, a.label, COUNT(*) AS n FROM articles a JOIN subjects s USING(subject_id) GROUP BY s.name,a.label HAVING COUNT(*)>=5 ORDER BY n DESC;

-- 03_monthly
SELECT substr(date,1,7) AS month,label,COUNT(*) AS n FROM articles WHERE date IS NOT NULL GROUP BY month,label ORDER BY month,label;

-- 04_long_articles
SELECT article_id,title,word_count FROM articles WHERE word_count>(SELECT AVG(word_count) FROM articles) ORDER BY word_count DESC LIMIT 20;

-- 05_rank_by_subject
WITH ranked AS (SELECT s.name,a.title,a.word_count,ROW_NUMBER() OVER(PARTITION BY a.subject_id ORDER BY a.word_count DESC,a.article_id) AS rank FROM articles a JOIN subjects s USING(subject_id)) SELECT * FROM ranked WHERE rank<=3;

-- 06_model_errors
SELECT p.model,COUNT(*) AS evaluated,SUM(p.predicted<>a.label) AS errors,ROUND(AVG(p.predicted=a.label),4) AS accuracy FROM predictions p JOIN articles a USING(article_id) GROUP BY p.model;

-- 07_confusion
SELECT p.model,a.label AS actual,p.predicted,COUNT(*) AS n FROM predictions p JOIN articles a USING(article_id) GROUP BY p.model,a.label,p.predicted ORDER BY p.model,a.label,p.predicted;

-- 08_error_by_subject
WITH totals AS (SELECT p.model,s.name,COUNT(*) AS n,SUM(p.predicted<>a.label) AS errors FROM predictions p JOIN articles a USING(article_id) JOIN subjects s USING(subject_id) GROUP BY p.model,s.name) SELECT *,ROUND(1.0*errors/n,4) AS error_rate FROM totals ORDER BY error_rate DESC;

-- 09_cumulative_volume
WITH monthly AS (SELECT substr(date,1,7) AS month,COUNT(*) AS n FROM articles WHERE date IS NOT NULL GROUP BY month) SELECT month,n,SUM(n) OVER(ORDER BY month ROWS UNBOUNDED PRECEDING) AS cumulative_n FROM monthly ORDER BY month;

-- 10_model_ranking
SELECT model,validation_f1,test_f1,fit_seconds,peak_rss_mib,DENSE_RANK() OVER(ORDER BY validation_f1 DESC) AS validation_rank FROM experiments ORDER BY validation_rank,model;