# Fake News Detection - AAI-2602

**Kurmet Adilzhan, Tolegenov Nursultan, Issakulov Zeiin**  
Programming for Data Analysis and Database | Applied Artificial Intelligence

Python-проект: обработка новостей, SQLite, 10 SQL-запросов, четыре подхода к классификации, анализ ошибок, консольное меню и автоматически обновляемые PDF-отчёт/презентация.

## Статус поставки

Код проверен на **синтетическом наборе для тестирования программы**. Файлы Kaggle не включены: скачать их в среде разработки не удалось, сайт запросил вход. Приложенные PDF демонстрируют выполненный тестовый запуск, **не являются результатами исследования на Kaggle и не готовы к сдаче как такое исследование**. После загрузки реальных CSV команда ниже создаёт отдельную базу, обученные модели, настоящие метрики, графики, отчёт и презентацию. Никакие числа качества не выдуманы.

## Быстрый запуск (Python 3.10 или новее)

Откройте терминал в папке, где расположен `main.py`.

```bash
python -m venv .venv
```

Windows PowerShell: `.venv\Scripts\Activate.ps1`  
Windows CMD: `.venv\Scripts\activate.bat`  
macOS/Linux: `source .venv/bin/activate`

```bash
python -m pip install -r requirements.txt
python main.py
```

Если активация PowerShell запрещена, используйте `.venv\Scripts\python.exe` вместо `python` без изменения системных политик.

Меню: 1 - обучение на CSV; 2 - демонстрация; 3 - классификация; 4 - SQL-анализ; 5 - обновление PDF; 0 - выход.

## Реальный эксперимент

1. Скачать и распаковать датасет: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Поместить `Fake.csv` и `True.csv` в `data/raw/`. Ожидаемые колонки: `title,text,subject,date`.
3. Выполнить:

```bash
python main.py run --raw data/raw
```

Альтернатива загрузки при настроенном Kaggle API:

```bash
python -m pip install kaggle
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw --unzip
```

Учётные данные Kaggle настраиваются по официальной документации; токены нельзя помещать в GitHub. На странице датасета указана лицензия CC BY-NC-SA 4.0: сохраняйте атрибуцию и проверьте условия перед распространением самих данных. Репозиторий содержит ссылку, а не исходные статьи.

## BoW, TF-IDF и embeddings

| Метод | Представление | Классификатор |
|---|---|---|
| baseline | Нет текстовых признаков | Most-frequent DummyClassifier |
| bow | Частоты 1-2-грамм, L2-нормализация | Logistic Regression |
| tfidf | TF-IDF 1-2-грамм, sublinear TF | Logistic Regression |
| lsa | TF-IDF -> TruncatedSVD, до 128 измерений -> L2 | Logistic Regression |
| minilm (дополнительно) | Замороженный all-MiniLM-L6-v2, усреднение чанков | Logistic Regression |

**LSA - плотные латентно-семантические эмбеддинги, а не нейросеть и не pretrained transformer.** Они обучаются только на train. В таблицах названия различаются. Если преподаватель подразумевает под embeddings именно нейросетевые представления, запускайте MiniLM:

```bash
python -m pip install -r requirements-embeddings.txt
python main.py run --raw data/raw --include-minilm
```

Первый запуск MiniLM требует интернет для весов; CPU-обработка полного датасета может занимать заметное время. В отчёте явно перечислены реально выполненные модели. При ошибке MiniLM весь запуск помечается `failed`, успешное завершение не имитируется. Скачанные веса кэшируются стандартным механизмом Hugging Face; для офлайн-предсказаний кэш должен сохраниться. Размер файла `.joblib` не включает эти веса.

## Демонстрация и проверка

```bash
python main.py run --demo
python -m unittest discover -s tests -v
```

Демонстрация создаёт 240 **искусственных** текстов по шаблонам. Высокие показатели на них проверяют лишь связность программного конвейера. Это не репрезентативная выборка и не доказательство детекции фейков.

Каждый запуск создаёт новую папку `runs/<UTC timestamp>_<dataset kind>_<id>/`. Скопируйте путь из вывода программы:

```bash
python main.py predict --run "runs/ИМЯ_ПАПКИ" --text "The agency published official figures after the public meeting."
python main.py sql --run "runs/ИМЯ_ПАПКИ"
python main.py report --run "runs/ИМЯ_ПАПКИ"
```

Классификация предназначена для английского текста. Метка - прогноз на основе паттернов датасета; `probability_fake` - некалиброванный модельный балл, не вероятность фактической ложности утверждения. Для проверки конкретной новости нужны источники и фактчекинг. Загружайте только собственные доверенные `.joblib` файлы.

## Что создаёт программа

- `run.json`: статус, версии библиотек, оборудование ОС, ограничения, выбранная модель, метрики.
- `data_manifest.json`: реальные размеры CSV, SHA-256, число строк, удалений, классов и частей выборки.
- `articles.csv`, опционально `articles.parquet`: очищенные записи с идентификаторами и split.
- `news.db`: SQLite с внешними ключами и индексами.
- `comparison.csv`, `metrics_*.json`: сравнение качества, времени и RSS.
- `predictions_*.csv`, `errors_*.csv`, `length_errors_*.csv`: предсказания и анализ ошибок.
- `terms_*.csv`: коэффициенты слов BoW/TF-IDF, а не причинные объяснения.
- `sql/`: 10 результатов, исходные запросы и схема SQL.
- `figures/`: баланс классов, длины текстов, сравнение, confusion matrices, ER-диаграмма.
- `Report.pdf`, `Presentation.pdf`: автоматически обновлённые документы для этого запуска.
- `models/`, `worker_*.log`: модели и логи.

CSV-таблицы и метрики содержат `dataset_kind` и `run_id`, чтобы демонстрационные результаты нельзя было спутать с реальными при переносе файла. `artifact_status` обозначает завершённый этап экспорта; актуальный общий статус всего запуска хранится в `run.json`. В тестах проверены четыре офлайн-метода; MiniLM реализован, но в данной среде не запускался из-за отсутствия библиотеки/весов.

## Методология

`0 = real`, `1 = fake`. Пустые тела удаляются. Дубликаты определяются по SHA-256 нормализованного тела **до** разделения. Все группы с противоречивыми метками исключаются. Деление stratified 70/15/15, seed=42; после округления доли могут немного отличаться. Словарь, IDF и SVD обучаются только на train. Subject/date/source не входят в признаки; маркер Reuters удаляется. Настройки фиксированы, поиска гиперпараметров нет. Победитель выбирается по validation macro-F1; test используется для итогового сравнения. При равенстве сохраняется порядок методов. После просмотра test нельзя подбирать настройки на его результатах.

Метрики: accuracy, precision/recall для fake, macro-F1, ROC-AUC. Fit time включает построение представления и классификатора; для MiniLM также первый запуск/загрузку энкодера. Predict time включает векторизацию теста. Одно измерение времени на метод, один seed. Модели запускаются в отдельных процессах с одним вычислительным потоком; RSS измеряется каждые 20 мс и включает интерпретатор и данные. Это приближённый пик процесса, не точный размер модели. Файл модели измеряется отдельно. Сравнение времени MiniLM без прогрева не равно сравнению только алгоритмической скорости.

Чтение CSV использует `chunksize=5000`; метки int8, счётчик слов int32, разреженные матрицы float32 и максимум 30 000 признаков. Полный очищенный текст и словарь дедупликации остаются в RAM: это **не** полностью out-of-core обучение. Parquet сохраняется при установленном pyarrow. Для намного большего корпуса потребуются SQL-дедупликация и incremental learning.

Рекомендуемый порог задания: >=500 000 строк **или** >=200 000 000 байт исходных CSV. Флаг `volume_threshold_met` вычисляется по реальным файлам. При невыполнении порога согласуйте исключение; искусственное дублирование не применяется.

## Архитектура

| Файл | Ответственность |
|---|---|
| main.py | CLI и интерактивное меню, обработка ошибок |
| newscheck/data.py | Чтение, очистка, дедупликация, разбиение, manifest |
| newscheck/database.py | ER-схема, внешние ключи, SQL, Pandas-экспорт |
| newscheck/models.py | Baseline, BoW, TF-IDF, LSA и MiniLM |
| newscheck/worker.py | Обучение, метрики, память, ошибки, сохранение модели |
| newscheck/workflow.py | Изолированные процессы, run status, выбор модели |
| newscheck/charts.py | Графики Matplotlib |
| newscheck/reporting.py | PDF-отчёт, PDF-презентация, ER-диаграмма |
| tests/ | Проверки ошибок ввода, утечек, SQL и полного процесса |

## GitHub и Moodle

Создайте репозиторий в своём аккаунте и загрузите исходники; можно через интерфейс GitHub или команды ниже. Учётная запись GitHub здесь не подключена, поэтому удалённый репозиторий автоматически не создан.

```bash
git init
git add .
git commit -m "Add reproducible fake news detection project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fake-news-detection.git
git push -u origin main
```

После реального запуска добавьте в репозиторий небольшие таблицы метрик/графики/документы из нужного run, сохранив `run.json` и `data_manifest.json`. Не загружайте большие CSV, базу или модели обычным Git: в задании разрешена ссылка на данные. `runs/` исключён из Git по умолчанию, чтобы случайно не опубликовать полный корпус.

В Moodle загрузите **обновлённые документы реального запуска**, а не приложенные демонстрационные PDF. Срок - за 2-3 дня до вашего экзамена. `docs/defense_notes.md` содержит объяснения и сценарий защиты; распределение вклада в документах предложено для команды и должно соответствовать фактической работе.

## Источники

1. Kaggle data card: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. ISOT data README: https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf
3. scikit-learn feature extraction: https://scikit-learn.org/stable/modules/feature_extraction.html
4. scikit-learn common pitfalls: https://scikit-learn.org/stable/common_pitfalls.html
5. Sentence Transformers: https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
6. SQLite window functions: https://www.sqlite.org/windowfunctions.html
