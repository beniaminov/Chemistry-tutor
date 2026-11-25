# Локальная RAG-система: установка и запуск

## Выбранная конфигурация

- Разрешённый корпус: только `учебники/5eb6e10c5b_himija_8_klass_o_s_gabrieljan_2025_g.pdf`.
- Embeddings: локальная `intfloat/multilingual-e5-base`, 768 измерений, cosine.
- Хранилище: Qdrant embedded/persistent в `.local-data/qdrant`; Docker и открытый Qdrant-порт не нужны.
- Поиск: dense Qdrant + BM25 по локальным текстам точек, объединение Reciprocal Rank Fusion (RRF).
- Генерация: локальная Qwen2.5-1.5B-Instruct GGUF Q4_K_M через `llama-cpp-python` на CPU.
- OCR: локальный Tesseract 5, языки rus+eng; применяется к страницам без текстового слоя.
- В LLM prompt кодом включается не более трёх результатов RRF; ограничение проверяется до генерации.
- После ответа добавляется детерминированная строка источников со страницами переданных фрагментов.
- DeepSeek и любые внешние LLM API не используются. Интернет нужен только для установки Python-пакетов и однократной загрузки весов.

Эта реализация — CLI-прототип и пока не подключена к сайту. Qwen 1.5B выбрана с учётом CPU и 8 ГБ RAM; квантизация уменьшает требования к памяти, но возможны ошибки и упрощённые объяснения. Проверяйте ответы по ссылкам на страницы учебника.

## Установка на Windows

Запускайте команды в PowerShell из корня проекта. Требуется Python 3.12 x64. Окружение `.venv-e5` уже содержит E5; дополните его зависимостями приложения:

```powershell
.\.venv-e5\Scripts\python.exe -m pip install -r requirements-local-rag.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Скачайте зафиксированную Qwen GGUF-модель в игнорируемый Git каталог. `huggingface_hub` установлен как зависимость E5 и умеет возобновлять скачивание:

```powershell
New-Item -ItemType Directory -Force .local-models | Out-Null
.\.venv-e5\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='Qwen/Qwen2.5-1.5B-Instruct-GGUF', filename='qwen2.5-1.5b-instruct-q4_k_m.gguf', revision='91cad51170dc346986eccefdc2dd33a9da36ead9', local_dir='.local-models'))"
```

Файл модели — 1,117,320,736 байт (около 1.04 ГиБ), SHA-256 `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`. Скрипт проверяет точный размер. E5 загружается из локального Hugging Face cache с закреплённой ревизией; если кэш недоступен, сначала выполните инструкцию в [Локальной E5](Локальная-E5.md).

В выбранном PDF 176 страниц-сканов без встроенного текста, поэтому для работы индекса требуется Tesseract и языковые данные. Ниже установщик Tesseract 5.5.3 проверяется по закреплённому SHA-256 и ставится в пользовательский каталог проекта:

```powershell
New-Item -ItemType Directory -Force .local-runtime | Out-Null
$installer = Join-Path (Get-Location) '.local-runtime\tesseract-setup.exe'
curl.exe -L --fail --retry 3 -o $installer 'https://github.com/tesseract-ocr/tesseract/releases/download/5.5.3/tesseract-ocr-w64-setup-5.5.3.20260724.exe'
$expected = 'bee9e3434bd94fd65387d9be28cd467a41f61b1275383b55b0f59a1331270ae4'
if ((Get-FileHash $installer -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'Tesseract installer checksum mismatch' }
$target = Join-Path (Get-Location) '.local-runtime\tesseract'
Start-Process -FilePath $installer -ArgumentList @('/S',"/D=$target") -Wait
New-Item -ItemType Directory -Force "$target\tessdata" | Out-Null
curl.exe -L --fail --retry 3 -o "$target\tessdata\rus.traineddata" 'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/rus.traineddata'
curl.exe -L --fail --retry 3 -o "$target\tessdata\eng.traineddata" 'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/eng.traineddata'
```

Этот setup устанавливает runtime под `.local-runtime`, не в системный каталог; перед установкой SHA-256 проверяется по GitHub release metadata. Не запускайте неофициальные установщики. После загрузки весов E5 и Qwen, языковых данных OCR и установки пакетов runtime работает с локальными файлами.

## Индексация и вопрос

```powershell
.\.venv-e5\Scripts\python.exe scripts\local_rag.py index
.\.venv-e5\Scripts\python.exe scripts\local_rag.py ask "Что такое атом?"
```

`index` обрабатывает только конкретный allowlisted PDF, запускает OCR для страниц без текстового слоя, сохраняет страницы, SHA-256 исходника и стабильные ID фрагментов. OCR выполняется на CPU (180 DPI); пустые после OCR страницы перечисляются в отчёте. Повторный запуск с тем же источником и настройками выполняет upsert; при изменении PDF или конфигурации без `--rebuild` индексация остановится:

```powershell
.\.venv-e5\Scripts\python.exe scripts\local_rag.py index --rebuild
```

Каталог `.local-data/` содержит базу Qdrant и исключён из Git; не удаляйте его, если нужно сохранить индекс. Вопрос загружается через локальную E5, dense и BM25 ранжируют кандидатов, RRF выбирает не более трёх фрагментов. CLI показывает страницы, затем запускает локальную Qwen и печатает ответ.

## Проверки

Проверить синтаксис и зависимости:

```powershell
.\.venv-e5\Scripts\python.exe -m py_compile scripts\local_rag.py
.\.venv-e5\Scripts\python.exe -c "import pymupdf, qdrant_client, rank_bm25, llama_cpp; print('Local RAG dependencies OK')"
```

Для запроса проверить строку `Retrieved chunks sent to local LLM` — она не должна превышать 3. Сверяйте сформированный ответ и страницы с учебником: retrieval/generation smoke-test не заменяет экспертную оценку фактической точности.

Первый локальный end-to-end smoke-test по вопросу «Что запрещено делать при работе в химической лаборатории?» прошёл: retrieval выбрал 3 фрагмента (2 со стр. 20, 1 со стр. 174), ответ модели — «В химической лаборатории запрещено пробовать на вкус вещества.» Время retrieval — 12,7 с, генерации — 45,5 с, наблюдавшийся peak RSS процесса — 2044 МиБ. Источники страниц переданных фрагментов добавляются отдельной строкой автоматически. Это один контрольный вопрос, а не оценка общей точности системы.

## Ресурсы и границы

- Ранее измеренный peak RSS E5 smoke-test — около 1.63 ГиБ; он использовал синтетические строки и не включал Qdrant.
- GGUF весит около 1.04 ГиБ на диске; измеренный peak RSS процесса в одном end-to-end запросе — 2044 МиБ. Это не общий peak ОС и не показатель параллельной нагрузки сайта/Qdrant.
- Первичная OCR-индексация всех 176 страниц заняла около 1031 с (17 минут); пик RSS во время индексации отдельно не измерялся.
- OCR, индексация и генерация выполняются на CPU; точность распознавания формул нужно сверять с изображением страницы.
- Вопрос содержит только выбранные фрагменты, но эти данные остаются локально. Не включайте логи полного контекста в общие отчёты.
- PDF может содержать OCR-ошибки; не считать найденный текст достоверным без сверки оригинальной страницы.
