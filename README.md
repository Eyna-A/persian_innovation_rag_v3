# Persian Innovation FAQ RAG — بدون Chunking

یک RAG فارسی برای پرسش‌وپاسخ‌های صندوق نوآوری و شکوفایی. در این نسخه **هر ردیف CSV دقیقاً یک سند در Weaviate است**؛ چون سؤال‌وجواب‌ها واحدهای معنایی کوتاه و مستقل‌اند، تقسیم متن به chunk انجام نمی‌شود.

## معماری

1. خواندن و اعتبارسنجی CSV (`category`, `question`, `answer`)
2. ساخت متن قابل جستجو از دسته‌بندی + سؤال + پاسخ، بدون تقسیم متن
3. ساخت embedding محلی با `BAAI/bge-m3`
4. ذخیره هر FAQ و بردار آن در Weaviate
5. بازیابی Hybrid (BM25 + برداری)؛ `HYBRID_ALPHA=0` فقط BM25، مقدار `1` فقط برداری، و `0.5` ترکیبی
6. مرتب‌سازی مجدد اختیاری با `BAAI/bge-reranker-v2-m3`
7. ساخت prompt دقیق و تولید پاسخ فارسی با API سازگار با OpenAI؛ دما پایین برای پاسخ‌های factual
8. ارزیابی بازیابی با Precision@K، Recall@K و MRR

## پیش‌نیازها

- Python 3.11 یا جدیدتر
- Docker Desktop
- اتصال اینترنت برای دانلود مدل‌های embedding/reranker در اجرای اول
- یک کلید API برای مدل زبانی (مثلاً Google AI Studio)

## راه‌اندازی در Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

فایل `.env` را باز کن و `OPENAI_API_KEY` را با کلید واقعی جایگزین کن. مقادیر پیش‌فرض برای endpoint سازگار با OpenAI در Google AI Studio تنظیم شده‌اند؛ اگر ارائه‌دهنده دیگری داری، `OPENAI_BASE_URL` و `LLM_MODEL` را مطابق مستندات همان ارائه‌دهنده تغییر بده.

## اجرای Weaviate

```powershell
docker compose up -d
docker compose ps
```

## ورود اطلاعات به پایگاه داده

```powershell
python rag.py ingest
```

اگر می‌خواهی Collection قبلی را حذف و از نو بسازی:

```powershell
python rag.py ingest --reset
```

**هشدار:** `--reset` تمام داده‌های Collection انتخاب‌شده را پاک می‌کند و دوباره از CSV می‌سازد.

## پرسیدن سؤال یا چت

```powershell
python rag.py ask "صندوق نوآوری چه خدماتی ارائه می‌دهد؟"
python rag.py chat
```

در حالت چت، برای خروج `exit` و برای پاک کردن تاریخچه `/clear` را بنویس. تاریخچه فقط برای پیوستگی گفتگو استفاده می‌شود؛ retrieval بر اساس سؤال فعلی انجام می‌شود.

## فعال کردن reranker

در `.env` مقدار زیر را تنظیم کن:

```env
ENABLE_RERANKER=true
```

سپس سؤال را دوباره اجرا کن. مدل reranker ممکن است دانلود نسبتاً بزرگی داشته باشد. ابتدا بدون reranker اجرا و ارزیابی کن؛ بعد نتیجه‌ها را مقایسه کن.

## ارزیابی Retrieval

در `evaluation/evaluation_dataset.csv` مقدار `expected_source_question` را با متن دقیق سؤال متناظر در `data/innovation_faq.csv` پر کن. سپس:

```powershell
python evaluate.py
```

این اسکریپت Precision@K، Recall@K و MRR را با چند مقدار K چاپ می‌کند. برای هر query یک FAQ مرجع در نظر گرفته شده؛ بنابراین Recall در این نسخه یعنی آیا سند مرجع در top-K پیدا شده است. این ارزیابی، درستی پاسخ تولیدشده توسط LLM را اندازه نمی‌گیرد.

## نکات مهم

- Embeddingها به‌صورت محلی محاسبه می‌شوند؛ API فقط برای تولید پاسخ نهایی استفاده می‌شود.
- در محیط محلی، Weaviate با دسترسی anonymous اجرا می‌شود؛ این تنظیم را برای سرور عمومی استفاده نکن.
- فایل `.env` و کلید API را در Git قرار نده.
- کد و syntax بررسی می‌شوند، اما اجرای end-to-end به Docker فعال، دانلود مدل‌ها و API key معتبر نیاز دارد.
