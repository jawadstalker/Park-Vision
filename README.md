# Park-Vision — سامانه هوشمند تشخیص جای پارک خالی

سامانه‌ای که با تحلیل تصاویر دوربین‌های ثابت خیابانی، وضعیت اشغال یا خالی
بودن جای‌های پارک حاشیه‌ای را به‌صورت خودکار تشخیص می‌دهد. برخلاف اکثر
پروژه‌های مشابه که برای پارکینگ‌های بالانگر با خط‌کشی ثابت طراحی شده‌اند،
این سامانه برای **پارک حاشیه‌ای خیابانی** (دوربین با زاویه مایل، بدون
خط‌کشی از پیش مشخص) ساخته شده و از الگوریتم **Gap Detection** برای اندازه‌گیری
فاصله واقعی (متر) بین خودروهای متوالی استفاده می‌کند.

## معماری

```
تصویر/ویدیوی دوربین
        │
        ▼
YOLOv8 (تشخیص خودرو)
        │
        ▼
SORT (ردیابی خودرو بین فریم‌ها، فقط برای ویدیو)
        │
        ▼
تبدیل پرسپکتیو (پیکسل → متر واقعی، بر اساس کالیبراسیون هر دوربین)
        │
        ▼
Gap Detection (فاصله بین خودروها → جای خالی/اشغال)
        │
        ▼
FastAPI (POST /detect, POST /calibrate, GET /status)
        │
        ├──▶ Streamlit Dashboard (نمایش لحظه‌ای)
        └──▶ ابزار کالیبراسیون کلیکی
```

## ساختار پروژه

```
Park-Vision/
├── app/
│   ├── main.py             برنامه FastAPI
│   ├── routes.py           سه endpoint: /detect, /calibrate, /status
│   ├── schemas.py          مدل‌های Pydantic
│   ├── calibration.py      ذخیره/بارگذاری homography هر دوربین
│   ├── perspective.py      تبدیل پیکسل به مختصات واقعی
│   ├── vehicle_detector.py پوشش YOLOv8
│   ├── tracker.py           الگوریتم SORT برای ردیابی خودرو در ویدیو
│   ├── gap_detector.py     هسته اصلی: الگوریتم Gap Detection
│   └── visualization.py    توابع رسم برای داشبورد
├── streamlit_app.py         داشبورد نمایش لحظه‌ای (تصویر/ویدیو)
├── calibrate_tool.py         ابزار کلیکی انتخاب نقاط کالیبراسیون
├── process_video.py          CLI برای پردازش آفلاین یک ویدیوی ضبط‌شده
├── dataset_tools/             ابزارهای آماده‌سازی دیتاست و فاین‌تیون
│   ├── extract_frames.py
│   ├── organize_dataset.py
│   ├── dataset_stats.py
│   ├── train.py
│   └── evaluate.py
├── tests/                     ۳۵ تست pytest برای تمام ماژول‌های بالا
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
└── docker-compose.yml
```

## نصب و اجرا (بدون Docker)

```bash
pip install -r requirements.txt
```

**۱. کالیبراسیون دوربین** (یک‌بار به‌ازای هر دوربین، قبل از هر کار دیگه‌ای):

```bash
streamlit run calibrate_tool.py
```
عکس رفرنس دوربین را آپلود کنید، ۴ نقطه با مختصات واقعی متری وارد کنید،
پیش‌نمایش نمای بالا را بررسی و ذخیره کنید.

**۲. اجرای API:**

```bash
uvicorn app.main:app --reload
```
مستندات تعاملی: `http://localhost:8000/docs`

> اولین بار که سرور بالا می‌آید، دانلود و بارگذاری اولیه‌ی PyTorch/YOLO
> ممکن است حدود ۶۰ ثانیه طول بکشد — طبیعی است.

**۳. داشبورد لحظه‌ای:**

```bash
streamlit run streamlit_app.py
```

**۴. پردازش آفلاین یک ویدیوی ضبط‌شده:**

```bash
python process_video.py street_footage.mp4 street-01 results.jsonl
```

## اجرا با Docker

```bash
docker compose up --build
```

سه سرویس بالا می‌آید:

| سرویس | آدرس | توضیح |
|---|---|---|
| `api` | `http://localhost:8000` | FastAPI |
| `dashboard` | `http://localhost:8501` | داشبورد لحظه‌ای |
| `calibrate` | `http://localhost:8502` | ابزار کالیبراسیون |

هر سه سرویس یک `volume` مشترک (`calibrations`) دارند، پس کالیبراسیونی
که در سرویس `calibrate` ذخیره می‌کنید، بلافاصله در `api` و `dashboard`
هم در دسترس است.

## آماده‌سازی دیتاست و فاین‌تیون

جزئیات کامل در [`dataset_tools/README.md`](dataset_tools/README.md).
خلاصه جریان کار:

```bash
python dataset_tools/extract_frames.py video.mp4 raw_frames/ --street azadi --lighting day
# برچسب‌گذاری با LabelImg یا Roboflow
python dataset_tools/dataset_stats.py raw_frames/
python dataset_tools/organize_dataset.py raw_frames/ raw_labels/ dataset/
python dataset_tools/train.py dataset/data.yaml
python dataset_tools/evaluate.py runs/detect/street_parking/weights/best.pt dataset/data.yaml
```

## تست‌ها

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

۳۵ تست، شامل تست واحد الگوریتم Gap Detection، ردیابی SORT، کالیبراسیون،
و تست یکپارچه API با `TestClient`.

## معیارهای پذیرش سند فنی

| معیار | آستانه | ابزار سنجش |
|---|---|---|
| Precision تشخیص خودرو | ≥ ۸۵٪ | `dataset_tools/evaluate.py` |
| Recall تشخیص خودرو | ≥ ۸۰٪ | `dataset_tools/evaluate.py` |
| mAP@0.5 | ≥ ۸۰٪ | `dataset_tools/evaluate.py` |
| زمان پردازش هر فریم (CPU، بدون GPU) | < ۳ ثانیه | `dataset_tools/evaluate.py` |
| دقت تشخیص اشغال/خالی (نور روز) | ≥ ۹۰٪ | نیازمند تست میدانی با داده واقعی |
| دقت تشخیص اشغال/خالی (نور غروب/سایه) | ≥ ۸۰٪ | نیازمند تست میدانی با داده واقعی |

## وضعیت فعلی پروژه

بخش کد و زیرساخت (API، ردیابی، پنل، ابزار کالیبراسیون، ابزارهای دیتاست،
تست‌ها، Docker) کامل و تست‌شده است. مراحل باقی‌مانده — جمع‌آوری واقعی
داده از خیابان‌های مشهد، فاین‌تیون روی داده واقعی، و تست نهایی میدانی —
کار میدانی هستند و باید توسط تیم پروژه انجام شوند.
