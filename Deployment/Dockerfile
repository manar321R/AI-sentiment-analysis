# استخدام نسخة بايثون رسمية
FROM python:3.9

# تحديد مجلد العمل داخل السيرفر
WORKDIR /code

# نسخ ملف المكتبات وتثبيتها
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# نسخ باقي ملفات المشروع (الكود، المودل، الواجهة)
COPY . .

# تشغيل سيرفر FastAPI على المنفذ 7860 (المنفذ المعتمد في Hugging Face)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]