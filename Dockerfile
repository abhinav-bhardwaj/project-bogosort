FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    vaderSentiment \
    matplotlib \
    flask \
    imageio \
    shap \
    gunicorn

COPY . .

EXPOSE 7860

CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120"]
