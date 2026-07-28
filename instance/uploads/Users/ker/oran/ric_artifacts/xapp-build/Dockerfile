FROM python:3.11-slim
LABEL name="oran-autoformer" \
      description="ML xApp auto-built by xApp Builder"
WORKDIR /app
COPY model.pkl /app/model.pkl
COPY xapp_main.py /app/xapp_main.py
ENV MODEL_PATH=/app/model.pkl
CMD ["python3", "/app/xapp_main.py"]
