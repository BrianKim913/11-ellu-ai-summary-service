FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn
WORKDIR /app
COPY ./ /app/
EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
