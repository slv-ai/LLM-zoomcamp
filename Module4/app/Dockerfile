FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Run Streamlit
CMD ["python", "-u", "-m", "streamlit", "run", "app.py"]