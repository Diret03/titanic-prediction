FROM python:3.12
LABEL authors="Diret"

WORKDIR /flask-titanic-app

# Combine all apt-get operations and clean up in a single layer
RUN apt-get update && \
    apt-get install -y nodejs npm --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Combine pip installations to reduce layers
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install flask tensorflow scikit-learn joblib numpy

COPY package.json .
COPY package-lock.json .
RUN npm install

# Copy the rest of the application
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]