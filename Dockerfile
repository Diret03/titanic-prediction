FROM python:3.12
LABEL authors="Diret"

WORKDIR /flask-app

RUN pip install --upgrade pip
COPY requirements.txt .

RUN pip install flask
RUN pip install tensorflow
RUN pip install scikit-learn
RUN pip install joblib
RUN pip install numpy

# install Node.js and npm for Tailwind CSS
RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*

# Copy and install Node.js dependencies for Tailwind CSS
COPY package.json .
COPY package-lock.json .
RUN npm install

#Copy the entire application to the working directory
COPY . .


EXPOSE 5000

#default command to run flask app
CMD ["python", "app.py"]

