FROM python:3.9

WORKDIR /app

COPY ../datasets/ ./datasets

COPY ../trained_models ./trained_models

COPY ../src/ ./src

COPY ../requirements.txt .

COPY flask_server.py .

RUN pip3 install -r requirements.txt

EXPOSE 5000

CMD ['python3', 'flask_server.py']



