FROM python:3.9

WORKDIR /app

COPY ../datasets/ ./datasets

COPY ./src ./src

COPY ../requirements.txt .

RUN pip3 install -r requirements.txt

CMD ['python3', '-m', 'src.scripts.ml.train_optimize_save_models']