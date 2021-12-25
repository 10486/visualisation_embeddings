FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install python3.9 -y
RUN apt-get install python3-pip -y
RUN mkdir application
WORKDIR application
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
RUN rm -rf requirements.txt
RUN pip install waitress
CMD python3 app/load_data_from_mlflow.py; waitress-serve --port=5000 backend:app
