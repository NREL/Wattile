FROM ghcr.io/haxall/hxpy:latest

RUN mkdir /intelcamp

COPY . /intelcamp

ENV PYTHONPATH=${PYTHONPATH}:${PWD}

RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN cd /intelcamp && poetry install --no-dev

EXPOSE 8888

ENTRYPOINT ["python"]
