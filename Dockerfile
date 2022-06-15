FROM ghcr.io/haxall/hxpy:latest

RUN mkdir /wattile

COPY . /wattile

ENV PYTHONPATH=${PYTHONPATH}:${PWD}

RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN cd /wattile && poetry install --no-dev

EXPOSE 8888

ENTRYPOINT ["python"]
