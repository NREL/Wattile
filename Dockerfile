FROM ghcr.io/haxall/hxpy:latest

RUN mkdir /wattile

# Copy wattile and models
COPY ./wattile /wattile/wattile
COPY ./tests/fixtures /trained_models

ENV PYTHONPATH=${PYTHONPATH}:${PWD}

# Install Dependices
COPY ./poetry.lock /wattile
COPY ./pyproject.toml /wattile
RUN pip3 install poetry==1.5.1
RUN poetry config virtualenvs.create false
RUN cd /wattile && poetry install --no-dev

# use non root user. 
RUN groupadd -r user && useradd -r -g user user
USER user

ENTRYPOINT ["python"]
