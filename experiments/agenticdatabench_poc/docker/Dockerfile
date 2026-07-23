ARG BASE_IMAGE=da-agent:latest
FROM ${BASE_IMAGE}

USER root

RUN pip install --no-cache-dir \
    dbfread \
    fuzzywuzzy \
    opencv-python-headless \
    pyshp \
    python-Levenshtein \
    thefuzz

ARG HOST_UID=1000
ARG HOST_GID=1000
RUN groupadd --gid "${HOST_GID}" dslighting \
    && useradd --uid "${HOST_UID}" --gid "${HOST_GID}" --create-home dslighting

ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace
USER dslighting

CMD ["sh", "-c", "while :; do sleep 3600; done"]
