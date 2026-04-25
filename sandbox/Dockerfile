FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --gid 1000 sandbox \
    && useradd --uid 1000 --gid sandbox --create-home --shell /usr/sbin/nologin sandbox

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        pandas \
        numpy \
        scikit-learn \
        joblib

WORKDIR /workspace
USER sandbox
