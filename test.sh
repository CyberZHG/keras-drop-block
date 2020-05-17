#!/usr/bin/env bash
pycodestyle --max-line-length=120 keras_drop_block tests && \
    nosetests --with-coverage --cover-html --cover-html-dir=htmlcov --cover-package=keras_drop_block tests
