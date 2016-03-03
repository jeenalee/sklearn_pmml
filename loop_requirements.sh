#!/usr/bin/env bash

while read p; do
    pip install $p
done < requirements.txt
