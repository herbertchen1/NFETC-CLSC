#!/usr/bin/env bash
echo "Downloading word embeddings..."
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip