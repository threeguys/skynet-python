#  Copyright 2020 Ray Cole
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import json

import numpy as np
import tensorflow as tf

MODEL_FILE = 'model.json'
CORPUS_FILE = 'corpus.text'
VOCAB_FILE = 'vocab.json'
CHARMAP_FILE = 'charmap.json'

def read_json_file(base_dir, filename):
    file_path = os.path.join(base_dir, filename)
    with open(file_path, 'rt') as f:
        return json.load(f)

def read_utf8_file(base_dir, filename):
    file_path = os.path.join(base_dir, filename)
    with open(file_path, 'rb') as f:
        return f.read().decode('utf-8')

def write_json_file(obj, base_dir, filename):
    file_path = os.path.join(base_dir, filename)
    with open(file_path, 'wt') as f:
        json.dump(obj, f)
    print('Wrote json file: %s' % file_path)
    return file_path

def write_utf8_file(text, base_dir, filename, overwrite=False):
    file_path = os.path.join(base_dir, filename)
    if not os.path.isfile(file_path) or overwrite:
        with open(file_path, 'wb') as f:
            f.write(text.encode('utf-8'))
        print('Wrote utf-8 file: %s' % file_path)
    return file_path

def read_config(model_dir):
    return read_json_file(model_dir, MODEL_FILE)

def read_corpus(model_dir):
    return read_utf8_file(model_dir, CORPUS_FILE)

def read_vocab(model_dir):
    return read_json_file(model_dir, VOCAB_FILE)

def read_charmap(model_dir):
    return read_json_file(model_dir, CHARMAP_FILE)

def write_config(model_dir, config):
    return write_json_file(config, model_dir, MODEL_FILE)

def write_corpus(model_dir, raw_text):
    return write_utf8_file(raw_text, model_dir, CORPUS_FILE)

def write_vocab(model_dir, vocab):
    return write_json_file(vocab, model_dir, VOCAB_FILE)

def write_charmap(model_dir, charmap):
    return write_json_file(charmap, model_dir, CHARMAP_FILE)

#
# Creates a vocabulary from the input text
# The total vocabulary is the unique set of characters and in addition, this function
# generates the folloing return values: vocab, char2idx, idx2char, encoded_text
#
#   vocab - sorted list of all unique characters the model can understand/speak
#   char2idx - map of character to its integer representation
#   idx2char - map of integer representations back to their corresponding characters
#   encoded_text - corpus text, encoded using integers according to the maps
#
def create_vocabulary(text):
    vocab = sorted(set(text))
    print('{} unique characters'.format(len(vocab)))
    char2idx = { u: i for i, u in enumerate(vocab) }
    idx2char = np.array(vocab)
    
    encoded_text = np.array([char2idx[c] for c in text])
    return vocab, char2idx, idx2char, encoded_text

# Creates a GRU model, based on https://www.tensorflow.org/tutorials/text/text_generation
def build_gru_model(vocab_size, embedding_dim, rnn_units, batch_size):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[ batch_size, None ]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
