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
import sys
import json
import argparse
import fileinput

import tensorflow as tf
import numpy as np

def create_vocabulary(text):
    vocab = sorted(set(text))
    print('{} unique characters'.format(len(vocab)))
    char2idx = { u: i for i, u in enumerate(vocab) }
    idx2char = np.array(vocab)
    
    text_as_int = np.array([char2idx[c] for c in text])
    return vocab, char2idx, idx2char, text_as_int

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])

def generate_text(model, start_string, char2idx, idx2char, num_generate=1000, temperature=1.0):
    # Evaluation step (generating text using the learned model)

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    # temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return ''.join(text_generated)

def read_config(model_dir):
    config_path = os.path.join(model_dir, 'model.json')
    with open(config_path, 'rt') as config_file:
        return json.load(config_file)

def read_raw_text(model_dir):
    raw_path = os.path.join(model_dir, 'corpus.text')
    with open(raw_path, 'rb') as raw_file:
        return raw_file.read().decode('utf-8')

def load_model(model_name):
    model_dir = os.path.join(os.getcwd(), model_name)

    config = read_config(model_dir)
    embedding_dim = config['embedding']
    rnn_units = config['rnn']

    # Read in all of the text into a single line
    raw_text = read_raw_text(model_dir)
    vocab, char2idx, idx2char, encoded_text = create_vocabulary(raw_text)
    # vocab = config['vocab']
    # char2idx = config['char2idx']
    # idx2char = np.array(vocab)

    print('Types vocab: %s char2idx: %s idx2char: %s' % (type(vocab), type(char2idx), type(idx2char)))

    checkpoint_dir = os.path.join(model_dir, 'checkpoints')

    print('Checkpt dir: %s' % checkpoint_dir)
    print('Embedding dim: %d' % embedding_dim)
    print('RNN Units: %d' % rnn_units)

    model = build_model(len(vocab), embedding_dim, rnn_units, 1)
    latest_chkpt = tf.train.latest_checkpoint(checkpoint_dir)
    print('Loading checkpoints: %s' % latest_chkpt)
    model.load_weights(latest_chkpt)
    model.build(tf.TensorShape([1, None]))
    return model, char2idx, idx2char

class Generator:
    def __init__(self, model_name):
        self.model, self.char2idx, self.idx2char = load_model(model_name)

    def generate(self, length=1000, seed=None):
        return generate_text(self.model, seed, self.char2idx, self.idx2char, length)

def generate_rnn(input_args=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, action='store', help='name of the model')
    parser.add_argument('--length', type=int, default=1000, action='store', help='length of text to generate')
    parser.add_argument('--seed', default='', action='store', help='seed text or blank to use random')
    args = parser.parse_args(input_args)

    # model, char2idx, idx2char = load_model(args.name)

    start = args.seed if len(args.seed) > 0 else u'dude'
    # generate_text(model, start, char2idx, idx2char, args.length)

    generator = Generator(args.name)
    print(generator.generate(args.length, start))
