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
import fileinput
import tensorflow as tf
import numpy as np
import argparse
import json

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

# Batches up the input into 100 character sequences for training
# Returns: encoded text, broken into sequences of length 100
def generate_sequences(text, text_as_int, idx2char):
    seq_length = 100
    examples_per_epoch = len(text)
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    return sequences

# Creates a GRU model, based on https://www.tensorflow.org/tutorials/text/text_generation
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[ batch_size, None ]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])

# Duplicates a single chunk into two separate lists, offset by
# one character for training. We are trying to make the model predict what the
# next character will be, so our correct answer is the actual next char
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# Helper to package up the text into a tensor dataset
def prepare_dataset(raw_text, encoded_text, idx2char, buffer_size, batch_size, drop_remainder=True):
    sequences = generate_sequences(raw_text, encoded_text, idx2char)
    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    return sequences, dataset

# Compiles the model to be ready for fitting, this function prints out
# a sample of the model shape, a summary, sets the loss function and then calls .compile()
def compile_model(model, dataset):
    # Print out some info about the model
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    print(model.summary())

    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    example_batch_loss  = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("scalar_loss:      ", example_batch_loss.numpy().mean())

    model.compile(optimizer='adam', loss=loss)

# Fits the model for a given number of epochs, checkpointing into the specified directory
def fit_model(model, dataset, checkpoint_dir, start_epoch=None, end_epoch=10):
    # Setup checkpointing
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "epoch-{epoch}-loss-{loss:.2f}"), verbose=1,
            save_weights_only=True)

    # Train the model
    if start_epoch is None:
        print('Training %d epochs from scratch' % end_epoch)
        history = model.fit(dataset, epochs=end_epoch, callbacks=[ checkpoint_callback ])
    else:
        print('Training epochs %d to %d' % (start_epoch, end_epoch))
        history = model.fit(dataset, epochs=end_epoch, initial_epoch=start_epoch, callbacks=[ checkpoint_callback ])

    return tf.train.latest_checkpoint(checkpoint_dir)

def write_training_config(model_dir, config):
    # Save the configuration
    config_path = os.path.join(model_dir, 'model.json')
    with open(config_path, 'wt') as config_file:
        json.dump(config, config_file)
    print('Wrote config to %s' % config_path)

def write_raw_text(model_dir, raw_text):
    raw_path = os.path.join(model_dir, 'corpus.text')
    if not os.path.isfile(raw_path):
        with open(raw_path, 'wb') as raw_file:
            raw_file.write(raw_text.encode('utf-8'))
        print('Wrote text to %s' % raw_path)

def train_model(model_name, raw_text, buffer_size, batch_size, embedding_dim, rnn_units, epochs):
    model_dir = os.path.join(os.getcwd(), model_name)
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')

    steps_per_epoch = int(len(raw_text) / (batch_size * 100)) # batch_size * sequence length
    if steps_per_epoch <= 0:
        raise ValueError('Invalid steps per epoch: %d somethin aint right!' % steps_per_epoch)

    # Create the vocab and encode the text
    vocab, char2idx, idx2char, encoded_text =  create_vocabulary(raw_text)
    # Split dataset into batches for training
    sequences, dataset = prepare_dataset(raw_text, encoded_text, idx2char, buffer_size, batch_size)
    # Setup the tf model
    model = build_model(len(vocab), embedding_dim, rnn_units, batch_size)

    if os.path.isdir(checkpoint_dir):
        latest_chkpt = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_chkpt is not None:
            print('Found checkpoint: %s loading weights...' % latest_chkpt)
            chkpt = model.load_weights(latest_chkpt)
            # chkpt.assert_consumed()
            model.build(tf.TensorShape([64, 100]))
            # re.search('epoch-([0-9]+)-.*', latest_chkpt)

    compile_model(model, dataset)
    num_iterations = int(model.optimizer.iterations.numpy())
    initial_epoch = round(num_iterations / steps_per_epoch)
    print('Optimizer epoch: %d iterations: %d steps-per: %d' % (initial_epoch, num_iterations, steps_per_epoch))

    # Training loop...
    start_epoch = initial_epoch if initial_epoch > 0 else None
    end_epoch = epochs + initial_epoch
    last_chkpt = fit_model(model, dataset, checkpoint_dir, start_epoch, end_epoch)

    config = {
        'name': model_name,
        'batch': batch_size,
        'buffer': buffer_size,
        'embedding': embedding_dim,
        'rnn': rnn_units,
        'epochs': epochs,
        'vocab': vocab,
        'char2idx': char2idx,
        'idx2char': list(idx2char)
    }

    write_training_config(model_dir, config)
    write_raw_text(model_dir, raw_text)

    # Should delete old checkpoint files here

def train_rnn(input_args=sys.argv):
    parser = argparse.ArgumentParser(description='train seq2seq RNN network based on text input')
    parser.add_argument('--name', required=True, action='store', help='name of the model')
    parser.add_argument('--batch', default=64, type=int, action='store', help='batch size')
    parser.add_argument('--buffer', default=10000, type=int, action='store', help='working shuffle buffer size')
    parser.add_argument('--type', default='rnn', action='store', help='type of model (rnn|?)')
    parser.add_argument('--dim', default=256, type=int, action='store', help='embedding dimension')
    parser.add_argument('--units', default=1024, type=int, action='store', help='rnn units')
    parser.add_argument('--epochs', default=10, type=int, action='store', help='number of epochs to train')
    parser.add_argument('--input', required=True, type=argparse.FileType('r', encoding='UTF-8'), action='store', help='input corpus text (-) for stdin')
    args = parser.parse_args(input_args)

    raw_text = ' '.join([line.rstrip() for line in args.input])
    train_model(args.name, raw_text, args.buffer, args.batch, args.dim, args.units, args.epochs)
