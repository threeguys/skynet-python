#!/usr/bin/python

import os
import json

from skynet.seq2seq.trainer import train_rnn

channel_names = json.loads(os.environ['SM_CHANNELS'])
hyperparameters = json.loads(os.environ['SM_HPS'])
input_path = os.environ['SM_INPUT_DIR']
input_config_path = os.environ['SM_INPUT_CONFIG_DIR']
output_data_path = os.environ['SM_OUTPUT_DATA_DIR']
model_path = os.environ['SM_MODEL_DIR']

train_rnn([
        '--name',   'sagemaker-gru-model',
        '--batch',  int(hyperparameters['batch-size']),
        '--buffer', int(hyperparameters['buffer-size']),
        '--type',   hyperparameters['model-type'],
        '--dim',    int(hyperparameters['embedding-dim']),
        '--units',  int(hyperparameters['rnn-units']),
        '--seq',    int(hyperparameters['seq-length']),
        '--epochs', int(hyperparameters['epochs']),
        '--input',  os.path.join(input_path, 'corpus'),
        '--output', output_data_path])
