#!/usr/bin/python

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

# import os
# import json

from skynet.seq2seq.trainer import train_rnn

# channel_names = json.loads(os.environ['SM_CHANNELS'])
# hyperparameters = json.loads(os.environ['SM_HPS'])
# input_path = os.environ['SM_INPUT_DIR']
# input_config_path = os.environ['SM_INPUT_CONFIG_DIR']
# output_data_path = os.environ['SM_OUTPUT_DATA_DIR']
# model_path = os.environ['SM_MODEL_DIR']

# train_rnn([
#         '--name',   'sagemaker-gru-model',
#         '--batch',  int(hyperparameters['batch-size']),
#         '--buffer', int(hyperparameters['buffer-size']),
#         '--type',   hyperparameters['model-type'],
#         '--dim',    int(hyperparameters['embedding-dim']),
#         '--units',  int(hyperparameters['rnn-units']),
#         '--seq',    int(hyperparameters['seq-length']),
#         '--epochs', int(hyperparameters['epochs']),
#         '--input',  os.path.join(input_path, 'corpus'),
#         '--output', output_data_path])

if __name__ == '__main__':
        train_rnn()
