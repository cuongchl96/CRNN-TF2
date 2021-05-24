import yaml
import argparse

from modules.parameters import base_config

config = {
    'model_build': {
        'backbone': 'resnet18_slim',
        'sequence_modeling': 'abilstm',
        'model_head': 'attention'
    },

    'model_params': {
        'imgH': 32,
        'imgW': 280,
        'num_channels': 3,
        'max_len': 12,
        'hidden_size': 256,
        'character': '0123456789'
    },

    'dataset': {
        'train_set': 'datasets/IDNum/idnum_train.tfrecords',
        'valid_set': 'datasets/IDNum/idnum_valid.tfrecords',
        'test_set': 'datasets/IDNum/idnum_test.tfrecords'

    },

    'data_augmentation': {
        'prob': 0.3,
        'augment_level': 5
    },

    'training_params': {
        'optimizer': 'Adadelta',
        'num_epochs': 50,
        'batch_size': 32,
        'initial_lnr': 1.0,
        'num_warmup_steps': 100,
        'log_dir': 'checkpoints/IDNum/ResNet18s-abilstm_256-attention',
        'log_interval': 200,
        'save_interval': 2000
    }
}

with open('modules/test_config.yaml', 'w') as f:
    yaml.dump(config, f, sort_keys=False)

# config = base_config.Config().from_yaml('modules/test_config.yaml')
# print(config.model_build.backbone)

# config = base_config.Config({'key': 1})
# print(config.__dict__)