import yaml
import argparse

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
    },

    'dataset': {
        'train_set': 'datasets/IDNum/idnum_train.tfrecords',
        'valid_set': 'datasets/IDNum/idnum_valid.tfrecords',
        'test_set': 'datasets/IDNum/idnum_test.tfrecords'

    },

    'training_param': {
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

# with open('test_config.yaml', 'w') as f:
#     yaml.dump(config, f, sort_keys=False)
parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, default=12, help='maximum-label-length')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
parser.add_argument('--character', type=str, default='0123456789', help='character label')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
opt = parser.parse_args()

print(type(opt))
with open('test_config.yaml', 'r') as f:
    data = yaml.load(f)
print(data.model_build.backbone)