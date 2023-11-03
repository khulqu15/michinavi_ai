import re

train_record_path = './datasets/train.record'
test_record_path = './datasets/test.record'
pipeline_config_path = './datasets/mobilenet_v2.config'
label_map_path = './datasets/label_map.pbtxt'
fine_tune_checkpoint = './datasets/mobilenet_v2/mobilenet_v2.ckpt-1.data-00000-of-00001'
model_dir = './datasets/training/'
num_classes = 1
batch_size = 96 #16
num_steps = 7500
num_eval_steps = 1000

with open(pipeline_config_path) as f:
    config = f.read()
    
with open(pipeline_config_path, 'w') as f:
    config = re.sub('label_map_path: ".*?"',
                    'label_map_path: "{}"'.format(label_map_path), config)
    
    config = re.sub('fine_tune_checkpoint: ".*?"',
                    'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), config)
    
    # Set train tf-record file path
    config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 
                    'input_path: "{}"'.format(train_record_path), config)
    
    # Set test tf-record file path
    config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 
                    'input_path: "{}"'.format(test_record_path), config)
    
    # Set number of classes.
    config = re.sub('num_classes: [0-9]+',
                    'num_classes: {}'.format(num_classes), config)
    
    # Set batch size
    config = re.sub('batch_size: [0-9]+',
                    'batch_size: {}'.format(batch_size), config)
    
    # Set training steps
    config = re.sub('num_steps: [0-9]+',
                    'num_steps: {}'.format(num_steps), config)
    
    f.write(config)