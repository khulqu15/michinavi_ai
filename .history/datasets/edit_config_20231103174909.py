import re

train_record_path = './datasets/train.record'
test_record_path = './datasets/test.record'
pipeline_config_path = './datasets/mobilenet_v2.config'
label_map_path = './datasets/label_map.pbtxt'
fine_tune_checkpoint = './datasets/mobile_v2/mobilenet_v2.ckpt-1'
model_dir = './datasets/training/'

with open(pipeline_config_path) as f:
    config = f.read()
    
with open(pipeline_config_path, 'w') as f:
    config = re.sub('label_map_path: ".*?"',
                    'label_map_path: "{}"'.format(label_map_path), config)
    
    config = re.sub('fine_tune_checkpoint: ".*?"',
                    'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), config)
    
    c