# config.py
import os.path


# new yolo config
train_cfg = {
    'lr_epoch': (90, 120),
    'max_epoch': 150,
    'min_dim': [416, 416]
}
