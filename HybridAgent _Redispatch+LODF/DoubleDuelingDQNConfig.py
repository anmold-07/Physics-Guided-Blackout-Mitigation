import os
import json

class DoubleDuelingDQNConfig():
    """
    DoubleDuelingDQN configurable hyperparameters
    exposed as class attributes
    """

    LR_DECAY_STEPS = 1024
    LR_DECAY_RATE = 1
    INITIAL_EPSILON = 0.9999
    FINAL_EPSILON = 0.05
    DECAY_EPSILON = 210*128
    DISCOUNT_FACTOR = 0.99
    PER_CAPACITY = 1024*94
    PER_ALPHA = 0.8
    PER_BETA = 0.5
    UPDATE_FREQ = 1
    UPDATE_TARGET_HARD_FREQ = -1
    UPDATE_TARGET_SOFT_TAU = 1e-3
    N_FRAMES = 6
    BATCH_SIZE = 64
    LR = 0.0005
    VERBOSE = False

    @staticmethod
    def from_json(json_in_path):
        with open(json_in_path, 'r') as fp:
            conf_json = json.load(fp)
        
        for k,v in conf_json.items():
            if hasattr(DoubleDuelingDQNConfig, k):
                setattr(DoubleDuelingDQNConfig, k, v)

    @staticmethod
    def to_json(json_out_path):
        conf_json = {}
        for attr in dir(DoubleDuelingDQNConfig):
            if attr.startswith('__') or callable(attr):
                continue
            conf_json[attr] = getattr(DoubleDuelingDQNConfig, attr)

        with open(json_out_path, 'w+') as fp:
            json.dump(fp, conf_json, indent=2)