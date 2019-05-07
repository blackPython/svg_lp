import os
import argparse
import sys

from smnist_base import NextFrameSmnist, sminst_input_func, smnist_model_fn
from smnist_params import smnist_hparams

import tensorflow as tf 
import numpy as np

import utils

from tensor2tensor.layers import common_layers
import cv2

parser = argparse.ArgumentParser(description = "Whole lot of crap")
parser.add_argument("--num_tests", default = 10, type = int)
parser.add_argument("--checkpoint_dir", default = "/home/aditya/data/t2t/output/smnist_fix/", type = str)
parser.add_argument("--out_dir", default = "smnist_out", type = str)
parser.add_argument("--prediction_prefix", default = "prediction", type = str)
parser.add_argument("--hparams", default = "batch_size=1,num_target_frames=100", type = str)
parser.add_argument("--input_prefix", default = "input", type = str)
parser.add_argument("--fps", type = int, default = 4)

args = parser.parse_args()


hparams = smnist_hparams() 
hparams.num_target_frames = 100

tf.logging.set_verbosity(tf.logging.INFO)

estimator = tf.estimator.Estimator(smnist_model_fn, model_dir=args.checkpoint_dir, params = {"hparams":hparams})

prediction_list = []
input_list = []
count = 0
for prediction in estimator.predict(input_fn= lambda: sminst_input_func(hparams,num_samples = args.num_tests)):
    predictions = prediction['predictions']
    prediction_list.append((predictions*255).astype(np.uint8))
    input_list.append((prediction['inputs']*255).astype(np.uint8))

for i,(predictions,inputs) in enumerate(zip(prediction_list,input_list)):
    predictions = map(lambda x: np.stack([np.squeeze(x)]*3, axis = 2),np.split(predictions, hparams.num_target_frames))
    inputs = map(lambda x: np.stack([np.squeeze(x)]*3, axis =2), np.split(inputs, hparams.num_input_frames))
    out_file = os.path.join(args.out_dir, args.prediction_prefix)+ str(i+1) + ".gif"
    in_file = os.path.join(args.out_dir, args.input_prefix) + str(i+1) + ".gif"
    utils.save_gif(out_file, predictions, args.fps)
    utils.save_gif(in_file, inputs, args.fps)