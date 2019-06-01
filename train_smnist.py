import argparse
import sys
import os

from smnist_base import NextFrameSmnist, sminst_input_func, smnist_model_fn, get_feature_columns
from smnist_params import smnist_hparams

import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser(description= "Command to train smnist")
parser.add_argument("--num_iterations", default = 180000,type=int, help ="The number of training steps")
parser.add_argument("--model_dir", default = "model_dir/", help = "The director to store summaries and checkpoints")
parser.add_argument("--hparams", default = "", type = str, help = "Hparams to override")
#parsing the arguments
args = parser.parse_args()
num_iterations = args.num_iterations
model_dir = args.model_dir

#getting model params
hparams = smnist_hparams()
hparams.parse(args.hparams)
hparams.add_hparam("model_dir",args.model_dir)
feature_columns = get_feature_columns(hparams)

tf.logging.set_verbosity(tf.logging.INFO)
strategy = tf.contrib.distribute.MirroredStrategy()
run_config = tf.estimator.RunConfig(model_dir = model_dir, train_distribute = strategy,  save_checkpoints_steps = 4000, save_summary_steps = 200)
estimator = tf.estimator.Estimator(smnist_model_fn, model_dir=model_dir,\
    config = run_config, params = {"hparams" : hparams})

train_spec = tf.estimator.TrainSpec(input_fn = lambda : sminst_input_func(hparams, tf.estimator.ModeKeys.TRAIN), max_steps = num_iterations)
eval_spec  = tf.estimator.EvalSpec(input_fn = lambda : sminst_input_func(hparams, tf.estimator.ModeKeys.TRAIN))

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
