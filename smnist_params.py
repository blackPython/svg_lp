import tensorflow as tf 
from tensorflow.contrib.training import HParams
def smnist_hparams():
    hparams = HParams()
    hparams.add_hparam("g_dim", 128)
    hparams.add_hparam("z_dim", 10)
    hparams.add_hparam("prior_rnn_layers", 1)
    hparams.add_hparam("posterior_rnn_layers", 1)
    hparams.add_hparam("predictor_rnn_layers", 2)
    hparams.add_hparam("num_input_frames", 5)
    hparams.add_hparam("num_target_frames", 10)
    hparams.add_hparam("beta", 1e-4)
    hparams.add_hparam("learning_rate", 0.002)
    hparams.add_hparam("frame_size", (64,64,1))
    hparams.add_hparam("batch_size", 100)
    hparams.add_hparam("rnn_size", 256)
    hparams.add_hparam("num_digits", 2)
    return hparams
