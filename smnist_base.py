import tensorflow as tf 
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_video
import tensorflow.layers as tfl
import utils
import numpy as np
from functools import partial
import os
import cv2

def l2_loss(val1,val2):
    return tf.reduce_mean(tf.square(val1-val2))

class NextFrameSmnist(object):
    def __init__(self, hparams, mode):
        self.hparams = hparams
        self.mode = mode

    @property
    def is_training(self):
        return self.mode == tf.estimator.ModeKeys.TRAIN
    

    def get_kl_loss(self, means, log_vars, means_p=None, log_vars_p=None):
        """Get KL loss for all the predicted Gaussians."""
        kl_loss = 0.0
        if means_p is None:
            means_p = tf.unstack(tf.zeros_like(means))
        if log_vars_p is None:
            log_vars_p = tf.unstack(tf.zeros_like(log_vars))
        enumerated_inputs = zip(means, log_vars, means_p, log_vars_p)
        if self.is_training:
            for mean, log_var, mean_p, log_var_p in enumerated_inputs:
                posterior_distribution = tf.distributions.Normal(mean, tf.exp(tf.multiply(0.5,log_var)))
                prior_distribution = tf.distributions.Normal(mean_p, tf.exp(tf.multiply(0.5,log_var_p)))
                kl_loss += tf.reduce_mean(tf.reduce_sum(tf.distributions.kl_divergence(posterior_distribution, prior_distribution), axis = -1))
        tf.summary.scalar("kl_raw", kl_loss)

        beta = self.hparams.beta
        return beta * kl_loss

    def dcgan_conv(self, x, nout, reuse = tf.AUTO_REUSE):
        """A 2D conv follwed by batch norm and LeakyRelu
        inputs:
            x - The tensor to be feed to the conv2d
            nout - number of output filters
            reuse - If reuse a conv2d with the same in the scope
        outputs:
            hidden - The output tensor which is of half height and width
        """
        hidden = tf.layers.conv2d( x, nout, 4, strides = 2, padding = "same")
        hidden = tf.layers.batch_normalization(hidden, momentum = 0.9, training = self.is_training)
        hidden = tf.nn.leaky_relu(hidden, alpha = 0.2)
        return hidden

    def dcgan_upconv(self, x, nout, reuse = tf.AUTO_REUSE):
        """A 2D conv transpose follwed by batch norm and LeakyRelu
        inputs:
            x - The tensor to be feed to the conv2d transpose
            nout - number of output filters
            reuse - If reuse a conv2d tranpose with the same in the scope
        outputs:
            hidden - The output tensor which is of half height and width
        """
        hidden = tf.layers.conv2d_transpose( x, nout, 4, strides = 2, padding = "same")
        hidden = tf.layers.batch_normalization(hidden, momentum = 0.9, training = self.is_training)
        hidden = tf.nn.leaky_relu(hidden, alpha = 0.2)
        return hidden

    def encoder(self, x):
        """ Constructs a DCGAN encoder
        inputs:
            x - 4D tensor (batch, height, width, channels)
        return:
            output - A tuple, where first element is the latent encoding and
                     the second is a list of skip outputs
        """
        dim = self.hparams.g_dim

        nf = 64

        # 64 x 64 x nc
        h1 = self.dcgan_conv(x, nf)
        
        #32 x 32 x nf
        h2 = self.dcgan_conv(h1, nf * 2)
        
        #16 x 16 x (nf*2)
        h3 = self.dcgan_conv(h2, nf * 4)
        
        #8 x 8 x (nf*4)
        h4 = self.dcgan_conv(h3, nf * 8)
        
        #4 x 4 x (nf*8)
        h5 = tf.layers.conv2d(h4, dim, 4)
        h5 = tf.layers.batch_normalization(h5, momentum = 0.9, training = self.is_training)
        h5 = tf.nn.tanh(h5)

        return tf.reshape(h5, [-1, dim]), [h1, h2, h3, h4]

    def decoder(self, x, skip):
        """ Contructs a DCGAN decoder
        inputs:
            x - Input encoding
            skip - Input skip connections
        return:
            output - Generated image 4D tensor
        """
        
        _, dim = common_layers.shape_list(x)
        x = tf.reshape(x, [-1, 1, 1, dim])

        nf = 64

        h1 = tf.layers.conv2d_transpose(x, nf * 8, 4)
        h1 = tf.layers.batch_normalization(h1, momentum = 0.9, training = self.is_training)
        h1 = tf.nn.leaky_relu(h1, alpha = 0.2)

        #4 x 4x nf*8
        h2 = tf.concat([h1, skip[3]], axis = 3)
        h2 = self.dcgan_upconv(h2, nf*4)

        #8 x 8 x nf*4
        h3 = tf.concat([h2, skip[2]], axis = 3)
        h3 = self.dcgan_upconv(h3, nf*2)

        #16 x 16 nf*2
        h4 = tf.concat([h3, skip[1]], axis = 3)
        h4 = self.dcgan_upconv(h4, nf)

        #32 x 32 x nf
        h5 = tf.concat([h4, skip[0]], axis = 3)
        h5 = tf.layers.conv2d_transpose(h5, 1, 4, strides = 2, padding = "same")
        h5 = tf.nn.sigmoid(h5)

        return h5

    def rnn_model(self, hidden_size, name,n_layers = 1):
        """ Return an LSTM Cell """
        layers_units = [hidden_size] * n_layers
        rnn_cell = tf.nn.rnn_cell.LSTMCell

        cells = [rnn_cell(units, name = name + "_"+ str(i)) for i,units in enumerate(layers_units)]
        stacked_rnn = tf.contrib.rnn.MultiRNNCell(cells)
        return stacked_rnn

    def deterministic_rnn(self, cell, inputs, states, output_size, scope):
        """Deterministic RNN step function.

        Args:
        cell: RNN cell to forward through
        inputs: input to RNN cell
        states: previous RNN state
        output_size: size of the output
        scope: scope of the current RNN forward computation parameters
        Returns:
        outputs: deterministic RNN output vector
        states: updated RNN states
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            embedded = tfl.dense(inputs, cell.output_size, name="embed")
            hidden, states = cell(embedded, states)
            outputs = tfl.dense(hidden, output_size, activation=tf.nn.tanh, name="output")

        return outputs, states

    def gaussian_rnn(self, cell, inputs, states, output_size, scope):
        """Deterministic RNN step function.

        Args:
        cell: RNN cell to forward through
        inputs: input to RNN cell
        states: previous RNN state
        output_size: size of the output
        scope: scope of the current RNN forward computation parameters
        Returns:
        mu: mean of the predicted gaussian
        logvar: log(var) of the predicted gaussian
        states: updated RNN states
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            embedded = tfl.dense( inputs, cell.output_size, name="embed")
            hidden, states = cell(embedded, states)
            mu = tfl.dense(hidden, output_size, activation=None, name="mu")
            logvar = tfl.dense(hidden, output_size, activation=None, name="logvar")

        return mu, logvar, states

    def process(self, inputs, targets):
        all_frames = tf.unstack(inputs, axis = 1) + tf.unstack(targets, axis = 1)
        hparams = self.hparams

        batch_size = common_layers.shape_list(all_frames[0])[0]
        
        z_dim = hparams.z_dim
        g_dim = hparams.g_dim
        rnn_size = hparams.rnn_size
        prior_rnn_layers = hparams.prior_rnn_layers
        posterior_rnn_layers = hparams.posterior_rnn_layers
        predictor_rnn_layers = hparams.predictor_rnn_layers

        num_input_frames = hparams.num_input_frames
        num_target_frames = hparams.num_target_frames
        num_all_frames = num_input_frames + num_target_frames

        #Creating RNN cells
        predictor_cell = self.rnn_model(rnn_size, "predictor", n_layers = predictor_rnn_layers)
        prior_cell = self.rnn_model(rnn_size, "prior", n_layers = prior_rnn_layers)
        posterior_cell = self.rnn_model(rnn_size, "posterior", n_layers = posterior_rnn_layers)

        #Getting RNN states 
        predictor_state = predictor_cell.zero_state(batch_size, tf.float32)
        prior_state = prior_cell.zero_state(batch_size, tf.float32)
        posterior_state = posterior_cell.zero_state(batch_size, tf.float32)

        #Encoding
        enc_frames, enc_skips = [], []
        with tf.variable_scope("encoder", reuse = tf.AUTO_REUSE):
            for frame in all_frames if self.is_training else all_frames[:num_input_frames]:
                enc, skip = self.encoder(frame)
                enc_frames.append(enc)
                enc_skips.append(skip)

        #Prediction
        prior_mus = []
        prior_logvars = []
        posterior_mus = []
        posterior_logvars = []
        predicted_frames = []
        z_positions = []
        skip = None
        if self.is_training:
            for i in range(1,num_all_frames):
                h = enc_frames[i-1]
                h_target = enc_frames[i]
                if i < num_input_frames:
                    skip = enc_skips[i-1]
                with tf.variable_scope("prediction", reuse = tf.AUTO_REUSE):
                    mu, log_var, posterior_state = self.gaussian_rnn(posterior_cell, h_target, posterior_state,
                        z_dim, "posterior")
                    mu_p, log_var_p, prior_state = self.gaussian_rnn(prior_cell, h, prior_state, z_dim, "prior")
                    z = utils.get_gaussian_tensor(mu,log_var)
                    h_pred, predictor_state = self.deterministic_rnn(predictor_cell, tf.concat([h,z], axis = 1),\
                        predictor_state, g_dim, "predictor")
                with tf.variable_scope("decoder", reuse = tf.AUTO_REUSE):
                    x_pred = self.decoder(h_pred, skip)
                predicted_frames.append(x_pred)
                prior_mus.append(mu_p)
                prior_logvars.append(log_var_p)
                posterior_mus.append(mu)
                posterior_logvars.append(log_var)
                z_positions.append(z)
        else:
            for i in range(1, num_all_frames):
                if i < num_input_frames:
                    h = enc_frames[i-1]
                    skip = enc_skips[i-1]
                else:
                    with tf.variable_scope("encoder", reuse = tf.AUTO_REUSE):
                        h, _ = self.encoder(predicted_frames[-1])
                mu = log_var = mu_p = log_var_p = None
                if i < num_input_frames:
                    h_target = enc_frames[i]
                    with tf.variable_scope("prediction", reuse = tf.AUTO_REUSE):
                        mu, log_var, posterior_state = self.gaussian_rnn(posterior_cell, h_target, posterior_state,\
                            z_dim, "posterior")
                        mu_p, log_var_p, prior_state= self.gaussian_rnn(prior_cell, h, prior_state, z_dim, "prior")
                        z = utils.get_gaussian_tensor(mu,log_var)
                        _, predictor_state = self.deterministic_rnn(predictor_cell, tf.concat([h,z], axis = 1), predictor_state,\
                            g_dim, "predictor")
                    x_pred = all_frames[i]
                else:
                    with tf.variable_scope("prediction", reuse = tf.AUTO_REUSE):
                        mu_p, log_var_p, prior_state = self.gaussian_rnn(prior_cell, h, prior_state, z_dim, "prior")
                        z = utils.get_gaussian_tensor(mu_p, log_var_p)
                        h_pred, predictor_state = self.deterministic_rnn(predictor_cell, tf.concat([h,z], axis = 1), predictor_state, g_dim, "predictor")
                    with tf.variable_scope("decoder", reuse = tf.AUTO_REUSE):
                        x_pred = self.decoder(h_pred,skip)
                predicted_frames.append(x_pred)
                prior_mus.append(mu_p)
                prior_logvars.append(log_var_p)
                posterior_mus.append(mu)
                posterior_logvars.append(log_var)
                z_positions.append(z)

        recon_loss = 0
        kl_loss = 0

        #recon loss
        recon_loss = l2_loss(tf.stack(predicted_frames), tf.stack(all_frames[1:]))*num_all_frames

        if self.is_training:
            #kl loss
            kl_loss = self.get_kl_loss(posterior_mus,posterior_logvars, prior_mus,\
                prior_logvars)
        pred_outputs = tf.stack(predicted_frames[num_input_frames-1:], axis = 1)
        rgb_frames = tf.tile(common_layers.convert_real_to_rgb(tf.stack(predicted_frames, axis = 1)), [1,1,1,1,3])
        all_frames = tf.stack(all_frames, axis = 1)
        all_frames_rgb = tf.tile(common_layers.convert_real_to_rgb(all_frames), [1,1,1,1,3])
        common_video.gif_summary("body_output", rgb_frames)
        common_video.gif_summary("all_ground_frames", all_frames_rgb)
        tf.summary.scalar("kl_loss", kl_loss)
        tf.summary.scalar("recon_loss", recon_loss)
        loss = recon_loss + kl_loss

        return pred_outputs, loss, tf.stack(z_positions,axis = 1)


def smnist_model_fn(features, labels, mode, params):
    del labels
    hparams = params['hparams']
    model = NextFrameSmnist(hparams,mode)

    inputs = features['inputs']
    targets = features['targets']

    predictions, loss, z_positions = model.process(inputs, targets)

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = hparams.learning_rate
        
        opt_summaries = ["loss", "gradients", "gradient_norm", "global_gradient_norm"]

        train_op = tf.contrib.layers.optimize_loss(
            name = "training",
            loss = loss,
            global_step = tf.train.get_or_create_global_step(),
            learning_rate = learning_rate,
            optimizer = "Adam",
            summaries = opt_summaries,
            colocate_gradients_with_ops = True)
        
        return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)
    
    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_summary_hook = tf.train.SummarySaverHook(save_steps = 1, output_dir =\
            os.path.join(hparams.model_dir, "eval"),summary_op = tf.summary.merge_all())
        return tf.estimator.EstimatorSpec(mode, loss = loss, evaluation_hooks =[eval_summary_hook,])
    
    elif mode == tf.estimator.ModeKeys.PREDICT:
        out_dict = {"predictions" : predictions, "inputs" : inputs, "z_positions" : z_positions}
        return tf.estimator.EstimatorSpec(mode, predictions = out_dict)
    
    else:
        raise NotImplementedError("This mode is not implemented")

def generate_smnist_sequence(seq_len, mnist_images, frame_size, num_digits):
    digit_indices = np.random.randint(0,mnist_images.shape[0], size = num_digits)
    digits = [mnist_images[digit_index] for digit_index in digit_indices]
    digit_size = 32
    image_size = frame_size[0]
    x = np.zeros((seq_len, image_size, image_size, 1), dtype = np.float32)
    for n in range(len(digits)):
        digit = np.expand_dims(cv2.resize(digits[n].astype(np.float32)/255.0, dsize = (digit_size,digit_size), interpolation = cv2.INTER_CUBIC),2)
        sx = np.random.randint(image_size - digit_size)
        sy = np.random.randint(image_size - digit_size)
        dx = np.random.randint(-4, 5)
        dy = np.random.randint(-4, 5)
        for t in range(seq_len):
            if sy < 0:
                sy = 0
                dx = np.random.randint(-4, 5)
                dy = np.random.randint(-4, 5)

            elif sy >= image_size - digit_size:
                sy = image_size - digit_size - 1
                dx = np.random.randint(-4, 5)
                dy = np.random.randint(-4, 5)

            if sx < 0:
                sx = 0
                dx = np.random.randint(-4, 5)
                dy = np.random.randint(-4, 5)

            elif sx >= image_size - digit_size:
                sx = image_size - digit_size - 1
                dx = np.random.randint(-4, 5)
                dy = np.random.randint(-4, 5)
            x[t, sy : (sy + digit_size), sx : (sx + digit_size), :] += digit
            sy += dy
            sx += dx
    
    x[x>1] = 1
    return x

def dataset_generator_func(hparams,mode):
    temp = tf.keras.datasets.mnist.load_data()
    smnist_images = temp[mode == tf.estimator.ModeKeys.EVAL][0]
    num_input_frames = hparams.num_input_frames
    num_target_frames = hparams.num_target_frames
    frame_size = hparams.frame_size
    seq_len = num_input_frames + num_target_frames
    num_digits = hparams.num_digits
    while True:
        features = {}
        x = generate_smnist_sequence(seq_len, smnist_images, frame_size, num_digits) 
        inputs, targets = np.split(x, [num_input_frames])
        features.update({"inputs":inputs, "targets":targets})
        yield features

#The same thing is used for all
def sminst_input_func(hparams, mode,num_samples = -1):
    num_input_frames = hparams.num_input_frames
    num_target_frames = hparams.num_target_frames
    frame_size = hparams.frame_size
    output_type = {"inputs":tf.float32, "targets":tf.float32}
    output_shape = {"inputs" : tf.TensorShape([num_input_frames] + list(frame_size)),\
        "targets" : tf.TensorShape([num_target_frames] + list(frame_size))}
    generator_func = partial(dataset_generator_func, hparams, mode)
    dataset = tf.data.Dataset.from_generator(generator_func, output_type, output_shape)
    dataset = dataset.take(num_samples)
    dataset = dataset.batch(hparams.batch_size)
    # dataset = dataset.prefetch(4)
    return dataset

def get_feature_columns(hparams):
    num_input_frames = hparams.num_input_frames
    num_target_frames = hparams.num_target_frames
    frame_size = hparams.frame_size    
    feature_columns = [tf.feature_column.numeric_column("inputs", shape = tuple(\
        [num_input_frames] + list(frame_size))), \
        tf.feature_column.numeric_column("targets", shape = tuple(\
        [num_target_frames] + list(frame_size)))]
    return feature_columns
