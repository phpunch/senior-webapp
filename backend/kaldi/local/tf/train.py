#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
import numpy as np
import random 
import os
import sys
from collections import defaultdict, OrderedDict

seed = 0
np.random.seed(seed)
random.seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']=str(seed)

from tensorflow.keras import backend as K 

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
tf.compat.v1.set_random_seed(seed)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
K.set_session(sess)

K.set_floatx('float32')

# import keras
from tensorflow.keras import Model, regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Dense, Conv1D, Bidirectional, Conv2D, CuDNNGRU, GlobalAveragePooling2D, Multiply, Reshape, MaxPooling1D, Input, BatchNormalization, Lambda, LeakyReLU, Concatenate, ReLU, Flatten, Attention, dot, Activation, concatenate
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import TruncatedNormal, Constant
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard

import logging
import argparse
import math

import glob
import sys


from examples_io import *


def conv_block(input_features, kernel_sizes, filters, strides, batchnorm_momentum, name):
    """ ResNet Conv Block with shortcut. The shortcut is projected by an extra conv+bn.
    Args:
        input_features: input features
        kernel_sizes: list. length is 2 or 3
        filters: list.
        strides: The stride applied to the first conv.
        params:
        relu: inherited from the network to define the relu function
        name:
    :return:
    """
    n_comp = len(kernel_sizes)
    assert(n_comp == len(filters) and (n_comp == 2 or n_comp == 3))
    feat = Conv1D(filters[0], kernel_sizes[0], 
                    padding='same', 
                    kernel_initializer=truncatedNormal(), 
                    bias_initializer=biasConstant(),
                    name=name + "_conv0")(input_features)
    # feat = tf.layers.conv2d(input_features,
    #                         filters[0],
    #                         kernel_sizes[0],
    #                         strides=strides,
    #                         padding='same',
    #                         activation=None,
    #                         use_bias=False,
    #                         kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
    #                         name=name+"_conv0")
    feat = BatchNormalization(momentum=batchnorm_momentum, name=name + "_bn0")(feat)
    # feat = ReLU(name=name + '_relu_0')(feat)
    feat = LeakyReLU(alpha=0.1)(feat)

    feat = Conv1D(filters[1], kernel_sizes[1], 
                    padding='same', 
                    kernel_initializer=truncatedNormal(), 
                    bias_initializer=biasConstant(),
                    name=name + "_conv1")(feat)
    # feat = tf.layers.conv2d(feat,
    #                         filters[1],
    #                         kernel_sizes[1],
    #                         padding='same',
    #                         activation=None,
    #                         use_bias=False,
    #                         kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
    #                         name=name+"_conv1")
    feat = BatchNormalization(momentum=batchnorm_momentum, name=name + "_bn1")(feat)

    if n_comp == 3:
        # feat = ReLU(name=name + '_relu_1')(feat)
        feat = LeakyReLU(alpha=0.1)(feat)
        feat = Conv1D(filters[0], kernel_sizes[0], 
                    padding='same', 
                    kernel_initializer=truncatedNormal(), 
                    bias_initializer=biasConstant(),
                    name=name + "_conv2")(feat)
        # feat = tf.layers.conv2d(feat,
        #                         filters[2],
        #                         kernel_sizes[2],
        #                         padding='same',
        #                         activation=None,
        #                         use_bias=False,
        #                         kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
        #                         name=name+"_conv2")
        feat = BatchNormalization(momentum=batchnorm_momentum, name=name + "_bn2")(feat)

    # Shortcut
    # When performing the shortcut projection, the kernel size is 1.
    # The #kernels equal to the last conv and the stride is consistent with the block's stride.
    shortcut = Conv1D(filters[-1], 1, 
                    padding='same', 
                    strides=strides,
                    kernel_initializer=truncatedNormal(), 
                    bias_initializer=biasConstant(),
                    name=name + "_conv_short")(input_features)
    # shortcut = tf.layers.conv2d(input_features,
    #                             filters[-1],
    #                             1,
    #                             strides=strides,
    #                             padding='same',
    #                             activation=None,
    #                             use_bias=False,
    #                             kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
    #                             name=name + "_conv_short")
    shortcut = BatchNormalization(momentum=batchnorm_momentum, name=name + "_bn_short")(shortcut)

    feat = Concatenate(axis=1)([feat, shortcut])
    # feat = ReLU(name=name + '_relu_final')(feat)
    feat = LeakyReLU(alpha=0.1)(feat)
    return feat


def identity_block(input_features, kernel_sizes, filters, batchnorm_momentum, name):
    """ ResNet Conv Block with shortcut. There are no strides so the shortcut does not need an extra conv.
    Args:
        input_features:
        kernel_sizes:
        filters:
        params:
        is_training:
        relu:
        name:
    :return:
    """
    n_comp = len(kernel_sizes)
    assert(n_comp == len(filters) and (n_comp == 2 or n_comp == 3))
    feat = Conv1D(filters[0], kernel_sizes[0], 
                    padding='same', 
                    kernel_initializer=truncatedNormal(), 
                    bias_initializer=biasConstant(),
                    name=name + "_conv0")(input_features)
    # feat = tf.layers.conv2d(input_features,
    #                         filters[0],
    #                         kernel_sizes[0],
    #                         padding='same',
    #                         activation=None,
    #                         use_bias=False,
    #                         kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
    #                         name=name + "_conv0")
    feat = BatchNormalization(momentum=batchnorm_momentum, name=name + "_bn0")(feat)
    # feat = ReLU(name=name + '_relu_0')(feat)
    feat = LeakyReLU(alpha=0.1)(feat)

    feat = Conv1D(filters[1], kernel_sizes[1], 
                    padding='same', 
                    kernel_initializer=truncatedNormal(), 
                    bias_initializer=biasConstant(),
                    name=name+"_conv1")(feat)
    # feat = tf.layers.conv2d(feat,
    #                         filters[1],
    #                         kernel_sizes[1],
    #                         padding='same',
    #                         activation=None,
    #                         use_bias=False,
    #                         kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
    #                         name=name+"_conv1")
    feat = BatchNormalization(momentum=batchnorm_momentum, name=name + "_bn1")(feat)

    if n_comp == 3:
        # feat = ReLU(feat, name=name + '_relu1')
        feat = LeakyReLU(alpha=0.1)(feat)
        feat = Conv1D(filters[2], kernel_sizes[2], 
                    padding='same', 
                    kernel_initializer=truncatedNormal(), 
                    bias_initializer=biasConstant(),
                    name=name+"_conv2")(feat)
        # feat = tf.layers.conv2d(feat,
        #                         filters[2],
        #                         kernel_sizes[2],
        #                         padding='same',
        #                         activation=None,
        #                         use_bias=False,
        #                         kernel_regularizer=tf.contrib.layers.l2_regularizer(params.weight_l2_regularizer),
        #                         name=name+"_conv2")
        feat = BatchNormalization(momentum=batchnorm_momentum, name=name + "_bn2")(feat)

    feat = Concatenate(axis=1)([feat, input_features])
    # feat = ReLU(name=name + '_relu_final')(feat)
    feat = LeakyReLU(alpha=0.1)(feat)
    return feat

def fully_connected_block(feats, name):
    feats = Dense(units=512,
                kernel_initializer=truncatedNormal(), 
                bias_initializer=biasConstant(),
                name=name + '_dense_1')(feats)
    feats = LeakyReLU(alpha=0.2, name=name + '_lrelu_1')(feats)
    feats = BatchNormalization(momentum=0.95, name=name + '_bn_1')(feats)

    feats = Dense(units=750, #############################################################################3 อย่าลืมแก้
                kernel_initializer=truncatedNormal(), 
                bias_initializer=biasConstant(),
                name=name + '_dense_2')(feats)
    feats = LeakyReLU(alpha=0.2, name=name + '_lrelu_2')(feats)
    feats = BatchNormalization(momentum=0.95, name=name + '_bn_2')(feats)
    return feats

class selfAttention(Layer):
    def __init__(self, n_head=5, hidden_dim=1500, penalty=0.1, **kwargs):
        super(selfAttention, self).__init__(**kwargs)
        self.n_head = n_head
        self.P = penalty
        self.hidden_dim = hidden_dim
        
    
    def build(self, input_shape):
        self.W1 = self.add_weight(name='w1', shape=[int(input_shape[2]), self.hidden_dim], initializer='glorot_uniform', # (dh, da)
                                  trainable=True)
        self.W2 = self.add_weight(name='W2', shape=[self.hidden_dim, self.n_head], initializer='glorot_uniform', # (da, dr)
                                  trainable=True)
    
    def call(self, x, **kwargs):
        d1 = K.dot(x, self.W1) # [batch_size, time_step, hidden_dim]
        tanh1 = K.relu(d1)
        d2 = K.dot(tanh1, self.W2)
        softmax1 = K.softmax(d2, axis=0) # [batch_size, time_step, n_head] ---> A.T
        A = K.permute_dimensions(softmax1, (0, 2, 1)) # [batch_size, n_head, time_step]
        emb_mat = K.batch_dot(A, x, axes=[2, 1]) # [batch_size, n_head, time_step] * [batch_size, time_step, feats]
        reshape = Flatten()(emb_mat)
        eye = K.eye(self.n_head)
        prod = K.batch_dot(softmax1, A, axes=[1, 2])
        loss = self.P * K.sqrt(K.sum(K.square(prod - eye)))
        self.add_loss(loss)
        return reshape

class ArcFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.25, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(int(input_shape[0][-1]), self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(K.clip(theta + self.m, 0, np.pi))
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

def truncatedNormal():
    return TruncatedNormal(stddev=0.1, seed=seed)
def biasConstant():
    return Constant(value=0.1)

class MyModel(object):
    def __init__(self, num_speakers=None):
        self.num_speakers = num_speakers
        self.vector_model = None
        self.model = None

    def create_model(self):
        self.model, self.vector_model = self.get_model()
        logger.info(self.model.summary(print_fn=logger.info))

    def create_vector_model(self):
        self.vector_model, _ = self.get_vector_model()



    def get_vector_model(self, inputs=Input(shape=(None, 23))):
            
            x = inputs
            x = K.expand_dims(x, 1)

            # Frame-level
            layer_sizes = [512, 512, 512]
            kernel_sizes = [(1, 5), (1, 5), (1, 7)]
            
            for i, (layer_size, kernel_size) in enumerate(zip(layer_sizes, kernel_sizes)):
                x = Conv2D(layer_size, kernel_size, 
                    padding='same', 
                    kernel_initializer=truncatedNormal(), 
                    bias_initializer=biasConstant(),
                    name='conv{}'.format(i + 1))(x)
                x = LeakyReLU(alpha=0.2)(x)
                x = BatchNormalization(momentum=0.95, name='batchnorm{}'.format(i + 1))(x)

            x = K.squeeze(x, 1)
            x = fully_connected_block(x, 'CNN')
            

            # shortcut = x
            # shortcut = CuDNNLSTM(512, return_sequences=True)(shortcut)
            # shortcut = fully_connected_block(shortcut, 'LSTM')
            # x = fully_connected_block(x, 'CNN')

            # Statistic pooling layer
            # mean_1 = Lambda(lambda x: K.mean(x, axis=1), name="mean_1")(shortcut)
            # std_1 = Lambda(lambda x: K.sqrt(K.var(x, axis=1) + 0.00001), name="std_1")(shortcut)
            # mean_2 = Lambda(lambda x: K.mean(x, axis=1), name="mean_2")(x)
            # std_2 = Lambda(lambda x: K.sqrt(K.var(x, axis=1) + 0.00001), name="std_2")(x)
            # x = Concatenate(axis=1)([mean_1, std_1, mean_2, std_2])

            # Statistic pooling layer
            mean = Lambda(lambda x: K.mean(x, axis=1), name="mean")(x)
            std = Lambda(lambda x: K.sqrt(K.var(x, axis=1) + 0.00001), name="std")(x)
            x = Concatenate(axis=1)([mean, std])
            # x = Bidirectional(CuDNNGRU(750, return_sequences=True))(x)
            # x = selfAttention(n_head=1, hidden_dim=1500, penalty=0.1)(x)

            # Segment-level
            embedding_a = Dense(units=256,
                                kernel_initializer=truncatedNormal(), 
                                bias_initializer=biasConstant(),
                                # activity_regularizer=regularizers.l2(l=0.001),
                                name='dense1')(x)

            embedding_a = LeakyReLU(alpha=0.1)(embedding_a)
            embedding_a = BatchNormalization(momentum=0.95, name='batchnorm_8')(embedding_a)

            utt_vector_model = Model(inputs=inputs, outputs=embedding_a) 

            return utt_vector_model, embedding_a
    
    def get_model(self):
            K.clear_session()
            NUM_CEPSTRUM = 23
            weight_decay = 1e-4

            inputs = Input(shape=(None, NUM_CEPSTRUM))
            
            
            utt_vector_model, embedding_a = self.get_vector_model(inputs)

            embedding_b = Dense(units=512, 
                                kernel_initializer=truncatedNormal(), 
                                bias_initializer=biasConstant(),
                                name='dense2')(embedding_a)#, kernel_regularizer=regularizers.l2(l=0.00002), bias_regularizer=regularizers.l2(l=0.00002))(embedding_a)
            embedding_b = LeakyReLU(alpha=0.2)(embedding_b)
            embedding_b = BatchNormalization(momentum=0.95, name='batchnorm9')(embedding_b)
            
            y = Input(shape=(self.num_speakers,))
            outputs = ArcFace(self.num_speakers, regularizer=regularizers.l2(weight_decay))([embedding_b, y])

            # outputs = Dense(units=self.num_speakers, 
            #                 # kernel_initializer=truncatedNormal(), # ต้องใส่ไหม?
            #                 bias_initializer=biasConstant(),
            #                 activation='softmax', 
            #                 name='softmax')(embedding_b) 
            model = Model(inputs=[inputs, y], outputs=outputs)
            optimizer = Adam(0.001)
            model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
            return model, utt_vector_model


    # def get_lr(self, initial_effective_lrate, final_effective_lrate, cur_itr, num_itr_tot):
    #     if cur_itr + 1 >= num_itr_tot:
    #         effective_learning_rate = final_effective_lrate
    #     else:
    #         effective_learning_rate = (initial_effective_lrate *
    #                                         math.exp(cur_itr *
    #                                                     math.log(final_effective_lrate / initial_effective_lrate)
    #                                                     / num_itr_tot))
    #     return effective_learning_rate
    
    def get_lr(self, current_lrate, initial_effective_lrate, final_effective_lrate, num_itr_tot, reduce_lrate):
        factor_lrate = (final_effective_lrate / initial_effective_lrate) ** (1/num_itr_tot)

        if (current_lrate < final_effective_lrate):
            return final_effective_lrate
        elif (reduce_lrate):
            return current_lrate / 2 if (current_lrate / 2 > final_effective_lrate) else final_effective_lrate
        else:
            return current_lrate * factor_lrate

    def save_model(self, model, path, name):
        if (not os.path.isdir(path)):
            os.mkdir(path)
        # model.save(os.path.join(path, '{}.h5'.format(name)))
        model.save_weights(os.path.join(path, '{}_weights.hdf5'.format(name)))
        # logger.info('{}.h5 is saved.'.format(name))
        logger.info('{}_weights.hdf5 is saved.'.format(name))

    def generator(self, data_loader):
        count = 0
        while True:
            count += 1
            try:
                batch_data, labels = data_loader.pop()
                batch_labels = to_categorical(labels, num_classes=self.num_speakers)
                # print(batch_data.shape)
                # print(labels)
                
                yield [batch_data, batch_labels], batch_labels
            except queue.Empty:
                # logger.info('Timeout reach when reading minibatch', count)
                continue

    def evaluate(self, egs_path, num_valid_archive, num_speakers):
        for i in range(num_valid_archive):
            valid_tar_path = os.path.join(egs_path, 'valid_egs.{}.tar'.format(i + 1))
            valid_data_loader = TarFileDataLoader(valid_tar_path, logger=None, queue_size=64)

            minibatch_count = valid_data_loader.count

            loss = self.model.evaluate_generator(self.generator(valid_data_loader), steps=minibatch_count, workers=1)
        return loss[0]

    def make_embedding(self, input_stream, output_stream, model_path, min_chunk_size, chunk_size, use_gpu, logger):

        num_fail = 0
        num_success = 0
        self.vector_model.load_weights(model_path)

        for key, mat in kaldi_io.read_mat_ark(input_stream):
            # Processing features with key '1089-134686-0000' which have shape '(924, 23)'
            logger.info("Processing features with key '%s' which have shape '%s'" % (key, str(mat.shape)))
            # total_segments += 1

            num_rows = mat.shape[0]
            if num_rows == 0:
                logger.warning("Zero-length utterance: '%s'" % key)
                num_fail += 1
                continue

            if num_rows < min_chunk_size:
                logger.warning("Minimum chunk size of %d is greater than the number of rows in utterance: %s" %
                                (min_chunk_size, key))
                num_fail += 1
                continue
            this_chunk_size = chunk_size
            if num_rows < chunk_size:
                logger.info("Chunk size of %d is greater than the number of rows in utterance: %s, "
                            "using chunk size of %d" % (chunk_size, key, num_rows))
                this_chunk_size = num_rows
            elif chunk_size == -1:
                this_chunk_size = num_rows

            num_chunks = int(np.ceil(num_rows / float(this_chunk_size)))
            logger.info("num_chunks: %d" % num_chunks)
            
            xvector_avg = 0
            tot_weight = 0.0

            for chunk_idx in range(num_chunks): # 1
                # If we're nearing the end of the input, we may need to shift the
                # offset back so that we can get this_chunk_size frames of input to
                # the nnet.
                offset = min(this_chunk_size, num_rows - chunk_idx * this_chunk_size)
                if offset < min_chunk_size:
                    continue
                logger.info("offset: %d" % offset)
                sub_mat = mat[chunk_idx * this_chunk_size: chunk_idx * this_chunk_size + offset, :]
                data = np.reshape(sub_mat, (1, sub_mat.shape[0], sub_mat.shape[1]))
                pred = self.vector_model.predict_on_batch(data) # reshape to fit input_1
                xvector = pred[0]
                tot_weight += offset
                xvector_avg += offset * xvector

            xvector_avg /= tot_weight
            kaldi_io.write_vec_flt(output_stream, xvector_avg, key=key)
            num_success += 1

    def train(self):
        restore_best_weight = False
        best_model = None
        best_vector_model = None

        num_fail = 0
        num_fail_patience = 0
        patience = 10
        reduce_lrate_epochs = 5
        current_loss = np.nan
        current_lrate = initial_effective_lrate
        num_itr_tot = num_train_archive*epochs
        d = defaultdict(int)

        for epoch in range(epochs):
            print("Epoch :", epoch)
            for i in range(num_train_archive):
                logger.info("ITR: \t{}".format(i))
                print("ITR: \t{}".format(i))
                
                tar_path = os.path.join(egs_path, 'egs.{}.tar'.format(i + 1))
                data_loader = TarFileDataLoader(tar_path, logger=None, queue_size=16)

                minibatch_count = data_loader.count
                total_segments = 0
                total_segments_len = 0
                losses = 0
                # count = 0

                lr = current_lrate
                logger.info("Learning Rate: \t{}".format(lr))
                print("Learning Rate: \t{}".format(lr))
                K.set_value(self.model.optimizer.lr, lr)

                history = self.model.fit_generator(self.generator(data_loader), steps_per_epoch=minibatch_count, use_multiprocessing=False, workers=1)
                losses += history.history['loss'][0]

                training_loss = losses
                logger.info("Training Loss: \t{}".format(training_loss))
                logger.info("total_segments : \t{}".format(total_segments))
                logger.info("total_segments_len : \t{}".format(total_segments_len))

                validation_loss = self.evaluate(egs_path, num_valid_archive, num_speakers)
                logger.info("Validation Loss: \t{}".format(validation_loss))
                print("Validation Loss: \t{}".format(validation_loss))
                if (validation_loss >= current_loss):
                    num_fail += 1
                    num_fail_patience += 1
                    print("Reduce Validation loss ... fail : {}".format(num_fail))
                    print("Reduce Validation loss ... fail : {}".format(num_fail_patience))
                    logger.info("Reduce Validation loss ... fail : {}".format(num_fail))
                    
                    if (num_fail >= reduce_lrate_epochs):
                        current_lrate = self.get_lr(current_lrate, initial_effective_lrate, final_effective_lrate, num_itr_tot, True)
                        num_fail = 0
                    else:
                        current_lrate = self.get_lr(current_lrate, initial_effective_lrate, final_effective_lrate, num_itr_tot, False)

                    

                    if (restore_best_weight and best_model is not None):
                        print("Restore best weight")
                        self.model = best_model
                        self.vector_model  = best_vector_model
                else:
                    num_fail = 0
                    num_fail_patience = 0
                    current_loss = validation_loss
                    best_model = self.model
                    best_vector_model = self.vector_model
                    current_lrate = self.get_lr(current_lrate, initial_effective_lrate, final_effective_lrate, num_itr_tot, False)

                    self.save_model(self.model, model_path, '{}_best'.format(ver))
                    self.save_model(self.vector_model, model_path, '{}_vector_best'.format(ver))
                print("========================================================================")

                # EarlyStop and exit the program
                if (num_fail_patience >= patience):
                    self.save_model(best_model, model_path, '{}_best'.format(ver))
                    self.save_model(best_vector_model, model_path, '{}_vector_best'.format(ver))
                    sys.exit()

            self.save_model(self.model, model_path, '{}_epoch{}'.format(ver, epoch))
            self.save_model(self.vector_model, model_path, '{}_vector_epoch{}'.format(ver, epoch))
            
        logger.info("Dict {}".format(OrderedDict(sorted(d.items()))))
        self.save_model(self.model, model_path, '{}_best'.format(ver))
        self.save_model(self.vector_model, model_path, '{}_vector_best'.format(ver))


        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_speakers", help="the number of speakers", type=int)
    parser.add_argument("--ver", help="the model version", type=int)
    args = parser.parse_args()
    
    # constant
    model_path = os.path.join('exp/pvector_net/models')
    egs_path = os.path.join('exp/pvector_net/egs')
    num_train_archive = len(glob.glob(egs_path + '/egs.*.npy'))
    num_valid_archive = len(glob.glob(egs_path + '/valid_egs.*.npy'))
    ver = args.ver
    initial_effective_lrate = 0.001
    final_effective_lrate = 1e-6
    epochs = 6
    num_speakers = args.num_speakers
    
    assert num_speakers > 0 and num_train_archive > 0 and num_valid_archive > 0, "AssertError"
    # Create and configure logger
    LOG_FORMAT = "%(levelname)s %(asctime)s - %(message)s"
    logging.basicConfig(filename="train_logs/trainer_{}.log".format(ver), level=logging.DEBUG, format=LOG_FORMAT, filemode='w')
    logger = logging.getLogger()

    logger.info("Version : \t{}".format(ver))
    print("Version : \t{}".format(ver))
    logger.info("Egs path : \t{}".format(egs_path))
    logger.info("Number of training archives : \t{}".format(num_train_archive))
    logger.info("Number of valid archives : \t{}".format(num_valid_archive))
    logger.info("num_speakers: \t{}".format(num_speakers))
    print("Egs path : \t{}".format(egs_path))
    print("Number of training archives : \t{}".format(num_train_archive))
    print("Number of valid archives : \t{}".format(num_valid_archive))
    print("num_speakers: \t{}".format(num_speakers))
    
    my_model = MyModel(num_speakers)
    my_model.create_model()
    my_model.train()