import kaldi_io
from time import time
import numpy as np
# from keras.models import Sequential, Model, load_model
# from keras.layers import Dense, Conv1D, GlobalAveragePooling1D, Dropout, Conv1D, MaxPooling1D, Input, BatchNormalization, Lambda 
# from keras.optimizers import Adam, SGD
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint, LambdaCallback, TensorBoard
# from keras.utils.vis_utils import plot_model
# from keras.layers.advanced_activations import LeakyReLU
# from keras import backend as K 
# import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import backend as K, Model, regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv1D, GlobalAveragePooling1D, Dropout, Conv1D, MaxPooling1D, Input, BatchNormalization, Lambda, ReLU, Concatenate, Layer, Flatten, Attention
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import TruncatedNormal, Constant
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
import pickle

def load_obj(obj_path, name):
    with open(obj_path + name + '.pkl', 'rb') as f:
        return pickle.load(f)
# def get_lr_metric(optimizer):
#     def lr(y_true, y_pred):
#         return optimizer.lr
#     return lr


class selfAttention(Layer):
    def __init__(self, n_head=5, hidden_dim=1536, penalty=0.1, **kwargs):
        super(selfAttention, self).__init__(**kwargs)
        self.n_head = n_head
        self.P = penalty
        self.hidden_dim = hidden_dim
        
    
    def build(self, input_shape):
        self.W1 = self.add_weight(name='w1', shape=[int(input_shape[2]), self.hidden_dim], initializer='glorot_uniform',
                                  trainable=True)
        self.W2 = self.add_weight(name='W2', shape=[self.hidden_dim, self.n_head], initializer='glorot_uniform',
                                  trainable=True)
    
    def call(self, x, **kwargs):
        d1 = K.dot(x, self.W1) # [batch_size, time_step, hidden_dim]
        tanh1 = K.relu(d1)
        d2 = K.dot(tanh1, self.W2)
        softmax1 = K.softmax(d2, axis=0) # [batch_size, time_step, n_head]
        A = K.permute_dimensions(softmax1, (0, 2, 1)) # [batch_size, n_head, time_step]
        emb_mat = K.batch_dot(A, x, axes=[2, 1]) # [batch_size, n_head, time_step] * [batch_size, time_step, feats]
        reshape = Flatten()(emb_mat)
        eye = K.eye(self.n_head)
        prod = K.batch_dot(softmax1, A, axes=[1, 2])
        self.add_loss(self.P * K.sqrt(K.sum(K.square(prod - eye))))
        return reshape

class MyModel(object):

    def __init__(self, MFCC_MAX_LEN=None):
        self.MFCC_MAX_LEN = MFCC_MAX_LEN
        self.NUM_CEPSTRUM = 23
        self.zero_pad = False

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        self.model = self.get_vector_model()

    def truncatedNormal(self):
        return TruncatedNormal(stddev=0.1, seed=0)
    def biasConstant(self):
        return Constant(value=0.1)

    def get_vector_model(self):
        # This returns a tensor
        inputs = Input(shape=(None, self.NUM_CEPSTRUM))
        x = inputs
        
        # Frame-level
        layer_sizes = [512, 512, 512, 512, 512*3]
        kernel_sizes = [5, 5, 7, 1, 1]
        embedding_sizes = [512, 512]
        
        for i, (layer_size, kernel_size) in enumerate(zip(layer_sizes, kernel_sizes)):
            x = Conv1D(layer_size, kernel_size, 
                   padding='same', 
                   kernel_initializer=self.truncatedNormal(), 
                   bias_initializer=self.biasConstant(),
                   name='conv{}'.format(i + 1))(x)
            x = ReLU()(x)
            x = BatchNormalization(momentum=0.95, name='batchnorm{}'.format(i + 1))(x)
            
        

        # Statistic pooling layer
        # mean = Lambda(lambda x: K.mean(x, axis=1), name="mean")(x)
        # std = Lambda(lambda x: K.sqrt(K.var(x, axis=1) + 0.00001), name="std")(x)
        # x = Concatenate(axis=1)([mean, std])
        
        # Self attention
        # x = selfAttention()(x)
        
        # x = attention_3d_block(x)

        # x = Attention(use_scale=True, causal=True)([x, x])
        mean = Lambda(lambda x: K.mean(x, axis=1), name="mean")(x)
        std = Lambda(lambda x: K.sqrt(K.var(x, axis=1) + 0.00001), name="std")(x)
        x = Concatenate(axis=1)([mean, std])
        print(x.shape)
        # Segment-level
        embedding_a = Dense(units=512,
                            kernel_initializer=self.truncatedNormal(), 
                            bias_initializer=self.biasConstant(),
                            name='dense1')(x)#, kernel_regularizer=regularizers.l2(l=0.00002), bias_regularizer=regularizers.l2(l=0.00002))(x)
        
        embedding_a = ReLU()(embedding_a)
        embedding_a = BatchNormalization(momentum=0.95, name='batchnorm6')(embedding_a)
        
        utt_vector_model = Model(inputs=inputs, outputs=embedding_a) 

        return utt_vector_model


    def make_embedding(self, input_stream, output_stream, model_path, min_chunk_size, chunk_size, use_gpu, logger):
        

        # start_time = time()
        # # set_cuda_visible_devices(use_gpu=use_gpu, logger=logger)

        # # config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # # if not use_gpu:
        # #     config.intra_op_parallelism_threads = 2
        # #     config.inter_op_parallelism_threads = 2

        # # with tf.Session(config=config) as sess:
        # # self.load_model(sess, model_dir, logger)

        # # total_segments = 0
        # # num_fail = 0
        # num_success = 0
        # scaler = load_obj('/media/punch/DriveE/linux/data_cmvn/obj_input/', 'scaler')

        # model = load_model("exp/pvector_net/vector_epoch2.h5")

        # for key, mat in kaldi_io.read_mat_ark(input_stream):
        #     # Processing features with key '1089-134686-0000' which have shape '(924, 23)'
        #     logger.info("Processing features with key '%s' which have shape '%s'" % (key, str(mat.shape)))
        #     # total_segments += 1

        #     # num_rows = mat.shape[0]
        #     # if num_rows == 0:
        #     #     logger.warning("Zero-length utterance: '%s'" % key)
        #     #     num_fail += 1
        #     #     continue

        #     # if num_rows < min_chunk_size:
        #     #     logger.warning("Minimum chunk size of %d is greater than the number of rows in utterance: %s" %
        #     #                     (min_chunk_size, key))
        #     #     num_fail += 1
        #     #     continue
        #     # this_chunk_size = chunk_size
        #     # if num_rows < chunk_size:
        #     #     logger.info("Chunk size of %d is greater than the number of rows in utterance: %s, "
        #     #                 "using chunk size of %d" % (chunk_size, key, num_rows))
        #     #     this_chunk_size = num_rows
        #     # elif chunk_size == -1:
        #     #     this_chunk_size = num_rows

        #     # num_chunks = int(np.ceil(num_rows / float(this_chunk_size)))
        #     # logger.info("num_chunks: %d" % num_chunks)
            
        #     # print("!!!!!!! CURRENT DIR !!!!!!!!")
        #     # print(os.getcwd())
        #     # model = self.get_model()
            
            
        #     scaled_mat = scaler.transform(mat)
        #     # print(scaled_mat)
        #     clean_mat = self.pad_mfcc(scaled_mat)
        #     clean_mat = clean_mat[np.newaxis, :]
        #     pred = model.predict(clean_mat) # reshape to fit input_1
        #     # print("pred")
        #     # print(pred)
        #     pred = pred.reshape(512, )
            

        #     # for chunk_idx in range(num_chunks): # 1
        #     #     # If we're nearing the end of the input, we may need to shift the
        #     #     # offset back so that we can get this_chunk_size frames of input to
        #     #     # the nnet.
        #     #     offset = min(this_chunk_size, num_rows - chunk_idx * this_chunk_size)
        #     #     if offset < min_chunk_size:
        #     #         continue
        #     #     logger.info("offset: %d" % offset)
        #     #     sub_mat = mat[chunk_idx * this_chunk_size: chunk_idx * this_chunk_size + offset, :]
        #     #     data = np.reshape(sub_mat, (1, sub_mat.shape[0], sub_mat.shape[1]))
        #     #     total_segments_len += sub_mat.shape[0]
        #     #     feed_dict = {self.input_x: data, self.dropout_keep_prob: 1.0, self.phase: False}
        #     #     gpu_waiting = time.time()
        #     #     xvector = sess.run(self.embedding[0], feed_dict=feed_dict)
        #     #     xvector = xvector[0]
        #     #     # logger.info("xvector: %s" % str(xvector.shape))
        #     #     total_gpu_waiting += time.time() - gpu_waiting
        #     #     tot_weight += offset
        #     #     xvector_avg += offset * xvector

        #     # xvector_avg /= tot_weight
        #     kaldi_io.write_vec_flt(output_stream, pred, key=key)
        #     num_success += 1

        # # logger.info("Processed %d features of average size %d frames. Done %d and failed %d" %
        # #             (total_segments, total_segments_len / total_segments, num_success, num_fail))

        # # logger.info("Total time for neural network computations is %.2f minutes." %
        # #             (total_gpu_waiting / 60.0))

        # # logger.info("Elapsed time for extracting whole embeddings is %.2f minutes." %
        # #             ((time.time() - start_time) / 60.0))

        ##########################################################################################################################################3

        start_time = time()
        # set_cuda_visible_devices(use_gpu=use_gpu, logger=logger)

        # config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # if not use_gpu:
        #     config.intra_op_parallelism_threads = 2
        #     config.inter_op_parallelism_threads = 2

        # with tf.Session(config=config) as sess:
        # self.load_model(sess, model_dir, logger)

        num_fail = 0
        num_success = 0

        # model = load_model(model_path,custom_objects={
        #     "K": backend,
        #     'Attention': Attention(use_scale=True, causal=True)
        # })

        self.model.load_weights(model_path)

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
            
            
        
            # clean_mat = mat[np.newaxis, :]
            # pred = self.model.predict_on_batch(clean_mat) # reshape to fit input_1
            # # print("pred")
            # # print(pred)
            # pred = pred.reshape(512, )
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
                # feed_dict = {self.input_x: data, self.dropout_keep_prob: 1.0, self.phase: False}
                # gpu_waiting = time.time()
                pred = self.model.predict_on_batch(data) # reshape to fit input_1

                # xvector = sess.run(self.embedding[0], feed_dict=feed_dict)
                # pred = pred.reshape(512, )
                xvector = pred[0]
                # logger.info("xvector: %s" % str(xvector.shape))
                tot_weight += offset
                xvector_avg += offset * xvector

            xvector_avg /= tot_weight
            kaldi_io.write_vec_flt(output_stream, xvector_avg, key=key)
            num_success += 1

        # logger.info("Processed %d features of average size %d frames. Done %d and failed %d" %
        #             (total_segments, total_segments_len / total_segments, num_success, num_fail))

        # logger.info("Total time for neural network computations is %.2f minutes." %
        #             (total_gpu_waiting / 60.0))

        # logger.info("Elapsed time for extracting whole embeddings is %.2f minutes." %
        #             ((time.time() - start_time) / 60.0))

        ###########################################################################################################################################

