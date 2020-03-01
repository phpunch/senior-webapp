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
from tensorflow.keras import backend, Model, regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv1D, GlobalAveragePooling1D, Dropout, Conv1D, MaxPooling1D, Input, BatchNormalization, Lambda, LeakyReLU, Concatenate
from tensorflow.keras.optimizers import Adam, SGD
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

class MyModel(object):

    def __init__(self, MFCC_MAX_LEN=None):
        self.MFCC_MAX_LEN = MFCC_MAX_LEN
        self.NUM_CEPSTRUM = 23
        self.zero_pad = False

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

    # def clean_mfcc(self, mfcc):
    #     if (self.MFCC_MAX_LEN > mfcc.shape[1]):
    #         # padding 
    #         pad_width = self.MFCC_MAX_LEN - mfcc.shape[1]
    #         mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    #     else:
    #         # cutoff
    #         mfcc = mfcc[:,:self.MFCC_MAX_LEN]
    #     return mfcc

    def pad_mfcc(self, mfcc):
        """
            preprocess mfcc by zero padding or repeat features
        """
        if (self.MFCC_MAX_LEN > mfcc.shape[0]):
            # padding 
            pad_width = self.MFCC_MAX_LEN - mfcc.shape[0]
            if (self.zero_pad):
                mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
            else:
                num_frames = mfcc.shape[0]
                num_repeat = self.MFCC_MAX_LEN // num_frames + 1 #if self.MFCC_MAX_LEN % num_frames == 0 else self.MFCC_MAX_LEN // num_frames + 1
                # repeat features
                mfcc = np.tile(mfcc, (num_repeat,1))
                mfcc = mfcc[:self.MFCC_MAX_LEN,:]
        else:
            # cutoff
            mfcc = mfcc[:self.MFCC_MAX_LEN,:]
        return mfcc

    

    def get_model(self, num_speakers):
        leaky_relu_rate = 0.2

        # This returns a tensor
        inputs = Input(shape=(self.MFCC_MAX_LEN, self.NUM_CEPSTRUM))
        
        # Frame-level
        x = Conv1D(512, 5, padding='valid', name='conv1')(inputs)
        x = LeakyReLU(alpha=leaky_relu_rate)(x)
        #
        intermediate_layer  = Model(inputs=inputs, outputs=x)
    
        x = BatchNormalization(name='batchnorm1')(x)

        x = Conv1D(512, 5, padding='valid', dilation_rate=1 , name='conv2')(x)
        x = LeakyReLU(alpha=leaky_relu_rate)(x)
        x = BatchNormalization(name='batchnorm2')(x)
        
        x = Conv1D(512, 7, padding='valid', dilation_rate=1, name='conv3')(x)
        x = LeakyReLU(alpha=leaky_relu_rate)(x)
        x = BatchNormalization(name='batchnorm3')(x)
        
        x = Conv1D(512, 1, padding='valid', dilation_rate=1, name='conv4')(x)
        x = LeakyReLU(alpha=leaky_relu_rate)(x)
        x = BatchNormalization(name='batchnorm4')(x)
        
        x = Conv1D(512 * 3, 1, padding='valid', dilation_rate=1, name='conv5')(x)
        x = LeakyReLU(alpha=leaky_relu_rate)(x)
        x = BatchNormalization(name='batchnorm5')(x)
        

        # Statistic pooling layer
        mean = Lambda(lambda x: K.mean(x, axis=1), name="mean")(x)
        std = Lambda(lambda x: K.sqrt(K.var(x, axis=1) + 0.00001), name="std")(x)
        x = Concatenate(axis=1)([mean, std])

        
        # Segment-level
        embedding_a = Dense(units=512, name='dense1')(x)#, kernel_regularizer=regularizers.l2(l=0.00002), bias_regularizer=regularizers.l2(l=0.00002))(x)
        embedding_a = LeakyReLU(alpha=leaky_relu_rate)(embedding_a)
        embedding_a = BatchNormalization(name='batchnorm6')(embedding_a)
        
        utt_vector_model = Model(inputs=inputs, outputs=embedding_a) 

        
        embedding_b = Dense(units=512, name='dense2')(embedding_a)#, kernel_regularizer=regularizers.l2(l=0.00002), bias_regularizer=regularizers.l2(l=0.00002))(embedding_a)
        embedding_b = LeakyReLU(alpha=leaky_relu_rate)(embedding_b)
        embedding_b = BatchNormalization(name='batchnorm7')(embedding_b)
        
        
        outputs = Dense(units=num_speakers, activation='softmax', name='softmax')(embedding_b) 
        model = Model(inputs=inputs, outputs=outputs)

        optimizer = SGD(0.01, decay=1e-5, momentum=0.5)
        model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        return model, utt_vector_model, intermediate_layer


        # # This returns a tensor
        # inputs = Input(shape=(self.MFCC_MAX_LEN, self.NUM_CEPSTRUM))
        
        # # Frame-level
        # x = Conv1D(512, 5, padding='valid', name='conv1')(inputs)

        # #
        # intermediate_layer  = Model(inputs=inputs, outputs=x)

        # x = BatchNormalization(name='batchnorm1')(x)
        # #x = Dropout(0.3)(x)
        # x = Conv1D(512, 7, padding='valid', activation='relu', dilation_rate=2 , name='conv2')(x)
        # x = BatchNormalization(name='batchnorm2')(x)
        # #x = Dropout(0.3)(x)
        # x = Conv1D(512, 7, padding='valid', activation='relu', dilation_rate=3, name='conv3')(x)
        # x = BatchNormalization(name='batchnorm3')(x)
        # #x = Dropout(0.3)(x)
        # x = Conv1D(512, 1, padding='valid', activation='relu', dilation_rate=3, name='conv4')(x)
        # x = BatchNormalization(name='batchnorm4')(x)
        # #x = Dropout(0.3)(x)
        # x = Conv1D(512 * 3, 1, padding='valid', activation='relu', dilation_rate=4, name='conv5')(x)
        # x = BatchNormalization(name='batchnorm5')(x)
        # # x = print_layer(x, "After Batch=")

        # #x = Dropout(0.3)(x) 

        # x = GlobalAveragePooling1D(name='globalaveragepooling1d')(x)
        
        # # Segment-level
        # embedding_a = Dense(units=512, activation='relu', name='dense1')(x)
        
        # utt_vector_model = Model(inputs=inputs, outputs=embedding_a) 

        
        
        # #embedding_a = Dropout(0.3)(embedding_a)
        # embedding_b = Dense(units=512, activation='relu', name='dense2')(embedding_a)
        # #embedding_b = Dropout(0.3)(embedding_b)
        
        # outputs = Dense(units=num_speakers, activation='softmax', name='softmax')(embedding_b) 
        # model = Model(inputs=inputs, outputs=outputs)
        # model.compile(optimizer=Adam(0.001),
        #             loss='categorical_crossentropy',
        #             metrics=['accuracy'])
        # return model, utt_vector_model, intermediate_layer

    # def get_vector_model(self):
    #     # This returns a tensor
    #     inputs = Input(shape=(self.MFCC_MAX_LEN, self.NUM_CEPSTRUM))
        
    #     # Frame-level
    #     x = Conv1D(512, 5, padding='valid', activation='relu')(inputs)
    #     x = BatchNormalization()(x)
    #     x = Dropout(0.3)(x)
    #     x = Conv1D(512, 3, padding='valid', activation='relu', dilation_rate=2)(x)
    #     x = BatchNormalization()(x)
    #     x = Dropout(0.3)(x)
    #     x = Conv1D(512, 3, padding='valid', activation='relu', dilation_rate=3)(x)
    #     x = BatchNormalization()(x)
    #     x = Dropout(0.3)(x)
    #     x = Conv1D(512, 3, padding='valid', activation='relu', dilation_rate=3)(x)
    #     x = BatchNormalization()(x)
    #     x = Dropout(0.3)(x)
    #     x = Conv1D(512, 3, padding='valid', activation='relu', dilation_rate=4)(x)
    #     x = BatchNormalization()(x)
    #     x = Dropout(0.3)(x)
    #     x = GlobalAveragePooling1D()(x)
        
    #     # Segment-level
    #     embedding_a = Dense(units=512, activation='relu')(x)
        
    #     utt_vector_model = Model(inputs=inputs, outputs=embedding_a) 
        
    #     # embedding_a = Dropout(0.3)(embedding_a)
    #     # embedding_b = Dense(units=512, activation='relu')(embedding_a)
    #     # embedding_b = Dropout(0.3)(embedding_b)
        
    #     # outputs = Dense(units=num_speakers, activation='softmax')(embedding_b) 
    #     # model = Model(inputs=inputs, outputs=outputs)
    #     # model.compile(optimizer=Adam(0.001),
    #     #             loss='categorical_crossentropy',
    #     #             metrics=['accuracy'])
    #     return utt_vector_model #, model

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

        # total_segments = 0
        # num_fail = 0
        num_success = 0

        model = load_model(model_path,custom_objects={
            "backend": backend,
        })

        for key, mat in kaldi_io.read_mat_ark(input_stream):
            # Processing features with key '1089-134686-0000' which have shape '(924, 23)'
            logger.info("Processing features with key '%s' which have shape '%s'" % (key, str(mat.shape)))
            # total_segments += 1

            # num_rows = mat.shape[0]
            # if num_rows == 0:
            #     logger.warning("Zero-length utterance: '%s'" % key)
            #     num_fail += 1
            #     continue

            # if num_rows < min_chunk_size:
            #     logger.warning("Minimum chunk size of %d is greater than the number of rows in utterance: %s" %
            #                     (min_chunk_size, key))
            #     num_fail += 1
            #     continue
            # this_chunk_size = chunk_size
            # if num_rows < chunk_size:
            #     logger.info("Chunk size of %d is greater than the number of rows in utterance: %s, "
            #                 "using chunk size of %d" % (chunk_size, key, num_rows))
            #     this_chunk_size = num_rows
            # elif chunk_size == -1:
            #     this_chunk_size = num_rows

            # num_chunks = int(np.ceil(num_rows / float(this_chunk_size)))
            # logger.info("num_chunks: %d" % num_chunks)
            
            # print("!!!!!!! CURRENT DIR !!!!!!!!")
            # print(os.getcwd())
            # model = self.get_model()
            
            
            # scaled_mat = scaler.transform(mat)
            # print(scaled_mat)
            # clean_mat = self.pad_mfcc(scaled_mat)
            clean_mat = mat[np.newaxis, :]
            pred = model.predict_on_batch(clean_mat) # reshape to fit input_1
            # print("pred")
            # print(pred)
            pred = pred.reshape(512, )
            

            # for chunk_idx in range(num_chunks): # 1
            #     # If we're nearing the end of the input, we may need to shift the
            #     # offset back so that we can get this_chunk_size frames of input to
            #     # the nnet.
            #     offset = min(this_chunk_size, num_rows - chunk_idx * this_chunk_size)
            #     if offset < min_chunk_size:
            #         continue
            #     logger.info("offset: %d" % offset)
            #     sub_mat = mat[chunk_idx * this_chunk_size: chunk_idx * this_chunk_size + offset, :]
            #     data = np.reshape(sub_mat, (1, sub_mat.shape[0], sub_mat.shape[1]))
            #     total_segments_len += sub_mat.shape[0]
            #     feed_dict = {self.input_x: data, self.dropout_keep_prob: 1.0, self.phase: False}
            #     gpu_waiting = time.time()
            #     xvector = sess.run(self.embedding[0], feed_dict=feed_dict)
            #     xvector = xvector[0]
            #     # logger.info("xvector: %s" % str(xvector.shape))
            #     total_gpu_waiting += time.time() - gpu_waiting
            #     tot_weight += offset
            #     xvector_avg += offset * xvector

            # xvector_avg /= tot_weight
            kaldi_io.write_vec_flt(output_stream, pred, key=key)
            num_success += 1

        # logger.info("Processed %d features of average size %d frames. Done %d and failed %d" %
        #             (total_segments, total_segments_len / total_segments, num_success, num_fail))

        # logger.info("Total time for neural network computations is %.2f minutes." %
        #             (total_gpu_waiting / 60.0))

        # logger.info("Elapsed time for extracting whole embeddings is %.2f minutes." %
        #             ((time.time() - start_time) / 60.0))

        ###########################################################################################################################################




###########################################################################################################################3

class ModelTF(object):

    def __init__(self):
        self.graph = None
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

    def load_model(self, sess, input_dir, logger):
        if logger is not None:
            logger.info("Start loading graph ...")
        saver = tf.train.import_meta_graph(os.path.join(input_dir, 'model.meta'))
        saver.restore(sess, os.path.join(input_dir, 'model'))
        self.graph = sess.graph
        self.input_x = self.graph.get_tensor_by_name("input_x:0")
        self.input_y = self.graph.get_tensor_by_name("input_y:0")
        self.num_classes = self.input_y.shape[1]
        self.learning_rate = self.graph.get_tensor_by_name("learning_rate:0")
        self.dropout_keep_prob = self.graph.get_tensor_by_name("dropout_keep_prob:0")
        self.phase = self.graph.get_tensor_by_name("phase:0")
        self.loss = self.graph.get_tensor_by_name("loss:0")
        self.optimizer = self.graph.get_operation_by_name("optimizer")
        self.accuracy = self.graph.get_tensor_by_name("accuracy/accuracy:0")
        self.embedding = [None] * 2  # TODO make this more general
        self.embedding[0] = self.graph.get_tensor_by_name("embed_layer-0/scores:0")
        self.embedding[1] = self.graph.get_tensor_by_name("embed_layer-1/scores:0")
        if logger is not None:
            logger.info("Graph restored from path: %s" % input_dir)

    def make_embedding(self, input_stream, output_stream, model_dir, min_chunk_size, chunk_size, use_gpu, logger):

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        if not use_gpu:
            config.intra_op_parallelism_threads = 2
            config.inter_op_parallelism_threads = 2

        with tf.Session(config=config) as sess:
            self.load_model(sess, model_dir, logger)

            total_segments = 0
            total_segments_len = 0
            total_gpu_waiting = 0.0
            num_fail = 0
            num_success = 0
            for key, mat in kaldi_io.read_mat_ark(input_stream):
                logger.info("Processing features with key '%s' which have shape '%s'" % (key, str(mat.shape)))
                total_segments += 1

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
                this_chunk_size = chunk_size # length
                if num_rows < chunk_size:
                    logger.info("Chunk size of %d is greater than the number of rows in utterance: %s, "
                                "using chunk size of %d" % (chunk_size, key, num_rows))
                    this_chunk_size = num_rows
                elif chunk_size == -1:
                    this_chunk_size = num_rows

                num_chunks = int(np.ceil(num_rows / float(this_chunk_size)))
                # logger.info("num_chunks: %d" % num_chunks)
                xvector_avg = 0
                tot_weight = 0.0

                for chunk_idx in range(num_chunks):
                    # If we're nearing the end of the input, we may need to shift the
                    # offset back so that we can get this_chunk_size frames of input to
                    # the nnet.
                    offset = min(this_chunk_size, num_rows - chunk_idx * this_chunk_size)
                    if offset < min_chunk_size:
                        continue
                    # logger.info("offset: %d" % offset)
                    sub_mat = mat[chunk_idx * this_chunk_size: chunk_idx * this_chunk_size + offset, :]
                    data = np.reshape(sub_mat, (1, sub_mat.shape[0], sub_mat.shape[1]))
                    total_segments_len += sub_mat.shape[0]
                    feed_dict = {self.input_x: data, self.dropout_keep_prob: 1.0, self.phase: False}
                    xvector = sess.run(self.embedding[0], feed_dict=feed_dict)
                    xvector = xvector[0]
                    # logger.info("xvector: %s" % str(xvector.shape))
                    tot_weight += offset
                    xvector_avg += offset * xvector

                xvector_avg /= tot_weight
                kaldi_io.write_vec_flt(output_stream, xvector_avg, key=key)
                num_success += 1