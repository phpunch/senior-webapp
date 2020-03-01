import h5py
import numpy as np
import kaldi_io
import io
import tarfile
import time
import queue
from threading import Thread
from kaldi_io import read_mat
from pprint import pprint

def process_range_file(range_file_path, minibatch_count, minibatch_size, logger=None):
    if logger is not None:
        logger.info('Start processing range file ...')
    fid = open(range_file_path, 'rt')
    utt_to_chunks = {}
    minibatch_info = np.ndarray(minibatch_count, dtype=object)
    print(minibatch_info[0])
    for line in fid:
        parts = line[:-1].split(' ') # 100-121669-0014 192 8 5 287 0
        utt_id = parts[0]
        minibatch_index, offset, length, label = int(parts[1]), int(parts[3]), int(parts[4]), int(parts[5]) # 1066-5330-0016-babble [192] 39 [193] [287] [20]
                                                                                                            # [192] -> minibatch_index ซึ่งถูกสุ่มให้ไปอยู่กับ utterance อื่นๆแล้วเช่น 1116-132847-0032-music
        chunk = (minibatch_index, offset, length, label)
        if utt_id in utt_to_chunks:
            utt_to_chunks[utt_id].append(chunk) # 1081 -> chunk
        else:
            utt_to_chunks[utt_id] = [chunk]
        if minibatch_info[minibatch_index] is not None:
            minibatch_info[minibatch_index][0] += length
            assert minibatch_info[minibatch_index][1] == length
        else:
            minibatch_info[minibatch_index] = [length, length, 0] # [15552, 243, 0] -> ช่องแรกคือ cumulative length 243*64 (length * batch_size) 
    fid.close()
    
    for total_len, segment_size, index in minibatch_info:
        mini_size = total_len / segment_size
        assert mini_size % minibatch_size == 0 and mini_size >= minibatch_size
    if logger is not None:
        logger.info('Processing range file "%s" just finished.' % range_file_path)
    return utt_to_chunks, minibatch_info


def load_ranges_data(utt_to_chunks, minibatch_info, minibatch_size, scp_file_path, fea_dim, logger=None):
    num_err, num_done = 0, 0
    if logger is not None:
        logger.info('Start allocating memories for loading training examples ...')
    all_data = np.ndarray(len(minibatch_info), dtype=object)
    labels = np.ndarray(len(minibatch_info), dtype=object)
    for i in range(len(minibatch_info)): # initialize
        all_data[i] = np.zeros((minibatch_size, minibatch_info[i][1], fea_dim), dtype=np.float32) # shape = (64, length, 23)
        labels[i] = np.zeros(minibatch_size, dtype=np.int32) # shape (64)
    if logger is not None:
        logger.info('Start loading training examples to the memory ...')
    for key, mat in kaldi_io.read_mat_scp(scp_file_path):
        got = utt_to_chunks.get(key)
        if key is None:
            if logger is not None:
                logger.info("Could not create examples from utterance '%s' "
                            "because it has no entry in the ranges input file." % key)
            num_err += 1
        else:
            num_done += 1
            for minibatch_index, offset, length, label in got:
                info = minibatch_info[minibatch_index] # [15552, 243, 0]
                mm = mat[offset:offset + length, :]
                dat = all_data[minibatch_index]
                assert dat.shape[1] == mm.shape[0] and dat.shape[2] == mm.shape[1]
                dat[info[2], :, :] = mm
                labels[minibatch_index][info[2]] = label
                info[2] += 1
    if logger is not None:
        logger.info('Loading features finished with {0} errors and {1} success from total {2} files.'.
                    format(num_err, num_done, num_err + num_done))
    return all_data, labels


def load_scp2dic(scp_file):
    fid = open(scp_file)
    out_dic = {}
    try:
        for line in fid:
            _line = line.strip()
            # at least 3 chars must be in a line
            if len(_line) < 3:
                continue
            (key, read_info) = _line.split(' ')
            out_dic[key] = read_info
    finally:
        fid.close()
    return out_dic


def __read_scp(scp_file):
    fid = open(scp_file)
    try:
        for line in fid:
            (key, read_info) = line.strip().split(' ')
            read_info = \
                read_info.replace('/mnt/matylda6/zeinali/kaldi-trunk/egs/sre16/v2/exp/all_train_no_sil/',
                                  '/mnt/scratch03/tmp/zeinali/kaldi-trunk/egs/sre16/v2/exp/all_train_no_sil/')
            yield key, read_info
    finally:
        fid.close()


def load_ranges_info(utt_to_chunks, minibatch_info, minibatch_size, scp_file_path, fea_dim, logger=None):
    num_err, num_done = 0, 0
    if logger is not None:
        logger.info('Start allocating memories for loading training examples ...')
    all_data_info = np.ndarray(len(minibatch_info), dtype=object)
    labels = np.zeros(len(minibatch_info), dtype=object)
    for i in range(len(minibatch_info)):
        all_data_info[i] = []
        labels[i] = np.zeros(minibatch_size, dtype=np.int32)
    if logger is not None:
        logger.info('Start loading training examples to the memory ...')
    for key, read_info in __read_scp(scp_file_path):
        got = utt_to_chunks.get(key)
        if key is None:
            if logger is not None:
                logger.info("Could not create examples from utterance '%s' "
                            "because it has no entry in the ranges input file." % key)
            num_err += 1
        else:
            num_done += 1
            for minibatch_index, offset, length, label in got:
                info = minibatch_info[minibatch_index]
                # print(read_info) # path_ark_and_offset # /exp/train_combined_no_sil/xvector_feats_train_combined.40.ark:385075627
                # print(info) # [13824, 216, 63] 63 คือ จำนวนที่เจอ? -> เป็น index ที่เพิ่มขึ้นเรื่อยๆ เพื่อไปใส่ใน array
                # print(offset)
                all_data_info[minibatch_index].append((read_info, offset, length, info[1], fea_dim))
                labels[minibatch_index][info[2]] = label
                info[2] += 1
    if logger is not None:
        logger.info('Loading features finished with {0} errors and {1} success from total {2} files.'.
                    format(num_err, num_done, num_err + num_done))
    return all_data_info, labels


def save_data_info_hd5(hd5_file_path, minibatch_info, all_data_info, labels, fea_dim):
    hdf5_file = h5py.File(hd5_file_path, mode='w')
    hdf5_file.create_dataset(name='labels', data=labels)
    for i in range(all_data_info.shape[0]):
        print(i)
        mat = np.zeros((len(all_data_info[i]), minibatch_info[i][1], fea_dim), dtype=np.float32)
        for j, read_info in enumerate(all_data_info[i]):
            m = kaldi_io.read_mat(read_info[0])
            assert m.shape[1] == mat.shape[2] and read_info[2] == mat.shape[1]
            mat[j, :, :] = m[read_info[1]:read_info[1] + read_info[2], :]
        hdf5_file.create_dataset(name=str(i), data=mat)
    hdf5_file.close()


def __add2tar_file(tar_file, array, name):
    my_buffer = io.BytesIO()
    np.save(my_buffer, array)
    size = my_buffer.tell()
    my_buffer.seek(0)
    info = tarfile.TarInfo(name=name)
    info.size = size
    tar_file.addfile(tarinfo=info, fileobj=my_buffer)


def save_data_info_tar(tar_file_path, minibatch_info, all_data_info, fea_dim, logger, downsampled=False): # save only features not labels
    tar_file = tarfile.TarFile(tar_file_path, 'w')
    for i in range(all_data_info.shape[0]):
        logger.info('Writing minibatch: %d' % (i + 1))
        len_1 = minibatch_info[i][1] / 2 if downsampled else minibatch_info[i][1]
        # mat = np.zeros((len(all_data_info[i]), len_1, fea_dim), dtype=np.float32)
        mat = np.zeros((len(all_data_info[i]), len_1, fea_dim), dtype=np.float16) # (640, length, 23)
        for j, read_info in enumerate(all_data_info[i]): # ไล่ทุกตัวในแต่ละ minibatch
            m = kaldi_io.read_mat(read_info[0])
            len_2 = read_info[2] / 2 if downsampled else read_info[2]
            assert m.shape[1] == mat.shape[2] and len_2 == mat.shape[1]
            temp = m[read_info[1]:read_info[1] + read_info[2], :] # [1] -> offset, [2] -> length
            if downsampled:
                # start from frame 1 to work fine for both odd and even array size
                temp = temp[1::2, :]
                assert temp.shape[0] == len_2
            # mat[j, :, :] = temp
            mat[j, :, :] = temp.astype(dtype=np.float16)
        __add2tar_file(tar_file, mat, 'minibatch_' + str(i) + '.npy')
    tar_file.close()


class DataLoader(object):

    def __init__(self, train_data, train_labels, sequential_loading, logger=None, queue_size=5):
        assert train_data.shape[0] == train_labels.shape[0]
        self.train_data = [None] * train_data.shape[0]
        self.train_labels = [0] * train_labels.shape[0]
        for i in range(train_data.shape[0]):
            self.train_data[i] = train_data[i]
            self.train_labels[i] = train_labels[i]
        self.sequential_loading = sequential_loading
        self._total_count = len(self.train_data)
        self.count = self._total_count
        self.logger = logger
        if sequential_loading:
            self.queue = queue.Queue(queue_size)
            self.thread = Thread(target=self.__load_data)
            self.thread.start()

    def __load_data(self):
        while len(self.train_data) > 0:
            data = self.train_data.pop()
            label = self.train_labels.pop()
            mat = np.zeros((len(data), data[0][3], data[0][4]), dtype=np.float32)
            start_time = time.time()
            for i, read_info in enumerate(data):
                m = read_mat(read_info[0])
                assert m.shape[1] == mat.shape[2] and read_info[2] == mat.shape[1]
                mat[i, :, :] = m[read_info[1]:read_info[1] + read_info[2], :]
            if self.logger is not None:
                self.logger.info("Loading one minibatch take %d seconds." % (time.time() - start_time))
            self.queue.put((mat, label))

    def pop(self, timeout=30):
        if self.sequential_loading:
            if self._total_count == 0:
                return None, None
            return self.queue.get(block=True, timeout=timeout)
        else:
            if len(self.train_data) == 0:
                return None, None
            return self.train_data.pop(), self.train_labels.pop()


class TarFileDataLoader(object):

    def __init__(self, tar_file, logger=None, queue_size=5):
        self._train_labels = np.load(tar_file.replace('.tar', '.npy'), allow_pickle=True)
        self._tar = tarfile.open(tar_file, 'r')
        self._names = self._tar.getnames()
        self._total_count = len(self._names)
        self.count = self._total_count
        self._read_index = 0
        assert self._total_count == self._train_labels.shape[0]
        self._logger = logger
        self.queue = queue.Queue(queue_size)
        self._thread = Thread(target=self.__load_data)
        self._thread.daemon = True
        self._thread.start()

    def __load_data(self):
        while self._read_index < len(self._names):
            name = self._names[self._read_index]
            idx = int(name[:-4].split('_')[1])
            label = self._train_labels[idx]
            start_time = time.time()
            
            array_file = io.BytesIO()
            array_file.write(self._tar.extractfile(name).read())
            array_file.seek(0)
            mat = np.load(array_file)


            # mat = np.load(self._tar.extractfile(name))
            if self._logger is not None:
                self._logger.info("Loading one minibatch take %d seconds." % (time.time() - start_time))
            self.queue.put((mat, label))
            self._read_index += 1

    def pop(self, timeout=30):
        if self._total_count == 0:
            return None, None
        return self.queue.get(block=True, timeout=timeout)


def __self_test():
    range_file_path = '/media/punch/DriveE/linux/kaldi/egs/sre16/x-vector-kaldi-tf/exp/xvector_nnet_1a/egs/temp/ranges.1'
    scp_file_path = '/media/punch/DriveE/linux/kaldi/egs/sre16/x-vector-kaldi-tf/exp/xvector_nnet_1a/egs/temp/feats.scp.1'
    tar_file_path = '/media/punch/DriveE/linux/kaldi/egs/sre16/x-vector-kaldi-tf/exp/xvector_nnet_1a/egs/egs.1.tar'
    npy_file_path = '/media/punch/DriveE/linux/kaldi/egs/sre16/x-vector-kaldi-tf/exp/xvector_nnet_1a/egs/egs.1.npy'
    minibatch_count = 640 # จำนวน minibatch ที่เราเซตไว้ตอนแรก
    minibatch_size = 64
    fea_dim = 23
    utt_to_chunks, minibatch_info = process_range_file(range_file_path, minibatch_count, minibatch_size)
    # print(minibatch_info) # มีตามจำนวน minibatch_count
    all_data_info, labels = load_ranges_info(utt_to_chunks, minibatch_info, minibatch_size, scp_file_path, fea_dim)
    # pprint(all_data_info) # list of list of a sub_minibatch (read_info, offset, length, info[1], fea_dim)
    # ('/media/punch/DriveE/linux/kaldi/egs/sre16/v5_sameSpeaker/exp/train_combined_no_sil/xvector_feats_train_combined.39.ark:243202883', 189, 389, 389, 23)
    # print(all_data_info.shape) # (640, )
    # print(labels) 
    # #array([  24,   45,   49,   71,   93,  105,  126,  129,  143,  145,  158,
    #     188,  247,  265,  274,  303,  306,  317,  321,  325,  342,  401,
    #     435,  485,  501,  517,  560,  573,  578,  595,  598,  617,  623,
    #     627,  630,  672,  675,  679,  699,  719,  725,  754,  771,  775,
    #     801,  806,  847,  870,  925,  926,  933,  935,  968,  976,  989,
    #    1001, 1018, 1019, 1065, 1080, 1111, 1112, 1116, 1131], dtype=int32)]  xxxxxxxxxxx 64 -> บอก label แต่ละตัวใน minibatch
    import logging
    logger=logging.getLogger() 
    save_data_info_tar(tar_file_path, minibatch_info, all_data_info, fea_dim, logger)
    np.save(npy_file_path, labels)

def __self_test2():
    tar_path = '/media/punch/DriveE/linux/kaldi/egs/sre16/x-vector-kaldi-tf/exp/xvector_nnet_1a/egs/egs.2.tar'

    data_loader = TarFileDataLoader(tar_path)
    print(data_loader.count)
    minibatch_count = data_loader.count
    for minibatch_idx in range(minibatch_count):
        try:
            batch_data, labels = data_loader.pop()
        except queue.Empty:
            # logger.warning('Timeout reach when reading minibatch index %d' % minibatch_idx)
            continue
        if batch_data is None:
            print('batch_data is None for minibatch index %d' % minibatch_idx)
            continue
        print(batch_data.shape, labels.shape) # (64, length, 23) (64, )
if __name__ == '__main__':
    __self_test()
