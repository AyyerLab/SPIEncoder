import sys
import os
import configparser
import time

import numpy as np
import cupy as cp
import h5py

from cupadman import CDetector, CDataset, Quaternion

class LossCalculator():
    def __init__(self, config_fname, num_streams=4):
        stime = time.time()
        self.num_streams = num_streams

        # Parse config file
        config = configparser.ConfigParser()
        config.read(config_fname)
        config_dir = os.path.dirname(config_fname)

        detector_fname = os.path.join(config_dir, config.get('encoder', 'in_detector_file'))
        photons_fname = os.path.join(config_dir, config.get('encoder', 'in_photons_file', fallback=''))
        self.output_folder = os.path.join(config_dir, config.get('encoder', 'output_folder', fallback='data/'))
        num_div = config.getint('encoder', 'num_div')
        self.ltype = config.get('encoder', 'loss_type', fallback='').lower()
        if self.ltype == '':
            print('Using default Euclidean distance loss')
            self.ltype = 'euclidean'
        if self.ltype not in ['euclidean', 'likelihood']:
            raise ValueError('loss_type must be one of euclidean or likelihood')
        elif self.ltype == 'euclidean':
            self.calc_loss = self._calc_loss_euclidean
        elif self.lytp == 'likelihood':
            raise NotImplementedError('Likelihood loss calculation not implemented yet')

        # Setup reconstruction
        # -- Generate detector, dataset, quaternions
        # -- Note the following three structs have data in CPU memory
        self.det = CDetector(detector_fname)
        if self.ltype == 'likelihood':
            self.dset = CDataset(photons_fname, self.det)
        else:
            self.dset = None
        self.quat = Quaternion(num_div)
        self.size = int(2*np.ceil(np.linalg.norm(self.det.qvals, axis=1).max()) + 3)
        self._move_to_gpu()
        
        # -- Get CUDA kernels
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(script_dir+'/kernels.cu', 'r') as f:
            kernels = cp.RawModule(code=f.read())
        self.k_slice_gen = kernels.get_function('slice_gen3d')
        self.k_sqdiff = cp.ReductionKernel('T x, T y', 'T z', 'pow(x-y, 2.)', 'a+b', 'z = a', '0', 'sqdiff')

        # -- Allocate other arrays
        self.bsize_pixel = int(np.ceil(self.det.num_pix/32.))
        self.stream_list = [cp.cuda.Stream() for _ in range(self.num_streams)]
        self.views = cp.zeros((self.num_streams, self.det.num_pix))

        print('Completed loss calculator setup: %f s' % (time.time() - stime))

    def _calc_loss_euclidean(self, img, model):
        losses = cp.empty(self.quat.num_rot)
        for i, r in enumerate(range(self.quat.num_rot)):
            snum = i % self.num_streams
            
            self.stream_list[snum].use()
            self.k_slice_gen((self.bsize_pixel,), (32,),
                    (model, self.quats[r], self.pixvals,
                     self.dmask, 0., self.det.num_pix,
                     self.size, self.views[snum]))
            losses[i] = self.k_sqdiff(img, self.views[snum])
            #losses[i] = cp.linalg.norm(self.views[snum] - img)
        return losses.max(), self.quats[losses.argmax(), :4]

    def _move_to_gpu(self):
        '''Move detector, dataset and quaternions to GPU'''
        self.dmask = cp.array(self.det.raw_mask)
        self.pixvals = cp.array(np.concatenate((self.det.qvals, self.det.corr[:,np.newaxis]), axis=1).ravel())

        self.quats = cp.array(self.quat.quats)

        if self.dset is not None:
            init_mem = cp.get_default_memory_pool().used_bytes()
            self.ones = cp.array(self.dset.ones)
            self.multi = cp.array(self.dset.multi)
            self.ones_accum = cp.array(self.dset.ones_accum)
            self.multi_accum = cp.array(self.dset.multi_accum)
            self.place_ones = cp.array(self.dset.place_ones)
            self.place_multi = cp.array(self.dset.place_multi)
            self.count_multi = cp.array(self.dset.count_multi)
            self.dset_mem = cp.get_default_memory_pool().used_bytes() - init_mem

