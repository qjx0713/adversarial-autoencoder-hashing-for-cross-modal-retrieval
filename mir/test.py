#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：qjx time:2018/10/29 0029

import h5py
import numpy as np

path = '/'.join(['F:/qjx/NUS-WIDE-TC10', 'nus-wide-tc10.mat'])
file = h5py.File(path)
labels = file['LAll'][:].transpose(1, 0)
tags = file['YAll'][:].transpose(1, 0)
index_Test = file['param']['indexTest'][:]
index_Test = np.squeeze(index_Test).astype(int)


file.close()
print(index_Test[0])