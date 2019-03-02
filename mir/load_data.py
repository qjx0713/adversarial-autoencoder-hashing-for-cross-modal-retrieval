import h5py
import numpy as np

def loading_data(path):
	print ('******************************************************')
	print ('dataset:{0}'.format(path))
	print ('******************************************************')

	file = h5py.File(path)
	images = file['IAll'][:].transpose(0,3,2,1)
	labels = file['LAll'][:].transpose(1,0)
	tags = file['YAll'][:].transpose(1,0)
	file.close()

	return images, tags, labels


def split_data(images, tags, labels, QUERY_SIZE, TRAINING_SIZE, DATABASE_SIZE):

	X = {}
	index_all = np.random.permutation(QUERY_SIZE+DATABASE_SIZE)
	ind_Q = index_all[0:QUERY_SIZE]
	ind_T = index_all[QUERY_SIZE:TRAINING_SIZE + QUERY_SIZE]
	ind_R = index_all[QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE]

	X['query'] = images[ind_Q, :, :, :]
	X['train'] = images[ind_T, :, :, :]
	X['retrieval'] = images[ind_R, :, :, :]

	Y = {}
	Y['query'] = tags[ind_Q, :]
	Y['train'] = tags[ind_T, :]
	Y['retrieval'] = tags[ind_R, :]

	L = {}
	L['query'] = labels[ind_Q, :]
	L['train'] = labels[ind_T, :]
	L['retrieval'] = labels[ind_R, :]
	return X, Y, L

def loading_NUS(path):
	print ('******************************************************')
	print ('dataset:NUS-WIDE')
	print ('******************************************************')
	image_path = '/'.join([path,'IAll','IAll','nus-wide-tc10-iall.mat'])
	image_file = h5py.File(image_path)
	images = image_file['IAll'][:].transpose(0,3,2,1)
	image_file.close()

	path = '/'.join([path,'nus-wide-tc10.mat'])
	file = h5py.File(path)
	labels = file['LAll'][:].transpose(1,0)
	tags = file['YAll'][:].transpose(1, 0)
	index_Test = file['param']['indexTest'][:]
	index_Test = np.squeeze(index_Test).astype(int)
	index_Train = file['param']['indexTrain'][:]
	index_Train = np.squeeze(index_Train).astype(int)
	index_Database = file['param']['indexDatabase'][:]
	index_Database = np.squeeze(index_Database).astype(int)
	file.close()

	return images, tags, labels, index_Test, index_Train, index_Database


def split_NUS(images, tags, labels, indexTest, indexTrain, indexDatabase):

	X = {}

	X['query'] = images[indexTest, :, :, :]
	X['train'] = images[indexTrain, :, :, :]
	X['retrieval'] = images[indexDatabase, :, :, :]

	Y = {}
	Y['query'] = tags[indexTest, :]
	Y['train'] = tags[indexTrain, :]
	Y['retrieval'] = tags[indexDatabase, :]

	L = {}
	L['query'] = labels[indexTest, :]
	L['train'] = labels[indexTrain, :]
	L['retrieval'] = labels[indexDatabase, :]
	return X, Y, L
