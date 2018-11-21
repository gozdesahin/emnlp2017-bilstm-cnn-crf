# This script trains the BiLSTM-CRF architecture for part-of-speech tagging using
# the universal dependency dataset (http://universaldependencies.org/).
# The code use the embeddings by Komninos et al. (https://www.cs.york.ac.uk/nlp/extvec/)
from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle
import argparse
from util.postprocessing import remove_except_last_model


def main():
	# :: Change into the working dir of the script ::
	abspath = os.path.abspath(__file__)
	dname = os.path.dirname(abspath)
	os.chdir(dname)

	# :: Logging level ::
	loggingLevel = logging.INFO
	logger = logging.getLogger()
	logger.setLevel(loggingLevel)

	ch = logging.StreamHandler(sys.stdout)
	ch.setLevel(loggingLevel)
	formatter = logging.Formatter('%(message)s')
	ch.setFormatter(formatter)
	logger.addHandler(ch)

    parser = argparse.ArgumentParser()
    parser.add_argument('-datasetName', type=str, default='conll2000_data/clean', help='Folder path to train,dev and test files are')
    parser.add_argument('-cuda_device', type=int, default=0, help='cpu:-1, others: gpu')
    parser.add_argument('-embeddings', type=str, default='pretrained/velmo_options.json', help='ELMO options file path')
    parser.add_argument('-model_save', type=str, default='models', help='path to save the model file')
    args = parser.parse_args()
    train_pos(args)


def train_pos(args):
	######################################################
	#
	# Data preprocessing
	#
	######################################################
	datasets = {
	    args.datasetName:                            #Name of the dataset
	        {'columns': {0:'tokens', 1:'POS', 2:'chunk_BIO'},   #CoNLL format for the input data. Column 1 contains tokens, column 3 contains POS information
	         'label': 'POS',                     #Which column we like to predict
	         'evaluate': True,                   #Should we evaluate on this task? Set true always for single task setups
	         'commentSymbol': None}              #Lines in the input data starting with this string will be skipped. Can be used to skip comments
	}


	# :: Path on your computer to the word embeddings. Embeddings by Komninos et al. will be downloaded automatically ::
	embeddingsPath = args.embeddings
	#'komninos_english_embeddings.gz'

	# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
	pickleFile = perpareDataset(embeddingsPath, datasets)


	######################################################
	#
	# The training of the network starts here
	#
	######################################################


	#Load the embeddings and the dataset
	embeddings, mappings, data = loadDatasetPickle(pickleFile)

	# Some network hyperparameters
	params = {'classifier': ['CRF'], 'LSTM-Size': [100], 'dropout': (0.25, 0.25)}

	model = BiLSTM(params)
	model.setMappings(mappings, embeddings)
	model.setDataset(datasets, data)

	#model.modelSavePath = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5" #Path to store models
	model.modelSavePath = args.model_save+"/[ModelName]_[Epoch].h5"
	model.fit(epochs=25)

    save_dir, model_init = os.path.split(fpath)
    print(save_dir)
    print(model_init)
    # remove trained files except from the last file
    remove_except_last_model(save_dir, model_init)

if __name__ == '__main__':
    main()