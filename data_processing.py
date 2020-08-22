import _pickle as cPickle
import numpy as np
from imgaug import augmenters as aug
import random
import os

class DataSetProcessing(object):

    def unpickle(file):
        with open(file,'rb') as file_reader:
            _dict=cPickle.load(file_reader,encoding='bytes')
        return _dict

    def one_hot_encode(vec,vals=100):
        n=len(vec)
        Y=np.zeros((n, vals))
        Y[range(n),vec]=1
        return Y

    def shuffleImages(all_train,all_train_labels):
        
        all_train_shuffled,all_train_labels_shuffled=[],[]
        combined=list(zip(all_train, all_train_labels))
        random.shuffle(combined)
        all_train_shuffled[:],all_train_labels_shuffled[:]=zip(*combined)

        return all_train_shuffled,all_train_labels_shuffled

    def performAugmentation(X_train_orig):
        _seq=aug.Sequential([
                aug.Fliplr(0.5),
                aug.CropAndPad(px=(-2, 2),sample_independently=True,pad_mode=["constant", "edge"]),
                aug.Affine(shear=(-10, 10),mode = ['symmetric','wrap']),#48
                aug.Add((-5, 5)),
                aug.Multiply((0.8, 1.2)),
            ],random_order=True)
        
        augmented_data= _seq.augment_images(X_train_orig)
        return augmented_data
    
    def loadData():
        
        filepath='cifar-100-python'

        train=DataSetProcessing.unpickle(os.path.join(filepath, "train"))
        test=DataSetProcessing.unpickle(os.path.join(filepath, "train"))
        meta=DataSetProcessing.unpickle(os.path.join(filepath, "meta"))

        X_train=np.asarray(train[b'data'])
        X_train_orig=X_train.reshape(50000,3,32,32).transpose(0,2,3,1).astype("uint8")


        augmented_X1= DataSetProcessing.performAugmentation(X_train_orig)

        all_train=[]
        all_train.extend(X_train_orig)
        all_train.extend(augmented_X1/255)

        all_train_labels=[]
        all_train_labels.extend(train[b'fine_labels'])
        all_train_labels.extend(train[b'fine_labels'])

        all_train_shuffled,all_train_labels_shuffled= DataSetProcessing.shuffleImages(all_train,all_train_labels)
        
        X_train_shuffled=np.asarray(all_train_shuffled)
        Y_train_labels_shuffled=DataSetProcessing.one_hot_encode(all_train_labels_shuffled,100)

        X_test=np.asarray(test[b'data'])
        X_test_shuffled=X_test.reshape(len(X_test),3,32,32).transpose(0,2,3,1)/255
        Y_test_shuffled=DataSetProcessing.one_hot_encode(test[b'fine_labels'],100)

        return X_train_shuffled,Y_train_labels_shuffled,X_test_shuffled,Y_test_shuffled




        

