from ast import arg
from cgi import test
from distutils.command.config import config
# from importlib.resources import path
import os
from pickletools import optimize
import sys
from pickle import TRUE
from platform import node
from pyexpat import model
from statistics import mode
import yaml

import argparse
from symbol import argument
import networkx as nx
# import node2vec
import pandas as pd
# make the loop shows progress
# from tqdm import tqdm

import numpy as np
# pytorch related package
import torch
import torch.nn as nn # contain neural network packages
import torch.nn.functional as F

# from torchinfo import summary
# boss is here !!
from torch_geometric.data import Data
#import training model class
from models import GCNN, GAT, TAGE, SAGE
import time

from sklearn.model_selection import KFold


DATASET_DIR = '/work/GNN/original_data'
_seperator = "\t"

#TODO: Check about lexical scoping in python; and you can sepcify cuda id as well; learn about that as well
# torch.cuda.is_available() is available flag remain throughout the program
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

modelsOption = {
    'GCNL' : GCNN,
    "GAT" : GAT,
    "TAGE": TAGE,
    'SAGE' : SAGE,
}

def retriveData(dataDir):
    global myDirName
    myDirName = os.path.join(DATASET_DIR, dataDir, "train.data") 
    # the schema is stored in the config file inside the directory
    # We don't need to explicitly call the close() method. It is done internally.
    with open(os.path.join(myDirName, "config.yml"), "r") as stream:
        try:
            conf_data = yaml.safe_load(stream)
            # print(conf_data)
            print("---------------------------------------configuration file opened successfully ------------------------------------------------")
        except yaml.YAMLError as exc:
            print("there is an error")
            print(exc)
    
    # read tsv file and get pandas.DataFrame
    try: 
        edgeTSV = pd.read_csv(os.path.join(myDirName, 'edge.tsv'), sep=_seperator, header=0)
        featuresTSV = pd.read_csv(os.path.join(myDirName, 'feature.tsv'), sep=_seperator, header=0)
    except IOError as err:
        print(err)
    
    # now iterate over all the dataframe
    edgeIterator = edgeTSV.iterrows()
    featureIterator = featuresTSV.iterrows()

    edges = []
    features = []

    for index, row in edgeIterator: #it does not have index of the row
        edges.append(row.to_list()) # since the data that we get from each row is source, destination and weight index so just wanted to get values not entries
    

    for index, row in featureIterator: #it has row index but it doesnot matter because array index will be enough
        fd = row.to_list()[1:]
        if len(fd) < 100:
            t = 100 - len(fd)
            fd = np.pad(fd, padWidth = (0, t), mode="constant")
        features.append(fd) # slice from second element
    
    assert len(features) == featuresTSV.shape[0] #check if we appended all?

    # shit, shape() is in tensor
    featureLength = len(features[0])
    edgeLength = len(edges)
    conf_data["edge_count"] = edgeLength

    # we can not feed array into pytorch so transform these to Tensor !! 
    # we could also do that with transform params in data loader of torch_geometry data as toTensor() where toTensor can be class with __init__ doing transformation
    # https://www.youtube.com/watch?v=X_QOZEko5uE&ab_channel=PythonEngineer

    # transpose is smart way of getting [[sources], [destination], [weight]]
    edges = torch.Tensor(edges).long().t()
    edge_index = edges[[0,1]]
    edges_attr = edges[[2]].t().float() #right now our data has weight of 1 but just for decimanl weight as well so casted to float


    # check before writing into config file
    
    conf_data["feature_count"] = featureLength

    features = torch.Tensor(np.array(features, dtype=float))
    # number of feature is number of node
    conf_data["n_node"] = features.shape[0]

    # load train label and test lable
    try: 
        # test_labels = pd.read_csv(os.path.join(myDirName, 'test.csv'), sep=_seperator, header=0)
        train_labels = pd.read_csv(os.path.join(myDirName, 'train.csv'), sep=_seperator, header=0)
    except IOError as err:
        print(err)
    
    labels = torch.zeros(features.shape[0]).long()
    ratio_VD = 0.15 #TODO: check if there is env file possible in python so that we can change this kind of value from a file
    # create empty tensor to store the value in prediction
    # we can store lable of vertex in an array because same data is divided into two part
    train_mask = np.zeros(features.shape[0],  dtype=bool)
    test_mask = np.zeros(features.shape[0],  dtype=bool)
    validation_mask = np.random.rand(features.shape[0]) < ratio_VD
    _class = [];
    # for index, row in test_labels.iterrows():
    #     node1, label1 = row.tolist() 
    #     labels[node1] = label1
    #     test_mask[node1] = 1
    #     _class.append(label1) if label1 not in _class else _class

    # Main Logic here is: https://stackoverflow.com/questions/2451386/what-does-the-caret-operator-do
    assert False == True^True
    # we need to make false in the validation mask which is test data so False = True * False
    validation_mask *= train_mask #now this is our Final validation mask
    # since we got validation mask from training data we need to update that mask as well 

    for index, row in train_labels.iterrows():
        node2, label2 = row.tolist()
        labels[node2] = label2 
        train_mask[node2] = 1
        _class.append(label2) if label2 not in _class else _class
    
    conf_data["class_count"] = len(_class)
    print("number of labels - ", len(_class))


    return conf_data, edge_index, edges_attr, features, labels, train_mask, test_mask

def retriveTestData(trainIndices, testIndices, feature, label, conf_data):
    print("traing feature array is")
    print(trainIndices)
    print("--------------------------------------------------------------------")
    feature = feature.numpy()
    label = label.numpy()
    train_mask = np.zeros(conf_data["n_node"],  dtype=bool)
    validation_mask = np.zeros(conf_data["n_node"], dtype=bool)
    
    trainFeatures = []
    testFeatures = []
    trainLabels = []
    testLabels = []

    trainingEdges = []
    testEdges = [] 
    myDirName = os.path.join(DATASET_DIR, 'a', "train.data") 
    edgeTSV = pd.read_csv(os.path.join(myDirName, 'edge.tsv'), sep=_seperator, header=0)
    edgeIterator = edgeTSV.iterrows()
    for index, row in edgeIterator:
        eachData = row.to_list()
        srcIndex = eachData[0]
        destIndex = eachData[1]
        if((srcIndex in trainIndices) and (destIndex in trainIndices)):
            print(srcIndex, destIndex)
            trainingEdges.append(row.to_list())
        if((srcIndex in testIndices) and (destIndex in testIndices)):
            testEdges.append(row.tolist())

    print(len(trainingEdges), len(testEdges))
    traingingEdgesTrain = torch.Tensor(trainingEdges).long().t()
    edge_indexTrain = traingingEdgesTrain[[0,1]]
    edges_attrTrain = traingingEdgesTrain[[2]].t().float()
    testEdgesTest = torch.Tensor(testEdges).long().t()
    edge_indexTest = testEdgesTest[[0,1]]
    edges_attrTest = testEdgesTest[[2]].t().float()

    for index in trainIndices:
        trainFeatures.append(feature[index])
        trainLabels.append(label[index])
        train_mask[index] = 1
           
    for index2 in testIndices:
        testFeatures.append(feature[index2])
        validation_mask[index2] = 1
        testLabels.append(label[index2])
    
    trainFeaturesT = torch.Tensor(np.array(trainFeatures, dtype=float))
    testFeaturesT = torch.Tensor(np.array(testFeatures, dtype=float))
    trainLabelsT = torch.Tensor(np.array(trainLabels, dtype=float))
    testLabelsT = torch.Tensor(np.array(testLabels, dtype=float))


    # train_maskT = torch.Tensor(np.array(train_mask, dtype=bool))
    # validation_maskT = torch.Tensor(np.array(validation_mask, dtype=bool))


    return train_mask, validation_mask, trainFeaturesT, trainLabelsT, testFeaturesT, testLabelsT, edge_indexTrain, edges_attrTrain, edge_indexTest, edges_attrTest

def accuracy_calculation(logits, labels):
    indices = torch.argmax(logits, dim=1)
    return (indices== labels).float().mean().item()

def get_mse(actual, predicted):
    loss = ((actual - predicted) ** 2).mean(axis=0)
    return loss


def get_accuracy(actual, predicted, threshold):
    correct = 0
    predicted_classes = []
    for prediction in predicted :
      if prediction >= threshold :
        predicted_classes.append(1)
      else :
        predicted_classes.append(0)
    for i in range(len(actual)):
      if actual[i] == predicted_classes[i]:
        correct += 1
    return correct / float(len(actual)) * 100.0


def main(args):
    # print(args)
    model_type = args.M
    print(model_type)
    # if this is running for first time in the dataset run the code otherwise use stored processed dataset from the multiple dataset files
    loadedFile = os.path.join(DATASET_DIR, args.dataset, "load.pt")
    if not os.path.exists(loadedFile):
        print("data loading from source")
        conf_data, edge_index, edges_attr, features, labels, train_mask, validation_mask = retriveData(args.dataset)
        torch.save([conf_data, edge_index, edges_attr, features, labels, train_mask, validation_mask], loadedFile)
    else:
        print("loaded from the loaded file")
        conf_data, edge_index, edges_attr, features, labels, train_mask, validation_mask = torch.load(loadedFile)

    # https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html
    # create homogeneous graph: Awesome tool
    #TODO: Watch more videos in pytorch-geometric
    # in documentation: https://pytorch-geometric.readthedocs.io/en/1.3.2/modules/data.html shape [num_nodes, num_node_features] mean shape should be this value
    # we already have that shape; don't get confused with [] it just mean tuple of shape tensor.shape()

    
    
    model = modelsOption["SAGE"](conf_data["class_count"], conf_data["feature_count"], hidden_layers=64, drop_out_rate = 0.6)
        # print("your model is created successfully")
        # print("sorry ", sys.exc_info()[0], "happened")

    # https://www.youtube.com/watch?v=7q7E91pHoW4&ab_channel=PythonEngineer
    # loss_fn = nn.CrossEntropyLoss() 
    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
    # https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9
    numb_epoch = 1000
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    model.resetParameter()
    model.train()
    eval_maxAccuracy = 0
    train_maxAccuracy = 0.0
        # print(f"train_loss : {loss:0.5f}" )
    
    # As the number of epochs increases, more number of times the weight are changed in the neural network and the curve goes from underfitting to optimal to overfitting curve.
    start = time.time()
    max_eval_epoch = -1.0
    fraction = 1/k_folds
    seg = int(conf_data["n_node"]*fraction)
    print("hello")
    for k_step in range(k_folds):
        print("------------------------------------")
        print(f'FOLD {k_step}')
        print("------------------------------------")
        
        traingingStart = 0
        trainingEnd = k_step*seg
        validatiionStart = trainingEnd
        validationEnd = k_step*seg + seg
        trainingSecStart = validationEnd
        trainingSecEnd = conf_data["n_node"]-1

        train_left_indices = list(range(traingingStart,trainingEnd))
        train_right_indices = list(range(trainingSecStart,trainingSecEnd))
                
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(validatiionStart,validationEnd))

        train_mask, testMask, trainFeature, trainLable, testFeature, testLabel, edge_indexTrain, edges_attrTrain, edge_indexTest, edges_attrTest = retriveTestData(train_indices, val_indices, features, labels, conf_data)



        trainData = Data(x=trainFeature, edge_index=edge_index, edge_attr=edges_attr, y=trainLable, train_mask = train_mask, validation_mask = validation_mask, test_mask = testMask  )
        testData = Data(x=testFeature, edge_index=edge_index, edge_attr=edges_attr, y=testLabel, train_mask = train_mask, validation_mask = validation_mask, test_mask = testMask  )

        # print(" i am main data", data)
        # print(data.validation_mask)
        # learned that after y it is same as rest parameter in JS and Data is nothing just a dictionary
        # https://pytorch.org/docs/stable/generated/torch.Tensor.to.html
        # creates an another array with data as an element and "device:cpu" as second element
        trainData = trainData.to(device)
        testData = testData.to(device)
        totalSize = trainData.size(dim=0)
        for epoch in range(numb_epoch):
            optimizer.zero_grad()
            print("training")        
            outgraph = model(trainData)
            # print("training accuracy")
            loss = F.nll_loss(outgraph[trainData.train_mask], trainData.y[trainData.train_mask])
            train_acc = accuracy_calculation(outgraph[trainData.train_mask], trainData.y[trainData.train_mask])
            if train_acc > train_maxAccuracy:
                    train_maxAccuracy = train_acc
            # print(train_acc)
            loss.backward()
            optimizer.step() 

            # set flag to disable grad calculation because you don't want to change parameter
            #  and weight on testing and validation
        
            with torch.no_grad():
                model.eval()
                print("testing")
                testOut = model(testData)
                evaluation_accuracy = accuracy_calculation(testOut[testData.test_mask], testData.y[testData.test_mask])
                if evaluation_accuracy > eval_maxAccuracy:
                    eval_maxAccuracy = evaluation_accuracy
                    max_eval_epoch = epoch
                elif epoch - max_eval_epoch >= 100: 
                    break
                model.train()
        # summary(model=model, input_size=(1, conf_data["feature_count"], 64, 64, conf_data["class_count"]))  
    end = time.time()
    print("training accuracy: ", train_maxAccuracy)
    print("testing accuracy: ", eval_maxAccuracy)
    elapsed_time = end - start
    print("Training time is:", elapsed_time, "seconds")

if __name__ == "__main__":
    print("------------------------------------------------------")
    print ("main.py execuated as main")
    # parse the args
    parser = argparse.ArgumentParser(description=' retrive process option argument')
    parser.add_argument("--dataset", choices=["a", "b", "c", "d", "e", "f"], required=True)
    parser.add_argument("--M", choices=["GNNL", "GAT"])
    parser.add_argument("--epoch", type=int, default=100) #defualt number of pass is 100
    _args = parser.parse_args()
    print(main(_args))

else:
     print ("i am executed from imported")



