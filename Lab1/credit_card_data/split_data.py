import torch
import pandas
import numpy as np

dataset=pandas.read_csv("creditcard_csv.csv")
#dataset: Pandas DF object, len: num of rows

def split_csv_data(f, training_share=0.75, val_share = 0.15, test_share=0.1):
    random_order = np.random.permutation(len(dataset))
    training_set = random_order[:int(training_share*len(dataset))]
    val_set = random_order[int(training_share*len(dataset)):int((training_share+val_share)*len(dataset))]
    test_set = random_order[int((training_share+val_share)*len(dataset)):]
    print(len(dataset), len(training_set), len(val_set), len(test_set), len(training_set) + len(val_set) + len(test_set) == len(dataset))
    training_data = dataset.iloc[training_set].to_csv("train_data.csv")
    validation_data = dataset.iloc[val_set].to_csv("val_data.csv")
    test_data = dataset.iloc[test_set].to_csv("test_data.csv")
        
    #return training_data, validation_data

split_csv_data(dataset)

