import json
import numpy as np

class DataSource:

    def __init__(self, seq_length=20, features_length=512, features_dir='features_20', model='base_0'):
        self.seq_length = seq_length
        self.features_dir = features_dir
        self.model = model
        self.features_length = features_length

        # Load data (features, filenames and labels)
        self.features_npy = np.load(f"{features_dir}/{model}/features-{features_length}.npy")
        self.filenames_npy = np.load(f"{features_dir}/{model}/filenames.npy")
        self.labels_npy = np.load(f"{features_dir}/{model}/labels.npy")

    def __transform_label(self, file_idx):
        label = self.labels_npy[file_idx]
        if label == 0:
            label = [1,0,0]
        if label == 1:
            label = [0,1,0]
        if label == 2:
            label = [0,0,1]
        return label

    def __structure_data(self, fold_number_str='0', data_type='train', data=(None, None)):
        filenames, predictions = data

        X_data = []
        Y_data = []
        for idx, filename in enumerate(filenames):
            features_array = []
            labels_array = []
            for i in range(self.seq_length):
                path_value = f"{predictions[idx]}/{filename}-{i+1}.png"
                try:
                    file_idx = np.where(self.filenames_npy == path_value)[0][0]
                except:
                    import pdb; pdb.set_trace()
                features = list(self.features_npy[file_idx])

                features_array.append(features)

                if i == 0:
                    labels_array.append(self.__transform_label(file_idx))
            X_data.append(features_array)
            Y_data.append(labels_array[0])

        # returns (X, Y)
        return np.array(X_data), np.array(Y_data)

    def get_data_from_fold(self, fold_number_str='0', data_type='train'):
        with open('cross_val.json', 'r') as json_file:
            data = json.load(json_file)

        # returns the filenames and labels 
        return (data[fold_number_str][data_type][0], data[fold_number_str][data_type][1])

    def get_train_from_fold(self, fold_number_str='0'):
        # returns (X, Y)
        return self.__structure_data(fold_number_str, 'train', self.get_data_from_fold(fold_number_str, 'train'))

    def get_test_from_fold(self, fold_number_str='0'):
        # returns (X, Y)
        return self.__structure_data(fold_number_str, 'test', self.get_data_from_fold(fold_number_str, 'test'))

    def count_videos(self, fold_number_str='0', data_type='train'):
        labels = np.array([self.get_data_from_fold(fold_number_str=fold_number_str, data_type=data_type)[1]])
        lbls, cnts = np.unique(labels, return_counts=True)

        # returns [[labels],[count]]
        return np.asarray((lbls, cnts))