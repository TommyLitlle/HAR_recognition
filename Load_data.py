import pickle as cp
import numpy as np
import collections
from sliding_window import sliding_window


# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113
NB_SENSOR_CHANNELS_WITH_FILTERING = 149

# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 18

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 24

# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 8

# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = int(SLIDING_WINDOW_LENGTH/2)
SLIDING_WINDOW_STEP_SHORT = SLIDING_WINDOW_STEP

def load_file(file_path):
    
    f = open(file_path, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(file_path))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
 
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test


def one_hot(y):
    """convert label from dense to one hot
      argument:
        label: ndarray dense label ,shape: [sample_num,1]
      return:
        one_hot_label: ndarray  one hot, shape: [sample_num,n_class]
    """
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y = y.reshape(len(y))
    n_values = np.max(y) + 1
    return np.eye(n_values)[np.array(y, dtype=np.int32)]  # Returns FLOATS



def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
  
    data_x, data_y = data_x.astype(np.float32), one_hot(data_y.reshape(len(data_y)).astype(np.uint8))
    #print(data_y)
    return data_x, data_y

def return_data():
     
    X_train, y_train, X_test, y_test = load_file('E:/CorrelationModel/HAR_recognition/18_HAR\OPPORTUNITYDATASET.txt')
    
    assert (NB_SENSOR_CHANNELS_WITH_FILTERING == X_train.shape[1] or NB_SENSOR_CHANNELS == X_train.shape[1])
    X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP_SHORT)
    X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    
    print(X_train.shape)
    
    return X_train, y_train, X_test, y_test

class DataSet(object):

    def __init__(self,
                 data,
                 labels,
                 one_hot=False,
                 dtype=np.float32,
                 reshape=False):
            
           
        self._num_examples=data.shape[0]
            #print(self._num_examples)
        if reshape:
            #assert data.shape[2] ==1
            data = data.reshape (data.shape[0],
                                      data.shape[1])
        if dtype == np.float32:
            data = data.astype(np.float32)
        self._data = data
        self._labels= labels
        self._epochs_completed =0
        self._index_in_epoch =0
        
    @property
    def data(self):
        return self._data
            
    @property
    def labels(self):
        return self._labels
            
    @property
    def num_exanples(self):
        return self._num_examples
            
    @property
    def epochs_completed(self):
        return self._epochs_completed
        
        
    def next_batch(self, batch_size, shuffle =True):
        """ Return the bext 'batch_size' examples from this data set."""
        start=self._index_in_epoch
                
        #Shuffle for the first epoch
                
        if self._epochs_completed ==0 and start ==0 and shuffle:
            perm0=np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._data=self.data[perm0]
            self._labels=self.labels[perm0]
                    
        #go to the next epoch
        if start+batch_size > self._num_examples:
            #finished epoch
            self._epochs_completed +=1
            #get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start : self._num_examples]
            labels_rest_part =self._labels[start: self._num_examples]
            #Shuffle the data      
            if shuffle:
                perm=np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._data=self.data[perm]
                self._labels=self.labels[perm]
                        
            # Start next epoch
            start = 0
            self._index_in_epoch =batch_size-rest_num_examples
            end=self._index_in_epoch
            data_new_part = self._data[start: end]
            labels_new_part = self._labels[start: end]
            return np.concatenate((data_rest_part, data_new_part), axis=0),np.concatenate((labels_rest_part, labels_new_part), axis=0)                    
        else:
            self._index_in_epoch += batch_size
            end=self._index_in_epoch
            return self._data[start: end], self._labels[start: end]
  
Datasets = collections.namedtuple('Datasets', ['train', 'test'])  
        
def read_data_sets(dtype = np.float32,
                    reshape =False):
    
    train_data, train_labels, test_data, test_labels = return_data()
    
    train =DataSet(train_data, train_labels, dtype =dtype, reshape =reshape)
    
    test = DataSet(test_data, test_labels, dtype=dtype, reshape =reshape)
    
    return Datasets(train=train, test=test)
    
if __name__ == "__main__":
    return_data()