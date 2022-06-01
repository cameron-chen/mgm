import numpy as np
from torch.utils.data import Dataset


def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
        data: original data

    Returns:
        norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data

def ZScaler(data):
    """Z score normalizer: with 3 standard deviation
    
    Args:
        data: original data

    Returns:
        norm_data: normalized data
    """
    m, s = data.mean(0), data.std(0)
    data = (data - m) / (3 * s)
    data = np.tanh(data)
    return data

def AROne_data_generation(no, D, seq_len, phi, s, burn=10):
    """Generate Autoregressive data of order 1

    Args:
        no: number of samples
        D: dimension of x
        seq_len: sequence length
        phi: parameters for AR model
        s: parameter that controls the magnitude of covariance matrix

    Returns:
        data: generated data
    """
    Sig = np.eye(D) * (1 - s) + s
    chol = np.linalg.cholesky(Sig)

    x0 = np.random.randn(no, D)
    x = np.zeros((seq_len+burn, no, D))
    x[0,:,:] = x0
    for i in range(1, seq_len+burn):
        x[i, ...] = phi*x[i-1] + np.random.randn(no, D) @chol.T
    
    x = x[-seq_len:, :, :]
    x = np.swapaxes(x, 0, 1)

    return x


def real_data_loading(
    dname, seq_len, stride=1, 
    trunc_head_perc=0, nor_method = 'min_max'):
    '''Load and preprocess real-world datasets.
    
    Args:
        dname: Elec
        seq_len: sequence length
        nor_method: normalization method, ["min_max", "z_score"]
    
    Returns:
        data: preprocessed data
    '''
    assert dname in ['Elec','Elec_low']
    assert nor_method in ["min_max", "z_score"]

    if dname == 'Elec' or dname == 'Elec_low':
        ori_data = np.loadtxt('./data/electric.csv', delimiter=',', skiprows=1)

    # normalize the data
    if nor_method == 'min_max':
        ori_data = MinMaxScaler(ori_data)
    elif nor_method == 'z_score':
        ori_data = ZScaler(ori_data)

    # truncate data
    if trunc_head_perc:
        assert trunc_head_perc<=1 and trunc_head_perc>0
        trunc_point= int(len(ori_data)*trunc_head_perc)
        ori_data = ori_data[:trunc_point]

    # preprocess the dataset
    temp_data = []
    ## cut data by sequence length
    for i in range(0, len(ori_data)-seq_len, stride):
        _x = ori_data[i:i+seq_len]
        temp_data.append(_x)

    # shuffle the data
    data = np.array(temp_data)
    np.random.shuffle(data)

    data = data.reshape(len(temp_data), seq_len, -1)

    #* return a array of shape: (batch size, sequence length, features)

    return data

class Dataset_full_data(Dataset):
    """Dataset object: maintain data
    
    Args:
        data
    """
    def __init__(self, dataset) -> None:
        super().__init__()

        self.dataset = dataset

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self) -> int:
        return self.dataset.shape[0]
