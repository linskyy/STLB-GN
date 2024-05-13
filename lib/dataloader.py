import torch
import numpy as np
import torch.utils.data
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        # data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return scaler

def split_data_by_days(data, val_days, test_days, interval=30):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    x=-T * int(test_days)
    test_data = data[-T*int(test_days):]
    val_data = data[-T*int(test_days + val_days): -T*int(test_days)]
    train_data = data[:-T*int(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    #load raw st dataset
    data = load_st_dataset(args.dataset)        # B, N, D
    #normalize st data
    # scaler = normalize_dataset(data, normalizer, args.column_wise)

    L, N, F = data.shape

    feature_list = [data]


    # numerical time_in_day
    time_ind    = [i%args.steps_per_day / args.steps_per_day for i in range(data.shape[0])]
    time_ind    = np.array(time_ind)
    time_in_day = np.tile(time_ind, [1, N, 1]).transpose((2, 1, 0))
    feature_list.append(time_in_day)

    # numerical day_in_week
    day_in_week = [(i // args.steps_per_day)%args.days_per_week for i in range(data.shape[0])]
    day_in_week = np.array(day_in_week)
    day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((2, 1, 0))
    feature_list.append(day_in_week)

    # data = np.concatenate(feature_list, axis=-1)
    x, y = Add_Window_Horizon(data, args.lag, args.horizon, single)
    x_day, y_day = Add_Window_Horizon(time_in_day, args.lag, args.horizon, single)
    x_week, y_week = Add_Window_Horizon(day_in_week, args.lag, args.horizon, single)
    x, y = np.concatenate([x, x_day, x_week], axis=-1), np.concatenate([y, y_day, y_week], axis=-1)

    # spilit dataset by days or by ratio
    if args.test_ratio > 1:
        x_train, x_val, x_test = split_data_by_days(x, args.val_ratio, args.test_ratio)
        y_train, y_val, y_test = split_data_by_days(y, args.val_ratio, args.test_ratio)
        # day_train, day_val, day_test = split_data_by_days(time_in_day, args.val_ratio, args.test_ratio)
        # week_train, week_val, week_test = split_data_by_days(day_in_week, args.val_ratio, args.test_ratio)
    else:
        x_train, x_val, x_test = split_data_by_ratio(x, args.val_ratio, args.test_ratio)
        y_train, y_val, y_test = split_data_by_ratio(y, args.val_ratio, args.test_ratio)
        # week_train, week_val, week_test = split_data_by_ratio(day_in_week, args.val_ratio, args.test_ratio)
    # #normalize st data
    scaler = normalize_dataset(x_train[..., :args.input_dim], normalizer, args.column_wise)
    # # scaler1 = normalize_dataset(data_val, normalizer, args.column_wise)
    # # scaler2 = normalize_dataset(data_test, normalizer, args.column_wise)
    # # scaler3 = normalize_dataset(data, normalizer, args.column_wise)
    #
    x_train[..., :args.input_dim] = scaler.transform(x_train[..., :args.input_dim])
    x_val[..., :args.input_dim] = scaler.transform(x_val[..., :args.input_dim])
    x_test[..., :args.input_dim] = scaler.transform(x_test[..., :args.input_dim])

    print('Train: ', x_train.shape, y_train.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)

    ##############get dataloader######################
    train_dataloader = data_loader(x_train, y_train, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val[..., 0]) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler

def get_adjacency_matrix2(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A
