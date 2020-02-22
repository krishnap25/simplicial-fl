import json
import os
import numpy as np
from .language_utils import word_to_indices, letter_to_vec,\
        bag_of_words, get_word_emb_arr, val_to_vec, split_line, \
        letter_to_idx

# TODO: capitalize global vars names and initialize to None
VOCAB_DIR = 0
emb_array = 0
vocab = 0
embed_dim = 0


def batch_data(data, batch_size, rng=None, shuffle=True, eval_mode=False, full=False):
    """
    data is a dict := {'x': [list], 'y': [list]} with optional fields 'y_true': [list], 'x_true' : [list]
    If eval_mode, use 'x_true' and 'y_true' instead of 'x' and 'y', if such fields exist
    returns x, y, which are both lists of size-batch_size lists
    """
    x = data['x_true'] if eval_mode and 'x_true' in data else data['x']
    y = data['y_true'] if eval_mode and 'y_true' in data else data['y']
    raw_x_y = list(zip(x, y))
    if shuffle:
        assert rng is not None
        rng.shuffle(raw_x_y)
    raw_x, raw_y = zip(*raw_x_y)
    batched_x, batched_y = [], []
    if not full:
        for i in range(0, len(raw_x), batch_size):
            batched_x.append(raw_x[i:i + batch_size])
            batched_y.append(raw_y[i:i + batch_size])
    else:
        batched_x.append(raw_x)
        batched_y.append(raw_y)
    return batched_x, batched_y


def read_data(train_data_dir, test_data_dir, split_by_user=True, dataset="femnist"):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    if dataset == 'sent140':
        global VOCAB_DIR
        global emb_array
        global vocab
        global embed_dim
        VOCAB_DIR = 'sent140/embs.json'
        emb_array, _, vocab = get_word_emb_arr(VOCAB_DIR)
        # print('shape obtained : ' + str(emb_array.shape))
        embed_dim = emb_array.shape[1]

    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    # START Old version :
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        print('reading train file ' + str(file_path))
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        print('reading test file ' + str(file_path))
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])
    # END Old version

    #counter = 0
    #for f in train_files:
    #    file_path = os.path.join(train_data_dir, f)
    #    with open(file_path, 'r') as inf:
    #        cdata = json.load(inf)
    #    clients.extend(cdata['users'])
    #    if 'hierarchies' in cdata:
    #        groups.extend(cdata['hierarchies'])
    #    train_data.update(cdata['user_data'])
    #    counter += 1
    #    if counter == 50:
    #        break

    #clients = [list(train_data.keys()). list(test_data.keys())]
    if split_by_user:
        clients = {
            'train_users': list(train_data.keys()),
            'test_users': list(test_data.keys())
        }
    else:
        clients = {
            'train_users': list(train_data.keys())
        }

    return clients, groups, train_data, test_data


def preprocess_data_x(list_inputs, dataset='femnist', center=False,
                      model_name=None):

    if dataset == 'femnist':
        if center:
            res = np.array(list_inputs) - np.tile(np.mean(list_inputs, axis=0), (len(list_inputs), 1))  # center data
            res = res.tolist()
            return res
        else:
            return list_inputs

    elif dataset == 'shakespeare':
        formatted_list_inputs = shakespeare_preprocess_x(list_inputs)
        if False: #center: # Never center for Shakespeare
            res = np.array(formatted_list_inputs) - np.tile(np.mean(formatted_list_inputs, axis=0), (len(formatted_list_inputs), 1))  # center data
            res = res.tolist()
            return res
        else:
            return formatted_list_inputs

    elif dataset == 'sent140':
        return sent140_preprocess_x(list_inputs).tolist()


def preprocess_data_y(list_labels, dataset='femnist', model_name=None):

    if dataset == 'femnist' and ('cnn' in model_name):
        # return labels as is
        return list_labels
        # return femnist_preprocess_y_int(list_labels)
    elif dataset == 'femnist': # one hot preprocess
        return femnist_preprocess_y_onehot(list_labels)
    elif dataset == 'shakespeare':
        return shakespeare_preprocess_y(list_labels)
    elif dataset == 'sent140':
        return sent140_preprocess_y(list_labels)


def femnist_preprocess_y_onehot(raw_y_batch):
    res = []
    for i in range(len(raw_y_batch)):
        num = np.zeros(62)  # Number of classes
        num[raw_y_batch[i]] = 1.0
        res.append(num)
    return res

def shakespeare_preprocess_x(raw_x_batch):
    x_batch = [[letter_to_idx(l) for l in x] for x in raw_x_batch]
    return x_batch


def shakespeare_preprocess_y(raw_y_batch):
    y_batch = [letter_to_idx(c) for c in raw_y_batch]
    return y_batch

## OLD: 
# def shakespeare_preprocess_x(raw_x_batch):
#     x_batch = [word_to_indices(word) for word in raw_x_batch]
#     x_batch = np.array(x_batch)
#     return x_batch
# 
# 
# def shakespeare_preprocess_y(raw_y_batch):
#     y_batch = [letter_to_vec(c) for c in raw_y_batch]
#     return y_batch


def sent140_preprocess_x(X):
    x_batch = [e[4] for e in X]  # list of lines/phrases
    x = np.zeros((len(x_batch), embed_dim))
    for i in range(len(x_batch)):
        line = x_batch[i]
        words = split_line(line)
        idxs = [vocab[word] if word in vocab.keys() else emb_array.shape[0] - 1
                for word in words]
        word_embeddings = np.mean([emb_array[idx] for idx in idxs], axis=0)
        x[i, :] = word_embeddings
    return x


def sent140_preprocess_y(raw_y_batch):
    res = []
    for i in range(len(raw_y_batch)):
        res.append(float(raw_y_batch[i]))
    return res
