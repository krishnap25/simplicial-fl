SIM_TIMES = ['small', 'medium', 'large']

MAIN_PARAMS = {  # (tot_num_rounds, eval_every_num_rounds, clients_per_round)
    'sent140': {
        'small': (10, 2, 2),
        'medium': (16, 2, 2),
        'large': (1000, 50, 100)
    },
    'femnist': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (400, 20, 100)  # TODO Change here constants for experimentation
    },
    'shakespeare': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (100, 1, 10000)
    }
}


MODEL_PARAMS = {
    'sent140.bag_dnn': (0.0003, 2),  # lr, num_classes
    'sent140.stacked_lstm': (0.0003, 25, 2, 100),  # lr, seq_len, num_classes, num_hidden ;
    'sent140.bag_log_reg': (0.0003, 2, round(1e9)),  # lr, num_classes, max_batch_size
    'femnist.cnn': (0.0003, 62, 16384),  # lr, num_classes, max_batch_size
    'femnist.erm_l2': (0.0003, 62, round(1e9)),  # lr, num_classes, max_batch_size
    'femnist.erm_log_reg': (0.1, 62, round(1e9)),  # lr, num_classes, max_batch_size
    'femnist.rsm_log_reg': (0.1, 62, round(1e9)),  # lr, num_classes, max_batch_size
    'femnist.rsm_l2': (0.0003, 62, round(1e9)),  # lr, num_classes, max_batch_size
    'femnist.log_reg': (2e-2, 62, round(1e9)),  # lr, num_classes, max_batch_size
    'femnist.erm_cnn_log_reg': (2e-2, 62, 16384),
    # Shakespeare
    # lr, seq_len, num_classes, num_hidden, num_lstm_layers, max_batch_size
    'shakespeare.stacked_lstm': (0.0003, 20, 53, 128, 1, 32768),
    'shakespeare.erm_l2': (0.0003, 20, round(1e9)),
    'shakespeare.rsm_l2': (0.0003, 20, round(1e9)),
    'shakespeare.erm_lstm_log_reg': (0.0003, 20, 53, 128, 1, 32768),
    # Sent 140
    'sent140.erm_log_reg': (0.001, 2, round(1e9)),
    'sent140.rsm_log_reg': (0.1, 2, round(1e9)),  # lr, num_classes, max_batch_size
    'sent140.erm_lstm_log_reg': (0.0003, 2, round(1e9)),  # lr, seq_len, num_classes, max_batch_size
}

MAX_UPDATE_NORM = 100000  # reject all updates larger than this amount

ACCURACY_KEY = 'accuracy'
BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'
AVG_LOSS_KEY = 'avg_loss'

# List of regularization parameters tested for validation
REGULARIZATION_PARAMS = [10**i for i in range(-10, -4)]

class OptimLoggingKeys:
    TRAIN_ACCURACY_KEY = 'train_accuracy'
    TRAIN_LOSS_KEY = 'train_loss'
    EVAL_ACCURACY_KEY = 'test_accuracy'
    EVAL_LOSS_KEY = 'test_loss'

TRAINING_KEYS = {OptimLoggingKeys.TRAIN_ACCURACY_KEY,
                 OptimLoggingKeys.TRAIN_LOSS_KEY,
                 OptimLoggingKeys.EVAL_LOSS_KEY}

AGGR_MEAN = 'mean'
