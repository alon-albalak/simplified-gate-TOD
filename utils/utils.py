import argparse
from torch import cuda

UNK_token = 0
PAD_token = 1
SOS_token = 2
EOS_token = 3
ENT_token = 4
SYS_token = 5
USR_token = 6

MAX_GPU_SAMPLES = 10
BINARY_SLOTS = ['hotel-parking', 'hotel-internet']
# CATEGORICAL_SLOTS = ['hotel-pricerange', 'hotel-book day', 'train-day', 'hotel-stars', 'restaurant-food', 'restaurant-pricerange', 'restaurant-book day']
CATEGORICAL_SLOTS = ['hotel-pricerange', 'hotel-book day', 'hotel-stars', 'hotel-area',
                     'train-day',
                     'attraction-area',
                     'restaurant-food', 'restaurant-pricerange', 'restaurant-area', 'restaurant-book day']
ALL_SLOTS = ['hotel-pricerange', 'hotel-type', 'hotel-parking', 'hotel-book stay', 'hotel-book day', 'hotel-book people',
             'hotel-area', 'hotel-stars', 'hotel-internet', 'train-destination', 'train-day', 'train-departure',
             'train-arriveby', 'train-book people', 'train-leaveat', 'attraction-area', 'restaurant-food',
             'restaurant-pricerange', 'restaurant-area', 'attraction-name', 'restaurant-name',
             'attraction-type', 'hotel-name', 'taxi-leaveat', 'taxi-destination', 'taxi-departure',
             'restaurant-book time', 'restaurant-book day', 'restaurant-book people', 'taxi-arriveby']


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_ID", type=str, default="")
    parser.add_argument("-bs", "--batch_size", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=MAX_GPU_SAMPLES)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--parallel_decode", type=bool, default=True)
    parser.add_argument("--hidden", type=int, default=400)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-dr", "--dropout", type=float, default=0.2)
    parser.add_argument('-clip', '--clip', help='gradient clipping', default=10, type=int)
    parser.add_argument('-tfr', '--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, default=0.5)
    parser.add_argument('--load_embedding', type=bool, default=True)
    parser.add_argument('--model_path', type=str, help="Use model_path if you want to load a pre-trained model")
    parser.add_argument('--lang_path', type=str, default="lang_data")
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--dataset', type=str, default='multiwoz')
    parser.add_argument('--task', type=str, default='DST')
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--eval_patience', type=int, default=1)
    parser.add_argument('--gen_sample', action='store_true')
    parser.add_argument('--train_data_ratio', type=int, default=100)
    parser.add_argument('--dev_data_ratio', type=int, default=100)
    parser.add_argument('--test_data_ratio', type=int, default=100)
    parser.add_argument('--percent_ground_truth', type=int, default=100)
    parser.add_argument('--appended_values', type=str, default=None,
                        choices=['NER', 'ground_truth', 'BERT_VE', 'DB'])
    parser.add_argument('--USR_SYS_tokens', action='store_true')
    parser.add_argument('--cpu_only', action='store_true')
    parser.add_argument('--mask_S_gate', action='store_true')

    args = parser.parse_args()

    if not args.cpu_only:
        setattr(args, 'device', 'cuda' if cuda.is_available() else 'cpu')
    else:
        setattr(args, 'device', 'cpu')
    setattr(args, 'UNK_token', UNK_token)
    setattr(args, 'PAD_token', PAD_token)
    setattr(args, 'SOS_token', SOS_token)
    setattr(args, 'EOS_token', EOS_token)
    setattr(args, 'ENT_token', ENT_token)
    setattr(args, 'SYS_token', SYS_token)
    setattr(args, 'USR_token', USR_token)
    setattr(args, 'unk_mask', True)
    setattr(args, 'early_stopping', None)
    setattr(args, 'drop_slots', list())
    setattr(args, 'append_SYS_values', list())

    if args.dataset == 'multiwoz_22':
        assert(args.lang_path == 'lang_data_multiwoz_22')

    assert(not(getattr(args, "appended_values") == "ground_truth" and getattr(args, "append_SYS_values"))),\
        "Ground truth values are not determined by the speaker, appending these values to the system utterance\
                will result in doubly appending values to the system utterance and user utterance"

    return vars(args)
