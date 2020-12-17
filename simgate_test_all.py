import os
import utils.utils
from models.simgate_TRADE import TRADE
from utils.simgate_multiwoz import prepare_data

kwargs = utils.utils.parse_args()
kwargs['binary_gates'] = True
kwargs['USR_SYS_tokens'] = True
model_dir = "save/binarySGate_USRSYStokens_bs120-TRADE-multiwozDST"
logger = simple_logger(kwargs) if kwargs['log_path'] else None

_, _, test, lang, slot_list, domain_gate, slot_gate, value_gate, domain_map, domain_slot_map, vocab_size_train = prepare_data(training=False, **kwargs)
kwargs['eval_slots'] = slot_list[3]

for model_path in os.listdir(model_dir):
    kwargs['model_path'] = os.path.join(model_dir,model_path)

    model = TRADE(lang, slot_list, domain_gate, slot_gate, value_gate, **kwargs)
    model.eval()

    model.test(test, slot_list[3], kwargs['eval_slots'], logger)