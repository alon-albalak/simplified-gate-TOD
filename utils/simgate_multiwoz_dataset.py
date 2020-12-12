import torch

UNK_token = 0
PAD_token = 1
EOS_token = 3

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Dataset(torch.utils.data.Dataset):
    """Custom dataset for multiwoz"""

    def __init__(self, data_info, source_word2id, target_word2id, mem_word2id):
        """Reads source and target sequences from txt files."""
        self.ID = data_info['ID']
        # list of domains by TURN, not by dialogue
        # self.turn_domain = data_info['turn_domain']
        # list of turn indices per dialogue [0,1,2,3,4, 0,1,2,3, etc]
        self.turn_id = data_info['turn_id']
        # dialogue history by turn
        self.dialog_history = data_info['dialog_history']
        # list of domain-slot value pairs by turn
        self.turn_belief = data_info['turn_belief']
        # list of domain gate labels by turn
        self.domain_gate_label = data_info['domain_gate_label']
        # list of slot gate labels by turn
        self.slot_gate_label = data_info['slot_gate_label']
        # list of value gate labels by turn
        self.value_gate_label = data_info['value_gate_label']
        # map of domain_gate_label's to domain
        self.domain_map = data_info['domain_map']
        # map of slot_gate_label's to domain_slots
        self.domain_slot_map = data_info['domain_slot_map']
        # self.turn_uttr = data_info['turn_uttr']
        # list of accumulated slot values by turn
        self.generate_y = data_info["generate_y"]
        self.num_total_seqs = len(self.dialog_history)
        self.src_word2id = source_word2id
        self.trg_word2id = target_word2id
        self.mem_word2id = mem_word2id

    def __len__(self):
        return self.num_total_seqs

    def __getitem__(self, index):
        """returns one datum (source and target)"""
        ID = self.ID[index]
        turn_id = self.turn_id[index]
        turn_belief = self.turn_belief[index]
        # gating_label = self.gating_label[index]
        domain_gate_label = self.domain_gate_label[index]
        slot_gate_label = self.slot_gate_label[index]
        value_gate_label = self.value_gate_label[index]
        domain_map = self.domain_map[index]
        domain_slot_map = self.domain_slot_map[index]
        # turn_uttr = self.turn_uttr[index]
        # turn_domain = self.preprocess_domain(self.turn_domain[index])
        generate_y = self.generate_y[index]
        generate_y = self.preprocess_slot(generate_y, self.trg_word2id)
        context = self.dialog_history[index]
        context = self.preprocess(context, self.src_word2id)
        context_plain = self.dialog_history[index]

        item_info = {
            "ID": ID,
            "turn_id": turn_id,
            "turn_belief": turn_belief,
            # "gating_label": gating_label,
            "domain_gate_label": domain_gate_label,
            "slot_gate_label": slot_gate_label,
            "value_gate_label": value_gate_label,
            "domain_map": domain_map,
            "domain_slot_map": domain_slot_map,
            "context": context,
            "context_plain": context_plain,
            # "turn_uttr_plain": turn_uttr,
            # "turn_domain": turn_domain,
            "generate_y": generate_y,
        }
        return item_info

    def preprocess(self, sequence, word2idx):
        """Converts words to ids."""
        context = [word2idx[word]
                   if word in word2idx else UNK_token for word in sequence.split()]
        context = torch.Tensor(context)
        return context

    def preprocess_slot(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            v = [word2idx[word] if word in word2idx else UNK_token for word in value.split()] + \
                [EOS_token]
            story.append(v)
        return story

    def preprocess_domain(self, turn_domain):
        domains = {"attraction": 0, "restaurant": 1, "taxi": 2,
                   "train": 3, "hotel": 4, "hospital": 5, "bus": 6, "police": 7}
        return domains[turn_domain]


def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        padded_seqs = torch.ones(len(sequences), max_len).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs.detach()  # torch.tensor(padded_seqs)
        return padded_seqs, lengths

    def merge_multi_response(sequences):
        '''
        merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        '''
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)
        max_len = max([max(l) for l in lengths])
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [PAD_token] * (max_len-len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths

    def mask_gates(mask, domain_gate_label, slot_gate_label, value_gate_label,
                   domain_map, domain_slot_map):
        '''
        add masking from the domain, slot, and value gates to the mask of y_lengths
        '''
        return mask*slot_gate_label

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True)
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge(item_info['context'])
    y_seqs, y_lengths = merge_multi_response(item_info["generate_y"])

    # gating_label = torch.tensor(item_info["gating_label"])
    domain_gate_label = torch.tensor(item_info["domain_gate_label"])
    slot_gate_label = torch.tensor(item_info["slot_gate_label"])
    value_gate_label = torch.tensor(item_info["value_gate_label"])

    # TODO: add masking from domain, slot, value gates to y_lengths and make them a new mask variable
    mask = mask_gates(y_lengths, domain_gate_label, slot_gate_label, value_gate_label,
                      item_info['domain_map'], item_info['domain_slot_map'])
    # turn_domain = torch.tensor(item_info["turn_domain"])

    item_info["context"] = src_seqs.to(DEVICE)
    item_info["context_len"] = src_lengths
    # item_info["gating_label"] = gating_label.to(DEVICE)
    item_info["domain_gate_label"] = domain_gate_label.to(DEVICE)
    item_info["slot_gate_label"] = slot_gate_label.type(torch.float).to(DEVICE)
    item_info["value_gate_label"] = value_gate_label.to(DEVICE)
    # item_info["turn_domain"] = turn_domain.to(DEVICE)
    item_info["mask"] = mask.to(DEVICE)
    item_info["generate_y"] = y_seqs.to(DEVICE)
    item_info["y_lengths"] = y_lengths.to(DEVICE)
    return item_info
