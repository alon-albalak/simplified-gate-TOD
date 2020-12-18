from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler

from models.simgate_TRADE import TRADE
from utils.simgate_multiwoz import prepare_data, prepare_data_multiwoz_22
from utils.logger import simple_logger
import utils.utils


def main(**kwargs):
    logger = simple_logger(kwargs) if kwargs['log_path'] else None

    avg_best, count, accuracy = 0.0, 0, 0.0
    if kwargs['dataset'] == 'multiwoz':
        train, dev, _, lang, slot_list, domain_gate, slot_gate, value_gate, domain_map, domain_slot_map, vocab_size_train = prepare_data(training=True, **kwargs)

    if kwargs['dataset'] == 'multiwoz_22':
        train, dev, _, lang, slot_list, domain_gate, slot_gate, value_gate, vocab_size_train = prepare_data_multiwoz_22(training=True, **kwargs)

    # TODO: if we're not using different slots of train vs eval, then remove this kwarg
    kwargs['eval_slots'] = slot_list[2]

    model = TRADE(lang, slot_list, domain_gate, slot_gate, value_gate, **kwargs)
    model.train()

    optimizer = Adam(model.parameters(), lr=kwargs['learning_rate'])
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, min_lr=kwargs['learning_rate']/20, verbose=True)

    gradient_accumulation_steps = kwargs['batch_size']/kwargs['train_batch_size']

    for epoch in range(30):
        print(f"Epoch {epoch}")
        if logger:
            logger.save()

        optimizer.zero_grad()

        # Initialize vars for std output
        total_loss = 0
        total_loss_pointer = 0
        total_loss_D_gate = 0
        total_loss_S_gate = 0

        pbar = tqdm(enumerate(train), total=len(train))
        for i, data in pbar:

            d_gate = data['domain_gate_label']
            s_gate = data['slot_gate_label']
            v_gate = data['value_gate_label']

            # Calculate outputs
            # if not kwargs['binary_gates']:
            #     outputs_pointer, D_gate_outputs, S_gate_outputs, V_gate_outputs, _ = model(data, slot_list[1])

            #     # calculate pointer_mask based on ground_truth domain_gate_labels and slot_gate_labels

            #     # Compute losses
            #     # loss_D_gate = model.calculate_loss_gate(D_gate_outputs, data['domain_gate_label'])
            #     loss_S_gate = model.calculate_loss_gate(S_gate_outputs, data['slot_gate_label'])
            #     # loss_V_gate = model.calculate_loss_gate(V_gate_outputs, data['value_gate_label'])

            # else:
            outputs_pointer, D_gate_outputs, S_gate_outputs, V_gate_outputs, _ = model.binary_forward(data, slot_list[1], domain_map)
            loss_D_gate = model.calculate_binary_loss_gate(D_gate_outputs, data['domain_gate_label'])
            loss_S_gate = model.calculate_binary_loss_gate(S_gate_outputs, data['slot_gate_label'])
            # loss_V_gate = model.calculate_loss_V_gate(V_gate_outputs, data['value_gate_label'])

            loss_pointer = model.calculate_loss_pointer(outputs_pointer, data['generate_y'], data['mask'])
            loss = loss_pointer + loss_D_gate + loss_S_gate

            # Calculate gradient
            loss.backward()

            # update vars for std output
            total_loss += loss.item()
            total_loss_pointer += loss_pointer.item()
            total_loss_D_gate += loss_D_gate.item()
            total_loss_S_gate += loss_S_gate.item()

            # update model weights
            if ((i+1) % gradient_accumulation_steps) == 0:
                clip_norm = clip_grad_norm_(model.parameters(), kwargs['clip'])
                optimizer.step()
                optimizer.zero_grad()

                # update logger
                if logger:
                    logger.training_update([
                        "training_batch",
                        {"loss": loss.item(),
                         "loss_pointer": loss_pointer.item(),
                         "loss_D_gate": loss_D_gate.item(),
                         "loss_S_gate": loss_S_gate.item()}])

                # Update std output
                batch_num = ((i+1)/gradient_accumulation_steps)
                pbar.set_description(
                    f"Loss: {total_loss/batch_num:.4f},Pointer loss: {total_loss_pointer/batch_num:.4f},D-gate loss {total_loss_D_gate/batch_num:.4f},S-gate loss: {total_loss_S_gate/batch_num:.4f}")

        if ((epoch+1) % kwargs['eval_patience']) == 0:
            model.eval()
            accuracy = model.evaluate(dev, slot_list[2], kwargs['eval_slots'], domain_map, domain_slot_map, avg_best, logger, kwargs['early_stopping'])
            model.train()
            scheduler.step(accuracy)

            if accuracy >= avg_best:
                avg_best = accuracy
                count = 0
                best_model = model
            else:
                count += 1

            if count == kwargs['patience'] or (accuracy == 1.0 and kwargs['early_stopping'] == None):
                if logger:
                    logger.save()
                print("ran out of patience, stopping early")
                break


if __name__ == "__main__":
    main(**utils.utils.parse_args())
