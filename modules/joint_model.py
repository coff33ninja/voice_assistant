import torch
from torch import nn
from transformers import DistilBertModel, DistilBertConfig
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class JointModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    intent_logits: Optional[torch.FloatTensor] = None
    slot_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class JointIntentSlotModel(nn.Module):
    def __init__(self, config: DistilBertConfig):  # Added type hint for config
        super().__init__()
        # Ensure that config is a DistilBertConfig instance or has the necessary attributes
        if not hasattr(config, "num_intent_labels") or not hasattr(
            config, "num_slot_labels"
        ):
            raise ValueError(
                "Config object must have 'num_intent_labels' and 'num_slot_labels' attributes."
            )

        self.num_intent_labels = config.num_intent_labels
        self.num_slot_labels = config.num_slot_labels
        self.config = config

        self.distilbert = DistilBertModel(config)

        # Intent classification head
        self.intent_classifier = nn.Linear(config.dim, self.num_intent_labels)

        # Slot filling head
        # Ensure config has seq_classif_dropout, if not, provide a default
        dropout_rate = (
            config.seq_classif_dropout
            if hasattr(config, "seq_classif_dropout")
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.slot_classifier = nn.Linear(config.dim, self.num_slot_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        intent_labels=None,  # These are for training
        slot_labels=None,  # These are for training
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,  # Always return dict for easier access
        )

        sequence_output = outputs.last_hidden_state
        # CLS token output (for intent classification)
        cls_output = sequence_output[:, 0, :]  # [batch_size, dim]
        intent_logits = self.intent_classifier(cls_output)

        # Sequence output (for slot filling)
        # Apply dropout to the sequence output before passing to the slot classifier
        sequence_output_dropout = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(
            sequence_output_dropout
        )  # [batch_size, seq_len, num_slot_labels]

        total_loss = None
        if intent_labels is not None and slot_labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            # Intent loss
            intent_loss = loss_fct(
                intent_logits.view(-1, self.num_intent_labels), intent_labels.view(-1)
            )

            # Slot loss
            # Only calculate loss for tokens that are not padding (where slot_labels != -100)
            if self.config.num_slot_labels > 0:  # Ensure there are slot labels
                active_loss = (
                    attention_mask.view(-1) == 1
                )  # Consider active tokens based on attention mask
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels.view(-1)[active_loss]

                # Ensure there are active labels to compute loss on, to avoid issues with empty tensors
                if active_logits.shape[0] > 0 and active_labels.shape[0] > 0:
                    # Filter out -100 labels before passing to CrossEntropyLoss
                    valid_labels_mask = active_labels != -100
                    valid_active_logits = active_logits[valid_labels_mask]
                    valid_active_labels = active_labels[valid_labels_mask]

                    if (
                        valid_active_logits.nelement() > 0
                        and valid_active_labels.nelement() > 0
                    ):  # Check if tensors are not empty
                        slot_loss = loss_fct(valid_active_logits, valid_active_labels)
                        total_loss = intent_loss + slot_loss
                    else:  # No valid slot labels to compute loss, only intent loss
                        total_loss = intent_loss
                        slot_loss = torch.tensor(0.0).to(
                            intent_loss.device
                        )  # Assign zero loss for slots
                else:  # No active tokens for slot loss calculation
                    total_loss = intent_loss
                    slot_loss = torch.tensor(0.0).to(
                        intent_loss.device
                    )  # Assign zero loss for slots
            else:  # No slot labels defined, only intent loss
                total_loss = intent_loss
                slot_loss = torch.tensor(0.0).to(
                    intent_loss.device
                )  # Assign zero loss for slots

        if not return_dict:
            output = (intent_logits, slot_logits) + outputs[
                2:
            ]  # outputs[2:] are hidden_states and attentions
            return ((total_loss,) + output) if total_loss is not None else output

        return JointModelOutput(
            loss=total_loss,
            intent_logits=intent_logits,
            slot_logits=slot_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
