import torch
from typing import List, Optional, Tuple, Union
from transformers.models.llama import LlamaForCausalLM


class EarlyExitLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, EARLY_STOP_LAYER):
        super().__init__(config)

        self.past_key_values = None
        self.early_exit_layer = EARLY_STOP_LAYER

    @torch.no_grad()
    def forward_draft_or_large_model(self, in_tokens_small=None, in_features_large=None, position_ids=None, start_layer=None, end_layer=None):
        use_cache = True
        assert self.past_key_values is not None

        if in_tokens_small is not None and in_features_large is not None:
            raise ValueError("You cannot specify both in_tokens_small and in_features_large at the same time")
        elif in_tokens_small is not None:
            batch_size, seq_length = in_tokens_small.shape
        elif in_features_large is not None:
            batch_size, seq_length, _ = in_features_large.shape
        else:
            raise ValueError("You have to specify either in_tokens_small or in_features_large")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if in_tokens_small is not None:
            focu_layer = 0
        else:
            focu_layer = self.early_exit_layer if start_layer is None else start_layer

        if self.past_key_values[focu_layer] is not None:
            past_key_values_length = self.past_key_values[focu_layer][0].shape[2]

        seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = in_tokens_small.device if in_tokens_small is not None else in_features_large.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        if in_tokens_small is not None:
            inputs_embeds = self.model.embed_tokens(in_tokens_small)

        target_device = inputs_embeds.device if in_tokens_small is not None else in_features_large.device
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=target_device
        )
        attention_mask = self.model._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds if in_tokens_small is not None else in_features_large, past_key_values_length
        )

        if hasattr(self.model, "medusa_mask") and self.model.medusa_mask is not None:
            medusa_mask = self.model.medusa_mask
            if medusa_mask.device != attention_mask.device:
                medusa_mask = medusa_mask.to(attention_mask.device)

            medusa_len = medusa_mask.size(-1)

            if medusa_len > 0:
                attention_mask[:, :, :, -medusa_len:][
                    medusa_mask == 0
                ] = attention_mask.min()

        hidden_states = inputs_embeds if in_tokens_small is not None else in_features_large

        if in_tokens_small is not None:
            LAYERS = self.model.layers[:self.early_exit_layer]
            layer_offset = 0
        else:
            s = self.early_exit_layer if start_layer is None else start_layer
            e = end_layer if end_layer is not None else len(self.model.layers)
            LAYERS = self.model.layers[s:e]
            layer_offset = s

        for idx, decoder_layer in enumerate(LAYERS):
            layer_idx = idx + layer_offset
            past_key_value = self.past_key_values[layer_idx]

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
            if layer_outputs[1] is not None:
                self.past_key_values[layer_idx] = layer_outputs[1]

        if in_features_large is not None:
            if start_layer is not None:
                return hidden_states
            return hidden_states, self.model.norm(hidden_states)

        return hidden_states
