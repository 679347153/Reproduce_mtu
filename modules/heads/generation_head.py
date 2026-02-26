import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers import AutoTokenizer, LlamaForCausalLM

from modules.build import HEADS_REGISTRY
@HEADS_REGISTRY.register()
class T5(nn.Module):
    def __init__(self, cfg, variant='t5-small', input_size=768, use_projection=True, **kwargs):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(variant)
        self.model.config.update(kwargs)
        hidden_size = self.model.config.d_model
        self.use_projection = use_projection
        if use_projection:
            self.input_proj = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size))
        else:
            assert input_size == hidden_size, "input_feat_size should be equal to hidden_size!"

    def forward(self, query_embeds, attention_masks, labels=None):
        if self.use_projection:
            query_embeds = self.input_proj(query_embeds)

        if labels is not None:
            outputs = self.model(encoder_outputs=[query_embeds], attention_mask=attention_masks, labels=labels)
            outputs = outputs.logits
        else:
            outputs = self.model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=query_embeds), attention_mask=attention_masks, do_sample=False)
            outputs = outputs[:, 1:] # remove the decoder start token for T5 generation output.
        return outputs

@HEADS_REGISTRY.register()
class Llama2(nn.Module):
    def __init__(self, cfg, variant='meta-llama/Llama-2-7b-hf', input_size=768, use_projection=True, **kwargs):
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained(variant)
        self.model.config.update(kwargs)
        hidden_size = self.model.config.hidden_size
        self.use_projection = use_projection
        if use_projection:
            self.input_proj = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size))
        else:
            assert input_size == hidden_size, "input_feat_size should be equal to hidden_size!"

    def forward(self, query_embeds, attention_masks, labels=None):
        if self.use_projection:
            query_embeds = self.input_proj(query_embeds)

        if labels is not None:
            outputs = self.model(inputs_embeds=query_embeds, attention_mask=attention_masks, labels=labels)
            logits = outputs.logits
            return logits
        else:
            # For generation, we need to handle it differently as Llama doesn't have an encoder-decoder structure
            generated_ids = self.model.generate(
                inputs_embeds=query_embeds,
                attention_mask=attention_masks,
                max_length=50,  # Adjust as needed
                num_return_sequences=1,
                do_sample=False
            )
            return generated_ids