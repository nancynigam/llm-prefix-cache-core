import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class HFEngine:
    def __init__(self, model_id: str = "gpt2", device: Optional[str] = None):
        self.model_id = model_id

        # Choose device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load tokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)

        # GPT-2 has no pad token -> use EOS as pad token
        # GPT-2 was trained on continuous text, not padded batches.
        # Batching requires all sequences to be the same length
        # Padding is needed, but GPT-2 has no pad_token.
        # EOS acts like a full stop + stop sign. GPT-2 treats anything after EOS as “nothing meaningful”
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        # Left padding is more stable for batching later
        self.tok.padding_side = "left"

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

        # Move model to device
        self.model.to(self.device)
        self.model.eval()


    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Convert text into model-ready tensors.
        Returns dict containing input_ids and attention_mask on self.device.
        attention_mask tells the model which tokens are real and which ones to ignore (padding tokens).
        
        Args:
            text: Input text string to encode
            
        Returns:
            Dictionary with keys:
            - 'input_ids': torch.Tensor of shape [1, sequence_length] containing token IDs
            - 'attention_mask': torch.Tensor of shape [1, sequence_length] with 1s for real tokens, 0s for padding
            
        Example:
            >>> engine = HFEngine(model_id="gpt2", device="cpu")
            >>> result = engine.encode("Hello, how are you?")
            >>> result
            {
                'input_ids': tensor([[15496, 11, 527, 499, 30, 0]]),
                'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])
            }
            >>> result['input_ids'].shape
            torch.Size([1, 6])  # [batch_size, sequence_length]
        """
        enc = self.tok(text, return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    
    @torch.inference_mode()
    def prefill(self, prefix_text: str) -> Tuple[Any, List[int], float]:
        """
        Run a forward pass on the prefix to produce past_key_values (PKV).
        Returns: (pkv, prefix_token_ids, prefill_ms)
        """
        enc = self.encode(prefix_text)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", None)

        # Save token ids as a list for hashing/cache keying later
        prefix_token_ids = input_ids[0].tolist()

        t0 = time.perf_counter()
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,                 # <-- critical
            output_hidden_states=False,
            output_attentions=False,
        )
        prefill_ms = (time.perf_counter() - t0) * 1000.0

        pkv = out.past_key_values
        return pkv, prefix_token_ids, prefill_ms
