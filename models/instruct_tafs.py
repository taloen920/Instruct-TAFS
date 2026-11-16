"""Instruct-TAFS: instruction-driven, type-aware farmland segmentation."""

from contextlib import contextmanager

import torch
from torch import nn

from .dptef import DPTEF
from .sisd import SISD
from .visual_encoder import DEFAULT_DINO_REPO, DINOv3Encoder


class InstructTAFS(nn.Module):
    """Visual encoding, language guidance, and collaborative decoding.

    Segmentation gradients update DPTEF and SISD. Language hidden states are
    detached so that only the text-generation objective updates Qwen LoRA.
    """

    def __init__(
        self,
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        dino_size="h",
        dino_repo_dir=None,
        dino_ckpt=None,
        text_dim=256,
        sisd_features=128,
        use_lora=True,
        lora_r=16,
        lora_alpha=None,
        lora_dropout=0.05,
        dtype=None,
        device="cuda",
        max_txt_len=256,
        gradient_checkpointing=False,
    ):
        super().__init__()
        try:
            from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise ImportError("Install a Transformers build with Qwen3VLForConditionalGeneration support.") from error

        device = torch.device(device)
        dtype = dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
        self.visual_encoder = DINOv3Encoder(dino_size, dino_repo_dir, dino_ckpt)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.padding_side = "right"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"
        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype=dtype, device_map=None
        )
        self.backbone.requires_grad_(False)
        self.peft_enabled = use_lora
        lora_alpha = lora_alpha if lora_alpha is not None else lora_r * 2
        if use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as error:
                raise ImportError("Install peft and accelerate to enable LoRA training.") from error
            candidates = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
            targets = sorted({name.rsplit(".", 1)[-1] for name, _ in self.backbone.named_modules()} & candidates)
            if not targets:
                raise RuntimeError("No supported LoRA target modules were found in Qwen3-VL.")
            self.backbone = get_peft_model(
                self.backbone,
                LoraConfig(
                    r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                    target_modules=targets, bias="none", task_type=TaskType.CAUSAL_LM,
                ),
            )
            if gradient_checkpointing:
                self.backbone.config.use_cache = False
                self.backbone.gradient_checkpointing_enable()
                self.backbone.enable_input_require_grads()
        config = self.backbone.config
        hidden_size = getattr(getattr(config, "text_config", config), "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Cannot infer the Qwen3-VL text hidden size.")
        self.dptef = DPTEF(hidden_size, text_dim, hidden_dim=text_dim * 2)
        self.sisd = SISD(
            self.visual_encoder.embed_dim,
            features=sisd_features,
            out_channels=self.visual_encoder.out_channels,
            text_dim=text_dim,
        )
        special_ids = set(self.tokenizer.all_special_ids)
        for token, token_id in self.tokenizer.get_added_vocab().items():
            if any(marker in token.lower() for marker in ("image", "vision", "video", "<img", "picture")):
                special_ids.add(token_id)
        self.register_buffer("special_token_ids", torch.tensor(sorted(special_ids), dtype=torch.long), persistent=False)
        self.max_txt_len = max_txt_len
        self.model_config = {
            "architecture": "Instruct-TAFS", "format_version": 2,
            "model_name": model_name, "dino_size": dino_size,
            "dino_repo_dir": str(dino_repo_dir or DEFAULT_DINO_REPO),
            "dino_ckpt": str(dino_ckpt) if dino_ckpt is not None else None,
            "text_dim": text_dim, "sisd_features": sisd_features,
            "use_lora": use_lora, "lora_r": lora_r, "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout, "max_txt_len": max_txt_len,
            "gradient_checkpointing": gradient_checkpointing,
        }
        self.to(device=device)

    @property
    def device(self):
        return next(self.dptef.parameters()).device

    @contextmanager
    def language_inference(self):
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            self.backbone.train(was_training)

    def encode_text_states(self, texts, append_eos=False):
        """Extract detached language states while keeping DPTEF outside no_grad."""
        if append_eos and self.tokenizer.eos_token:
            eos = self.tokenizer.eos_token
            texts = [text if text.endswith(eos) else text + eos for text in texts]
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_txt_len
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with self.language_inference():
            output = self.backbone(**tokens, output_hidden_states=True, return_dict=True, use_cache=False)
        return output.hidden_states[-1].detach(), tokens["input_ids"], tokens["attention_mask"].bool()

    def encode_language_guidance(self, instructions, descriptions):
        instruction_states, instruction_ids, attention = self.encode_text_states(instructions)
        valid = attention & ~torch.isin(instruction_ids, self.special_token_ids)
        description_states, description_ids, description_attention = self.encode_text_states(descriptions, append_eos=True)
        positions = torch.arange(description_ids.shape[1], device=self.device).expand_as(description_ids)
        last_valid = positions.masked_fill(~description_attention, -1).amax(1).clamp_min(0)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_positions = last_valid
        else:
            is_eos = description_ids.eq(eos_id) & description_attention
            eos_positions = positions.masked_fill(~is_eos, -1).amax(1)
            eos_positions = torch.where(eos_positions >= 0, eos_positions, last_valid)
        dtype = self.dptef.text_projection.weight.dtype
        return self.dptef(instruction_states.to(dtype), valid, description_states.to(dtype), eos_positions)

    def forward(self, pixel_values, instructions, descriptions):
        if len(instructions) != pixel_values.shape[0] or len(descriptions) != pixel_values.shape[0]:
            raise ValueError("Images, instructions, and descriptions must have equal batch sizes.")
        visual_features = self.visual_encoder(pixel_values)
        fused_text = self.encode_language_guidance(instructions, descriptions)
        height, width = pixel_values.shape[-2:]
        return self.sisd(visual_features, (height // 16, width // 16), fused_text, (height, width))

    def generate_descriptions(self, images, instructions, max_new_tokens=128):
        if len(images) != len(instructions) or not images:
            raise ValueError("Provide an equal, nonzero number of images and instructions.")
        prompts = []
        for image, instruction in zip(images, instructions):
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": instruction}
            ]}]
            prompts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        with self.language_inference():
            generated = self.backbone.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        texts = self.processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if any(not text.strip() for text in texts):
            raise RuntimeError("Qwen3-VL returned an empty attribute description.")
        return texts

    def save_lora(self, output_dir):
        if self.peft_enabled:
            self.backbone.save_pretrained(output_dir)

    def load_lora(self, adapter_dir):
        if not self.peft_enabled:
            raise ValueError("Enable LoRA before loading an adapter.")
        self.backbone.load_adapter(adapter_dir, adapter_name="evaluation")
        self.backbone.set_adapter("evaluation")
