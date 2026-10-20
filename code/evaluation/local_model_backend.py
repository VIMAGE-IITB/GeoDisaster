from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable


def _trust_remote_code() -> bool:
    return os.environ.get("LOCAL_MODEL_TRUST_REMOTE_CODE", "").strip().lower() in {"1", "true", "yes"}


def _format_chat(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        except Exception:
            pass
    chunks: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).upper()
        chunks.append(f"{role}: {msg.get('content', '')}")
    chunks.append("ASSISTANT:")
    return "\n".join(chunks)


def build_local_chat_backend(
    *,
    pretrained: str,
    max_new_tokens: int,
    adapter_path: str = "",
    adapter_root: str = "",
    system_to_adapter: dict[str, str] | None = None,
) -> Callable[[list[dict[str, Any]]], str]:
    """Create a self-contained local chat backend using Transformers.

    `adapter_path` loads one LoRA adapter for all calls. `adapter_root` loads
    per-role adapters from subdirectories named `orc`, `ga`, `pa`, and `vra`;
    `system_to_adapter` selects the active adapter from the system prompt.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=_trust_remote_code())
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": _trust_remote_code(),
        "device_map": "auto",
    }
    if hasattr(torch, "bfloat16"):
        load_kwargs["torch_dtype"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(pretrained, **load_kwargs)
    model.eval()

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)

    adapter_root_path = Path(adapter_root).resolve() if adapter_root else None
    loaded_adapters: set[str] = set()
    if adapter_root_path:
        from peft import PeftModel

        peft_model: Any | None = None
        for adapter_name in ("ORC", "GA", "PA", "VRA"):
            path = adapter_root_path / adapter_name.lower()
            if not path.exists():
                continue
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(
                    model,
                    str(path),
                    adapter_name=adapter_name,
                    is_trainable=False,
                )
            else:
                peft_model.load_adapter(str(path), adapter_name=adapter_name, is_trainable=False)
            loaded_adapters.add(adapter_name)
        if peft_model is not None:
            model = peft_model
            print(f"Loaded role adapters from {adapter_root_path}: {', '.join(sorted(loaded_adapters))}")

    def _call(messages: list[dict[str, Any]]) -> str:
        system_text = str(messages[0].get("content", "")) if messages else ""
        adapter_name = (system_to_adapter or {}).get(system_text)
        if adapter_name in loaded_adapters and hasattr(model, "set_adapter"):
            model.set_adapter(adapter_name)

        prompt = _format_chat(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt")
        first_device = next(model.parameters()).device
        inputs = {key: value.to(first_device) for key, value in inputs.items()}
        input_len = int(inputs["input_ids"].shape[-1])
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][input_len:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return _call
