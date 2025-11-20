# inference.py
import os
from io import BytesIO
from typing import Optional, Union

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    raise RuntimeError("Missing dependency: install with `pip install qwen-vl-utils`") from e


def _parse_dtype(s: Optional[str]) -> torch.dtype:
    s = (s or "").lower()
    if s in ("bf16", "bfloat16", "bloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    return torch.bfloat16


def _normalize_image(image: Union[str, Image.Image, bytes]) -> Union[str, Image.Image]:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(BytesIO(image)).convert("RGB")
    if isinstance(image, str):
        if image.startswith(("http://", "https://", "data:")):
            return image
        return Image.open(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def load_model_and_processor(
    base_model_id: str,
    lora_repo_or_path: Optional[str],
    device_map: str,
    torch_dtype: torch.dtype,
    lora_subfolder: Optional[str],
    hf_token: Optional[str],
    merge_lora: bool,
):
    model_kwargs = {"torch_dtype": torch_dtype, "device_map": device_map}

    if lora_repo_or_path:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise RuntimeError(
                "LoRA requested but `peft` is not installed. Install with `pip install peft`."
            ) from e

        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model_id, **model_kwargs)

        peft_model = PeftModel.from_pretrained(
            base,
            lora_repo_or_path,
            subfolder=lora_subfolder,
            token=hf_token,
            use_safetensors=True,
        )

        adapter_names = list(getattr(peft_model, "peft_config", {}).keys())
        target = "default" if "default" in adapter_names else (adapter_names[0] if adapter_names else None)
        if target is None:
            raise RuntimeError(f"No adapters found in {lora_repo_or_path}")
        try:
            peft_model.set_adapter(target)
        except AttributeError:
            setattr(peft_model, "active_adapter", target)

        model = peft_model.merge_and_unload() if merge_lora else peft_model
        proc_src = base_model_id
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model_id, **model_kwargs)
        proc_src = base_model_id

    model.eval()
    processor = AutoProcessor.from_pretrained(proc_src)
    return model, processor


@torch.inference_mode()
def infer_single_image(
    model,
    processor,
    image: Union[str, Image.Image, bytes],
    question: str,
    device_map: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float
) -> str:
    img = _normalize_image(image)

    message = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question.strip()},
            ],
        },
    ]

    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([message])

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,  # None for images
        padding=True,
        return_tensors="pt",
    )

    if device_map == "auto" and torch.cuda.is_available():
        inputs = inputs.to("cuda")
    elif device_map == "cuda":
        inputs = inputs.to("cuda")
    elif device_map == "cpu":
        inputs = inputs.to("cpu")

    tokenizer = processor.tokenizer
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id or eos_token_id

    do_sample = temperature is not None and float(temperature) > 0.0
    if not do_sample:
        temperature = None
        top_p = None

    out = model.generate(
        **inputs,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )

    gen_only = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


if __name__ == "__main__":

    IMAGE = "./assets/demo.png"
    QUESTION = "Which category shows the highest Average Price?"

    BASE_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
    DTYPE = "bf16"
    DEVICE_MAP = "auto"
    SYSTEM_PROMPT = "You are a helpful assistant."

    LORA_PATH = "omkarthawakar/EvoLMM"
    LORA_SUBFOLDER = "solver"
    HF_TOKEN = os.getenv("HF_TOKEN")

    MERGE_LORA = False

    MAX_NEW_TOKENS = 1024
    TEMPERATURE = 0.0

    torch_dtype = _parse_dtype(DTYPE)

    model, processor = load_model_and_processor(
        base_model_id=BASE_MODEL,
        lora_repo_or_path=LORA_PATH,
        device_map=DEVICE_MAP,
        torch_dtype=torch_dtype,
        lora_subfolder=LORA_SUBFOLDER,
        hf_token=HF_TOKEN,
        merge_lora=MERGE_LORA,
    )

    answer = infer_single_image(
        model=model,
        processor=processor,
        image=IMAGE,
        question=QUESTION,
        device_map=DEVICE_MAP,
        system_prompt=SYSTEM_PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE
    )
    print(answer)
