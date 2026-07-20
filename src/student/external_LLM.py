from typing import List, Tuple, cast
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from functools import lru_cache
from student.pymodel import MinimalSource

MODEL_NAME = "Qwen/Qwen3-0.6B"


@lru_cache(maxsize=1)
def load_model() -> Tuple[PreTrainedTokenizerBase, PreTrainedModel, str]:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)  # type: ignore[arg-type]
    return (tokenizer, model, device)


def build_context(
        sources: List[MinimalSource],
        max_context_length: int = 2000) -> str:
    context = ""
    for source in sources:
        text = Path(source.file_path).read_text(errors="ignore")
        snippet = text[
            source.first_character_index:source.last_character_index]
        context += f"File: {source.file_path}\nContent: {snippet}\n"
        if len(context) > max_context_length:
            break
    return context


def generate_answer(
        question: str,
        context: str,
        max_new_tokens: int = 256) -> str:
    tokenizer, model, device = load_model()
    input_text = (
        f"<|im_start|>user Question: {question}<|im_end|>\n"
        f"<|im_start|>system Context: {context}<|im_end|>\n"
        "<|im_start|>assistant Answer:"
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(  # type: ignore[operator]
            **inputs, max_new_tokens=max_new_tokens)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return cast(str, answer)
