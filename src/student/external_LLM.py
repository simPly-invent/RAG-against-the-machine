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
from student.errors import ModelLoadError, SourceFileError

MODEL_NAME = "Qwen/Qwen3-0.6B"


@lru_cache(maxsize=1)
def load_model() -> Tuple[PreTrainedTokenizerBase, PreTrainedModel, str]:
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as exc:
        raise ModelLoadError(MODEL_NAME, str(exc)) from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return (tokenizer, model, device)


def build_context(
        sources: List[MinimalSource],
        max_context_length: int = 2000) -> str:
    context = ""
    for source in sources:
        try:
            text = Path(source.file_path).read_text(errors="ignore")
        except OSError as exc:
            raise SourceFileError(source.file_path, str(exc)) from exc
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
    messages = [
        {"role": "system",
         "content": "Answer the question using only the provided context. "
                    "Be concise."},
        {"role": "user",
         "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    # apply_chat_template builds the exact prompt format the model expects.
    # enable_thinking=False keeps the token budget for the answer itself:
    # Qwen3 otherwise spends it inside <think>...</think>.
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return cast(str, answer).strip()
