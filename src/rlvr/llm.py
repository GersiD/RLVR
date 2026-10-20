from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, StopStringCriteria

'''
Manages LLM deployments locally that allow for easy access to LLMs and their APIs. This file is responsible for managing the deployment of LLMs, including starting and stopping the server, as well as providing an interface for interacting with the LLMs.
'''
@dataclass
class Rollout:
    '''
    Completions sampled by LLM.sample, laid out for LLM.log_probs.

    Prompts are left-padded to a shared length, so every completion starts at column prompt_len.

    Attributes:
        sequences: Prompt + completion token ids, shape (N, prompt_len + C).
        attention_mask: 1 for real prompt tokens and scored completion tokens, shape (N, prompt_len + C).
        completion_mask: 1 for the tokens each completion actually sampled, through its first EOS or the end of
            its stop string, shape (N, C).
        finished: Whether each completion ended on its own rather than being cut off at max_new_tokens,
            bool, shape (N,).
        prompt_len: Width of the padded prompt block.
        texts: Decoded completions, special tokens stripped.
    '''
    sequences: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    finished: torch.Tensor
    prompt_len: int
    texts: list[str]


def default_device() -> str:
    '''Pick CUDA, then Apple MPS, then CPU.'''
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _StopRecorder(StoppingCriteria):
    '''
    Wraps a stopping criterion and records how many completion tokens each row had when it first fired.

    Generation pads finished rows with pad_token_id, which for many models is also EOS, so the padding alone
    can't show where a row stopped on a stop string.
    '''

    def __init__(self, inner: StoppingCriteria, prompt_len: int, batch_size: int, device: torch.device):
        self.inner = inner
        self.prompt_len = prompt_len
        self.length = torch.full((batch_size,), -1, device=device)  # -1 until the row stops

    def __call__(self, input_ids: torch.LongTensor, scores: tuple[torch.FloatTensor] | None, **kwargs) -> torch.BoolTensor:
        done = self.inner(input_ids, scores, **kwargs)
        self.length[done & (self.length < 0)] = input_ids.shape[1] - self.prompt_len
        return done


class LLM(nn.Module, ABC):
    '''
    A trainable language model loaded from a local directory: the interface GRPO trains through.

    Subclasses load their weights in __init__ and implement sampling, per-token log-probs, and saving.
    parameters() must cover every trainable weight, so the model can go straight into a torch optimizer.
    '''

    def __init__(self, model_path: str | Path):
        '''
        Args:
            model_path: Local directory holding the model.

        Raises:
            FileNotFoundError: If model_path is not a directory.
        '''
        super().__init__()
        self.model_path = Path(model_path)
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")

    @property
    @abstractmethod
    def device(self) -> torch.device:
        '''Device the model's weights are on.'''

    @abstractmethod
    def sample(
        self,
        prompts: list[str],
        num_samples: int,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        stop: Sequence[str] = (),
    ) -> Rollout:
        '''
        Sample num_samples completions for each prompt, without gradients.

        The rows for one prompt must be contiguous. temperature=0 decodes greedily. A completion also ends once
        it contains any string in stop; the stop string stays in the completion and in completion_mask.
        '''

    @abstractmethod
    def log_probs(self, sequences: torch.Tensor, attention_mask: torch.Tensor, prompt_len: int) -> torch.Tensor:
        '''
        Log-probability of each completion token under the current weights, shape (N, C).

        Must keep the autograd graph, so a loss built from it can be backpropagated.
        '''

    @abstractmethod
    def save(self, path: str | Path) -> None:
        '''Write the model to path, in a form the subclass can load back from there.'''


class HFCausalLM(LLM):
    '''
    A causal LM run with Hugging Face transformers, such as Qwen2.5. Any checkpoint AutoModelForCausalLM can load
    should work.

    Use an Instruct checkpoint (e.g. "Qwen/Qwen2.5-0.5B-Instruct") for chat. Base checkpoints aren't trained
    on the chat template and tend to repeat themselves.
    '''

    def __init__(self, model_path: str | Path, device: str | None = None, dtype: torch.dtype = torch.float32):
        '''
        Args:
            model_path: Local directory holding the model, as written by save_pretrained or
                `hf download --local-dir` (e.g. "models/base/qwen2.5-0.5b").
            device: Device to load onto. Defaults to default_device().
            dtype: Weight dtype. Keep float32 for training: updates at RL learning rates (~1e-6) mostly
                round away in bf16.

        Raises:
            FileNotFoundError: If model_path is not a directory.
        '''
        super().__init__(model_path)
        # local_files_only: never fall back to downloading from the Hub
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.tokenizer.padding_side = "left"  # batched generation needs left padding
        # Load on CPU, then move: device_map="mps" with float32 segfaults (torch 2.14.1, transformers 5.20.2).
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype=dtype, local_files_only=True)
        self.model.to(device or default_device())  # pyright: ignore[reportArgumentType]

    @property
    def device(self) -> torch.device:
        return self.model.device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''Return next-token logits, shape (N, L, vocab_size).'''
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    @torch.no_grad()
    def sample(
        self,
        prompts: list[str],
        num_samples: int,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        stop: Sequence[str] = (),
    ) -> Rollout:
        '''Prompts are tokenized as plain text, without the chat template, which suits base checkpoints.'''
        was_training = self.training
        self.eval()
        enc = self.tokenizer(
            [p for p in prompts for _ in range(num_samples)], return_tensors="pt", padding=True
        ).to(self.device)
        prompt_len = enc["input_ids"].shape[1]
        stopper = None
        if stop:
            criterion = StopStringCriteria(self.tokenizer, list(stop))
            stopper = _StopRecorder(criterion, prompt_len, enc["input_ids"].shape[0], self.device)
        decoding = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
        sequences = self.model.generate(  # pyright: ignore[reportAttributeAccessIssue]
            **enc,
            **decoding,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([stopper]) if stopper else None,
        )
        self.train(was_training)

        completion = sequences[:, prompt_len:]
        width = completion.shape[1]
        is_eos = completion == self.tokenizer.eos_token_id
        # Sampled length: through the first EOS, or through the stop string if that came first. Everything after
        # is padding, which can't be told apart from EOS by token id alone.
        length = torch.where(is_eos.any(1), is_eos.int().argmax(1) + 1, width)
        stop_length = stopper.length if stopper else torch.full_like(length, -1)
        stopped = stop_length >= 0
        length = torch.where(stopped, torch.minimum(length, stop_length), length)
        completion_mask = (torch.arange(width, device=self.device)[None] < length[:, None]).long()
        return Rollout(
            sequences=sequences,
            attention_mask=torch.cat([enc["attention_mask"], completion_mask], dim=1),
            completion_mask=completion_mask,
            finished=is_eos.any(1) | stopped,
            prompt_len=prompt_len,
            texts=self.tokenizer.batch_decode(completion, skip_special_tokens=True),
        )

    def log_probs(self, sequences: torch.Tensor, attention_mask: torch.Tensor, prompt_len: int) -> torch.Tensor:
        logits = self(sequences, attention_mask)[:, prompt_len - 1:-1]  # logits at t predict token t + 1
        return logits.log_softmax(-1).gather(-1, sequences[:, prompt_len:, None]).squeeze(-1)

    def send_messages(self, messages: list[dict[str, str]]) -> str:
        '''
        Generate the model's reply to a chat conversation, using the chat template. Meant for evaluation.

        Args:
            messages: Chat history as role/content dicts, e.g. [{"role": "user", "content": "Who are you?"}].

        Returns:
            The reply text only (up to 256 new tokens), without the prompt or special tokens.
        '''
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256) # pyright: ignore[reportAttributeAccessIssue]
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def save(self, path: str | Path) -> None:
        '''Write the model and tokenizer to path in Hugging Face format, loadable with HFCausalLM(path).'''
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
