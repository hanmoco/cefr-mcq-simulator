# --- OS / CUDA ---
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Python ---
import json
import argparse
import random

# --- Third-party ---
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import LoraConfig, get_peft_model

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# =========================

LEVELS = ["B1", "B2", "C1", "C2"]
LABELS = [" A", " B", " C", " D"]

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def build_prompt(ex):
    passage = ex["text"].strip()
    question = ex["question"].strip()
    options = ex["options"]
    lines = []
    for i, option in enumerate(options):
        line = f"{chr(65+i)}) {option}"
        lines.append(line)
    prompt = (
        f"You are a learner with CEFR {ex['level']} level English proficiency. Answer the following question.\n\n"
        f"[Passage]\n{passage}\n\n"
        f"[Question]\n{question}\n\n"
        "[Options]\n" + "\n".join(lines) + "\n\n"
        "Answer:"
    )
    return prompt

def get_choice_token_ids(tokenizer):
    ids = []
    for label in LABELS:
        token = tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(token) != 1:
            raise ValueError(f"{repr(label)} is not a single token: {token}")
        ids.append(token[0])
    return torch.tensor(ids, dtype=torch.long)

def forward_choice_logits(model, batch, choice_ids, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False
    )
    logits = outputs.logits

    last_pos = attention_mask.sum(dim=1) - 1
    last_logits = logits[
        torch.arange(input_ids.size(0), device=device),
        last_pos
    ]

    choice_logits = last_logits[:, choice_ids]
    return choice_logits

def train_epoch(model, dataloader, optimizer, choice_ids, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()
        choice_logits = forward_choice_logits(
            model, batch, choice_ids, device
        )
        teacher_soft = batch["teacher_soft"].to(device)

        teacher_soft = safe_kl_teacher(teacher_soft)

        logp = F.log_softmax(choice_logits, dim=-1)
        loss = F.kl_div(logp, teacher_soft, reduction="batchmean")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def apply_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj"
        ],
    )
    model = get_peft_model(model, lora_config)
    return model

def safe_kl_teacher(teacher_soft, eps=1e-8):
    teacher_soft = torch.nan_to_num(
        teacher_soft, nan=0.0, posinf=0.0, neginf=0.0
    )
    teacher_soft = teacher_soft.clamp(min=eps)
    teacher_soft = teacher_soft / teacher_soft.sum(dim=-1, keepdim=True)
    return teacher_soft

@torch.no_grad()
def search_temperature(model, dataloader, choice_ids, device, temps):
    model.eval()
    all_logits, all_teacher_soft = [], []

    for batch in dataloader:
        logits = forward_choice_logits(
            model, batch, choice_ids, device
        )
        all_logits.append(logits)
        all_teacher_soft.append(batch["teacher_soft"].to(device))
    
    logits = torch.cat(all_logits, dim=0)
    teacher_soft = torch.cat(all_teacher_soft, dim=0)
    
    teacher_soft = safe_kl_teacher(teacher_soft)

    best_T, best_kl = None, float("inf")

    for T in temps:
        logp = F.log_softmax(logits/T, dim=-1)
        kl = F.kl_div(logp, teacher_soft, reduction="batchmean").item()
        if kl < best_kl:
            best_kl = kl
            best_T = T
    
    return best_T, best_kl

@torch.no_grad()
def evaluate(
    model, dataloader, choice_ids, device, T=1
):
    model.eval()
    kls, maes, mses = [], [], []
    top1_match, accuracy = [], []

    for batch in dataloader:
        logits = forward_choice_logits(
            model, batch, choice_ids, device
        )
        teacher_soft = batch["teacher_soft"].to(device)

        teacher_soft = safe_kl_teacher(teacher_soft)

        label = batch["labels"].to(device)
        logp = F.log_softmax(logits/T, dim=-1)

        prob = logp.exp()

        kls.append(F.kl_div(logp, teacher_soft, reduction="batchmean").item())
        maes.append(torch.mean(torch.abs(prob-teacher_soft)).item())
        mses.append(torch.mean((prob-teacher_soft)**2).item())

        model_top1 = prob.argmax(dim=1)
        learner_top1 = teacher_soft.argmax(dim=1)

        top1_match.append((model_top1 == learner_top1).float().mean().item())
        accuracy.append((model_top1 == label).float().mean().item())
    
    return {
        "KL": sum(kls) / len(kls),
        "MAE": sum(maes) / len(maes),
        "MSE": sum(mses) / len(mses),
        "Top1": sum(top1_match) / len(top1_match),
        "Acc": sum(accuracy) / len(accuracy)
    }

class MCQDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        ex = self.data[index]
        prompt = build_prompt(ex)
        enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding=False)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": int(ex["label"]),
            "teacher_soft": torch.tensor(ex["teacher_soft"], dtype=torch.float),
        }
    
def collate(batch, pad_token_id):
    B = len(batch)
    L = max(len(f["input_ids"]) for f in batch)

    input_ids = torch.full((B, L), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, L), dtype=torch.long)
    labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
    teacher_soft = torch.stack([ex["teacher_soft"] for ex in batch])

    for i, ex in enumerate(batch):
        input_ids[i, :len(ex["input_ids"])] = torch.tensor(ex["input_ids"])
        attention_mask[i, :len(ex["attention_mask"])] = torch.tensor(ex["attention_mask"])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "teacher_soft": teacher_soft
    }

def apply_split_json(data, split_json_path, split, passage_id="passage_id"):
    with open(split_json_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    
    target_pids = set(spec[split])
    filtered = []

    for ex in data:
        pid = ex[passage_id]
        if pid in target_pids:
            filtered.append(ex)

    return filtered

import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def _collect_val_logits_and_teacher(model, dataloader, choice_ids, device):
    model.eval()
    all_logits = []
    all_teacher = []
    for batch in dataloader:
        logits = forward_choice_logits(model, batch, choice_ids, device)  # (B, C)
        teacher = batch["teacher_soft"].to(device)                        # (B, C)
        teacher = safe_kl_teacher(teacher)                                # 안정화
        all_logits.append(logits.detach())
        all_teacher.append(teacher.detach())
    return torch.cat(all_logits, dim=0), torch.cat(all_teacher, dim=0)    # (N, C), (N, C)


def fit_temperature_lbfgs_for_level(
    logits,            # torch.Tensor (N, C) on device
    teacher_soft,      # torch.Tensor (N, C) on device (already safe_kl_teacher)
    init_T=1.0,
    max_iter=500,
    lr=0.01,
    clamp_T=(0.05, 20.0),
):
    """
    Minimize KL(teacher || softmax(logits/T)) w.r.t. T using LBFGS.
    Equivalent to minimizing CE(teacher, softmax(logits/T)) since H(teacher) is constant.
    """
    device = logits.device
    logT = torch.nn.Parameter(torch.tensor(math.log(init_T), dtype=torch.float32, device=device))
    opt = torch.optim.LBFGS([logT], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)

        T = torch.exp(logT)

        # optional clamp: keep T in a sane range (stabilizes LBFGS)
        if clamp_T is not None:
            T = torch.clamp(T, min=clamp_T[0], max=clamp_T[1])

        logp = F.log_softmax(logits / T, dim=-1)          # (N, C)
        
        # loss = -(teacher_soft * logp).sum(dim=-1).mean()  # scalar
        loss = F.kl_div(logp, teacher_soft, reduction="batchmean")
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        T = torch.exp(logT).item()
        if clamp_T is not None:
            T = float(min(max(T, clamp_T[0]), clamp_T[1]))
    return T

def search_temperature_lbfgs(model, val_loader, choice_ids, device,
                            init_T=1.0, max_iter=200, lr=0.1):
    logits, teacher = _collect_val_logits_and_teacher(model, val_loader, choice_ids, device)
    T = fit_temperature_lbfgs_for_level(
        logits, teacher,
        init_T=init_T,
        max_iter=max_iter,
        lr=lr,
        clamp_T=(0.05, 20.0),
    )
    # 참고용: 최적화된 T에서 KL도 계산해주기
    with torch.no_grad():
        logp = F.log_softmax(logits / T, dim=-1)
        kl = F.kl_div(logp, teacher, reduction="batchmean").item()
    return T, kl


@torch.no_grad()
def confusion_matrix(
    model, dataloader, choice_ids, device
):
    model.eval()
    counts = {"mO_lO": 0, "mX_lX": 0, "mO_lX": 0, "mX_lO": 0, "N": 0}

    for batch in dataloader:
        logits = forward_choice_logits(model, batch, choice_ids, device)
        teacher_soft = batch["teacher_soft"].to(device)
        label = batch["labels"].to(device)

        model_top1 = F.softmax(logits, dim=-1).argmax(dim=1)
        learner_top1 = teacher_soft.argmax(dim=1)

        model_correct = model_top1 == label
        learner_correct = learner_top1 == label

        counts["mO_lO"] += (model_correct & learner_correct).sum().item()
        counts["mX_lX"] += (~model_correct & ~learner_correct).sum().item()
        counts["mO_lX"] += (model_correct & ~learner_correct).sum().item()
        counts["mX_lO"] += (~model_correct & learner_correct).sum().item()
        counts["N"] += label.numel()
    
    return counts

@torch.no_grad()
def collect_test_outputs(
    model, dataloader, choice_ids, device, temps, level
):
    model.eval()
    rows = []

    T = temps[level]

    for batch in dataloader:
        logits = forward_choice_logits(model, batch, choice_ids, device)

        prob_pre = F.softmax(logits, dim=-1)
        prob_post = F.softmax(logits/T, dim=-1)

        labels = batch["labels"].to(device)
        teacher_soft = batch["teacher_soft"].to(device)

        teacher_top1 = teacher_soft.argmax(dim=1)
        model_pred_top1 = prob_pre.argmax(dim=1)

        B = labels.size(0)
        for i in range(B):
            rows.append({
                "id": None,
                "level": level,
                "label": int(labels[i].item()),
                "teacher_soft": teacher_soft[i].cpu().tolist(),
                "teacher_top1": int(teacher_top1[i].item()),
                "model_probs_preT": prob_pre[i].cpu().tolist(),
                "model_probs_postT": prob_post[i].cpu().tolist(),
                "model_pred_top1": int(model_pred_top1[i].item()),
            })
    
    return rows

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--split_json", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bf16", action="store_true")
    # parser.add_argument("--save_metrics", action="store_true")
    args = parser.parse_args(argv)

    data = load_jsonl(args.data_path)

    os.makedirs(args.out_dir, exist_ok=True)

    data_val = apply_split_json(
        data,
        args.split_json,
        split="val",
        passage_id="passage_id"
    )

    data_test = apply_split_json(
        data,
        args.split_json,
        split="test",
        passage_id="passage_id"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    device = torch.device(args.device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = None
    
    if args.bf16 and not torch.cuda.is_available():
        raise ValueError("--bf16 requires CUDA")

    choice_ids = get_choice_token_ids(tokenizer).to(device)

    val_loaders = {}
    test_loaders = {}

    for lv in LEVELS:
        val_loaders[lv] = DataLoader(
            MCQDataset([x for x in data_val if x["level"] == lv], tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
        )
        test_loaders[lv] = DataLoader(
            MCQDataset([x for x in data_test if x["level"] == lv], tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype
    ).to(device)
    
    temps = {}
    for lv in LEVELS:
        best_T, best_kl = search_temperature_lbfgs(
            model, val_loaders[lv], choice_ids, device,
            init_T=1.0,
            max_iter=200,
            lr=0.1,
        )
        temps[lv] = best_T
        # print(f"[Temp-LBFGS] {lv}: T={best_T:.4f}, KL={best_kl:.6f}")

    
    temp_path = os.path.join(args.out_dir, "temperature_by_level.json")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(
            {"temperatures": {lv: float(T) for lv, T in temps.items()}},
            f,
            ensure_ascii=False,
            indent=2
        )

    for split, loaders in [("VAL", val_loaders), ("TEST", test_loaders)]:
        print(f"\n========== {split} ==========")

        if split == "TEST":
            all_test_rows = []

        for lv in LEVELS:
            pre = evaluate(model, loaders[lv], choice_ids, device)
            post = evaluate(model, loaders[lv], choice_ids, device, T=temps[lv])

            cm = confusion_matrix(
                model, loaders[lv], choice_ids, device
            )
            print(
                f"{lv} | "
                f"KL {pre['KL']:.4f}/{post['KL']:.4f} | "
                f"MAE {pre['MAE']:.4f}/{post['MAE']:.4f} | "
                f"MSE {pre['MSE']:.4f}/{post['MSE']:.4f} | "
                f"Top1 {pre['Top1']:.4f} | "
                f"Acc {pre['Acc']:.4f} | "
                f"T {temps[lv]:.4f}"
            )
            print(f"Confusion Matrix {lv}: {cm}")

            if split == "TEST":
                rows = collect_test_outputs(
                    model, loaders[lv], choice_ids, device, temps, level=lv
                )

                test_items_lv = [ex for ex in data_test if ex["level"] == lv]
                for r, ex in zip(rows, test_items_lv):
                    r["id"] = ex["id"]
                
                all_test_rows.extend(rows)
        
        if split == "TEST":
            with open(os.path.join(args.out_dir, "test_outputs.jsonl"), "w", encoding="utf-8") as f:
                for r in all_test_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()