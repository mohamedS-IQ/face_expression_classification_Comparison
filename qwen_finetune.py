import math

import torch
from peft import LoraConfig, get_peft_model

import common
from qwen_zeroshot import load_model, predict_all, prompt_text, to_pil

EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
LORA_DIR = common.PROJECT_DIR / "qwen_lora"


def build_batch(processor, images, labels, prompt_len):
    texts = [prompt_text(processor, answer=common.EMOTIONS[label]) for label in labels]
    batch = processor(text=texts, images=images, padding=True, return_tensors="pt")
    targets = batch["input_ids"].clone()
    targets[:, :prompt_len] = -100
    targets[batch["attention_mask"] == 0] = -100
    batch["labels"] = targets
    return batch


def train(model, processor, x_train, y_train, epochs):
    processor.tokenizer.padding_side = "right"
    prompt_inputs = processor(text=[prompt_text(processor)], images=[to_pil(x_train[0])], return_tensors="pt")
    prompt_len = prompt_inputs["input_ids"].shape[1]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    total_steps = math.ceil(len(x_train) / BATCH_SIZE)
    model.train()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(x_train)).tolist()
        for step, start in enumerate(range(0, len(order), BATCH_SIZE), start=1):
            index = order[start:start + BATCH_SIZE]
            images = [to_pil(x_train[i]) for i in index]
            batch = build_batch(processor, images, y_train[index], prompt_len).to(model.device)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 50 == 0 or step == total_steps:
                print(f"epoch {epoch}   step {step}/{total_steps}   loss {loss.item():.4f}")
    return model


def main():
    args = common.parse_args(default_epochs=EPOCHS)
    common.set_seed()
    data = common.load_data()
    x_train, y_train = data["x_train"][:args.n_train], data["y_train"][:args.n_train]
    x_test, y_test = data["x_test"][:args.n_test], data["y_test"][:args.n_test]

    model, processor = load_model()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    with common.Timer() as train_timer:
        train(model, processor, x_train, y_train, args.epochs)
    model.save_pretrained(LORA_DIR)

    with common.Timer() as test_timer:
        predictions = predict_all(model, processor, x_test)

    unparsed = sum(p == -1 for p in predictions)
    extra = {"unparsed": unparsed, "trainable_params": common.count_parameters(model, trainable_only=True)}
    common.report("qwen_finetune", y_test, predictions, model, train_timer.seconds, test_timer.seconds, extra=extra)


if __name__ == "__main__":
    main()
