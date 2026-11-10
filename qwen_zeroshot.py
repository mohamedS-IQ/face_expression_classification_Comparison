import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

import common

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
IMAGE_SIZE = 224
BATCH_SIZE = 16
PROMPT = (
    "What emotion does the face in this image show? "
    "Answer with exactly one word from this list: " + ", ".join(common.EMOTIONS) + "."
)


def model_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_NAME, dtype=model_dtype())
    model.to(common.get_device())
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def to_pil(pixels):
    return Image.fromarray(pixels).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))


def prompt_text(processor, answer=None):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    if answer is None:
        return processor.apply_chat_template(messages, add_generation_prompt=True)
    messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return processor.apply_chat_template(messages)


def parse_answer(answer):
    answer = answer.lower()
    found = [(answer.find(name.lower()), label) for label, name in enumerate(common.EMOTIONS) if name.lower() in answer]
    return min(found)[1] if found else -1


def predict_batch(model, processor, images):
    texts = [prompt_text(processor)] * len(images)
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    answers = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return [parse_answer(answer) for answer in answers]


def predict_all(model, processor, images):
    model.eval()
    predictions = []
    for start in range(0, len(images), BATCH_SIZE):
        batch = [to_pil(x) for x in images[start:start + BATCH_SIZE]]
        predictions.extend(predict_batch(model, processor, batch))
        print(f"{len(predictions)}/{len(images)}", end="\r")
    print()
    return predictions


def main():
    args = common.parse_args()
    common.set_seed()
    data = common.load_data()
    x_test, y_test = data["x_test"][:args.n_test], data["y_test"][:args.n_test]
    internet_x, _ = common.load_internet_images()

    model, processor = load_model()
    warm_up_images = x_test[:BATCH_SIZE]
    predict_all(model, processor, warm_up_images)
    with common.Timer() as test_timer:
        predictions = predict_all(model, processor, x_test)
    internet = predict_all(model, processor, internet_x)

    extra = {"unparsed": sum(p == -1 for p in predictions)}
    common.report("qwen_zeroshot", y_test, predictions, model, 0.0, test_timer.seconds, internet, extra=extra)


if __name__ == "__main__":
    main()
