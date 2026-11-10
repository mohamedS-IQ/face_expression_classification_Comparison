import argparse
import base64
import io
import json
import urllib.request

import common
from qwen_zeroshot import PROMPT, parse_answer, to_pil

URL = "http://localhost:1234/v1/chat/completions"


def ask(model, pixels):
    buffer = io.BytesIO()
    to_pil(pixels).save(buffer, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    content = [{"type": "image_url", "image_url": {"url": image_url}}, {"type": "text", "text": PROMPT}]
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 5, "temperature": 0}
    request = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2vl-fer-finetuned", help="model name as LM Studio lists it")
    parser.add_argument("--n-test", type=int, default=200, help="use only the first N test images")
    args = parser.parse_args()

    data = common.load_data()
    x_test, y_test = data["x_test"][:args.n_test], data["y_test"][:args.n_test]
    predictions = []
    for pixels in x_test:
        predictions.append(parse_answer(ask(args.model, pixels)))
        print(f"{len(predictions)}/{len(x_test)}", end="\r")
    print()

    scores = common.evaluate(y_test, predictions)
    unparsed = sum(p == -1 for p in predictions)
    print(f"LM Studio {args.model}: accuracy {scores['accuracy']:.4f}   macro F1 {scores['macro_f1']:.4f}   "
          f"unparsed {unparsed}   on {len(y_test)} test images")

    internet_x, _ = common.load_internet_images()
    guesses = [parse_answer(ask(args.model, pixels)) for pixels in internet_x]
    pairs = [f"{true} -> {common.EMOTIONS[g] if g >= 0 else 'unclear'}" for true, g in zip(common.EMOTIONS, guesses)]
    print("internet images (true -> predicted): " + ", ".join(pairs))


if __name__ == "__main__":
    main()
