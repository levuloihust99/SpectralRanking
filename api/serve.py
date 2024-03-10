import torch
import logging
import argparse

import sanic
from sanic import Sanic
from sanic_cors import CORS
from typing import Text

from transformers import AutoTokenizer, T5ForConditionalGeneration

from lib.utils.logging import do_setup_logging
from lib.nn.modeling import T5CrossEncoder
from lib.data_helpers.dataloader import encode_texts, do_pad
from api.utils.middleware import register_middleware

do_setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Sanic("SpectralRanking")
CORS(app)
register_middleware(app)


@app.route("/score", methods=["POST"])
async def score(request):
    data = request.json
    article = data["article"]
    summaries = data["summaries"]
    generators = []
    summaries_inputs_ids = []
    for g, s in summaries.items():
        generators.append(g)
        g_input_ids = encode_texts(
            app.ctx.tokenizer,
            article, s,
            sep_token='<extra_id_0>',
            eos_token=app.ctx.tokenizer.eos_token
        )
        summaries_inputs_ids.append(g_input_ids)
    summaries_inputs = do_pad(summaries_inputs_ids, pad_token_id=app.ctx.tokenizer.pad_token_id)
    summaries_inputs = {
        "input_ids": torch.tensor(summaries_inputs["input_ids"]),
        "attention_mask": torch.tensor(summaries_inputs["attention_mask"])
    }
    with torch.no_grad():
        scores = app.ctx.model(input_ids=summaries_inputs["input_ids"], attention_mask=summaries_inputs["attention_mask"])
    score_map = {}
    for i in range(len(generators)):
        score_map[generators[i]] = scores[i].item()
    return sanic.json({"score": score_map})


def load_checkpoint(ckpt_path: Text, pretrained_model_path: Text) -> T5CrossEncoder:
    pretrained = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
    model = T5CrossEncoder.from_t5_for_conditional_generation(pretrained)
    saved_state = torch.load(ckpt_path, map_location=lambda s, t: s)
    model_state = saved_state["model_dict"]
    model.load_state_dict(model_state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--pretrained_model_path", default="VietAI/vit5-base")
    parser.add_argument("--port", type=int, default=4567)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    app.ctx.tokenizer = tokenizer
    model = load_checkpoint(args.ckpt_path, args.pretrained_model_path)
    app.ctx.model = model

    app.run(host="0.0.0.0", port=args.port, single_process=True)


if __name__ == "__main__":
    main()
