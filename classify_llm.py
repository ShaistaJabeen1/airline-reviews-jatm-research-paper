"""
Codebook-guided LLM classification of airline reviews
(organizational justice, Kano-informed category, service dimension, sentiment).

Model : Qwen/Qwen2.5-7B-Instruct (open weights), 4-bit NF4, greedy decoding (deterministic)
Usage : python classify_llm.py --input reviews_for_classification.csv --stage validation
        python classify_llm.py --input reviews_for_classification.csv --stage all
Output: llm_labels_<stage>.csv  +  run_info_<stage>.json (versions, model revision, prompt hash)

The prompt below was fixed BEFORE comparing any output with the human reference
standard. Do not edit it between the validation run and the full run.
"""
import argparse, json, os, re, time, hashlib, platform
import pandas as pd

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_CHARS = 4000          # full review text is used (only 5 of 4,225 reviews are longer than this)
BATCH = 8
MAX_NEW_TOKENS = 80
SEED = 20261008

LABELS = {
    "justice": ["Distributive Justice", "Procedural Justice", "Interactional Justice", "General/Other"],
    "kano": ["Must-be", "Performance", "Attractive", "General/Other"],
    "service": ["Flight Delays and Cancellations", "Refunds and Compensation", "Luggage and Baggage",
                "Staff and Cabin Crew", "Seat Comfort and Legroom", "In-flight Food and Meals",
                "Check-in and Boarding Experience", "Booking and Pricing Transparency", "General/Other"],
    "sentiment": ["1", "2", "3", "4", "5"],
}

SYSTEM_PROMPT = """You are an expert coder of airline customer reviews. Code each review on four fields, following this codebook exactly. Judge the MAIN point of the review.

FIELD 1 - justice (organizational justice; choose one)
- Distributive Justice: the OUTCOME received - refund, compensation or voucher denied, partial or unpaid; extra charges; money lost.
- Procedural Justice: the PROCESS - how problems, requests or complaints were handled: speed, transparency, information, rules, rebooking, call-centre or e-mail handling.
- Interactional Justice: how people were TREATED by staff - respect, politeness, empathy, honesty (positive or negative).
- General/Other: no fairness or treatment issue at all (e.g., only the seat or food is described).
Boundary rules: if the customer is mainly upset about NOT GETTING the money or outcome -> Distributive; if mainly about HOW the case was handled (no updates, ignored complaints, slow handling) -> Procedural. Impersonal systems, policies and communication channels -> Procedural; a specific person's behaviour -> Interactional.

FIELD 2 - kano (Kano-informed category; choose one)
- Must-be: a BASIC promise was broken or a basic entitlement/safety was breached - flight cancelled or long-delayed, missed connection, stranded, denied boarding, baggage lost or damaged, money legally owed not paid, threats, abuse or discrimination. Ask: was the core service delivered? If NO -> Must-be.
- Performance: the core service WAS delivered but its QUALITY LEVEL is judged better or worse - staff friendliness or rudeness, queue and waiting times, call-centre responsiveness, seat comfort, food, cleanliness, entertainment, value for money, short delays. Use Performance even if the customer is very angry.
- Attractive: something UNEXPECTED that exceeded normal expectations (staff going out of their way, surprise upgrade, exceptional care).
- General/Other: no service attribute can be identified.

FIELD 3 - service (the ONE service area the review focuses on most)
- Flight Delays and Cancellations: delays, cancellations, missed connections, schedule changes.
- Refunds and Compensation: getting money back or compensated AFTER a problem (refunds, compensation, vouchers, EU261 claims).
- Luggage and Baggage: lost, delayed or damaged baggage; baggage fees or allowance.
- Staff and Cabin Crew: behaviour of ground staff or cabin crew.
- Seat Comfort and Legroom: seats, legroom, cabin comfort.
- In-flight Food and Meals: food and drinks on board.
- Check-in and Boarding Experience: check-in, security, gates, boarding, denied boarding.
- Booking and Pricing Transparency: buying or changing a ticket - website/app booking, fares and price changes, hidden or extra fees at booking, unclear fare rules, change fees, seat-selection or add-on charges, payment problems.
- General/Other: complaint handling in general (call centre, no answer) with no other area, general experience, or no single main area.

FIELD 4 - sentiment (overall tone of the text, 1-5)
1 = very negative (intense anger, warnings to avoid); 2 = negative; 3 = neutral or mixed; 4 = positive with reservations; 5 = very positive.

Answer with ONLY a JSON object, no explanation, in exactly this form:
{"justice": "<label>", "kano": "<label>", "service": "<label>", "sentiment": <1-5>}"""

PROMPT_SHA = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]


def user_msg(text):
    return "Review:\n\"\"\"\n" + str(text)[:MAX_CHARS] + "\n\"\"\""


def _match(field, v):
    v = str(v).strip().strip('"').strip()
    opts = LABELS[field]
    if field == "sentiment":
        m = re.search(r"[1-5]", v); return m.group(0) if m else None
    low = {o.lower(): o for o in opts}
    if v.lower() in low: return low[v.lower()]
    for o in opts:  # tolerant matching, e.g. "Must-be Quality", "Procedural"
        if v.lower().startswith(o.lower().split()[0].lower()) and o != "General/Other":
            return o
    if v.lower().startswith("general") or v.lower() in ("other", "none"): return "General/Other"
    return None


def parse(raw):
    out = {f: None for f in LABELS}
    m = re.search(r"\{.*?\}", raw, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            for f in LABELS:
                if f in d: out[f] = _match(f, d[f])
        except Exception:
            pass
    for f in LABELS:  # fallback: regex per field
        if out[f] is None:
            mm = re.search(rf'"?{f}"?\s*:\s*"?([^",}}\n]+)', raw, re.I)
            if mm: out[f] = _match(f, mm.group(1))
    return out


def load_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=q, device_map="auto")
    model.eval()
    return tok, model


def make_generator(tok, model):
    import torch

    def gen(texts, extra_instruction=""):
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user_msg(t) + extra_instruction}],
            tokenize=False, add_generation_prompt=True) for t in texts]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                               temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
        return tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen


def run(df, gen, out_csv):
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
    done_ids = set(done["index"]) if len(done) else set()
    todo = df[~df["index"].isin(done_ids)]
    print(f"{len(done_ids)} already done, {len(todo)} to go")
    t0 = time.time()
    for s in range(0, len(todo), BATCH):
        chunk = todo.iloc[s:s + BATCH]
        raws = gen(chunk["text"].tolist())
        rows = []
        for (_, r), raw in zip(chunk.iterrows(), raws):
            p = parse(raw); retry = 0
            if any(v is None for v in p.values()):  # one stricter retry
                raw2 = gen([r["text"]], "\n\nReturn ONLY the JSON object with the four fields, using the exact labels from the codebook.")[0]
                p2 = parse(raw2); retry = 1
                p = {f: p[f] or p2[f] for f in p}; raw = raw + " ||RETRY|| " + raw2
            rows.append({"index": r["index"], "Code": r.get("Code", ""), **{f"llm_{f}": (p[f] or "PARSE_FAIL") for f in p},
                         "retry": retry, "raw_output": raw.strip()})
        pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
        n = s + len(chunk)
        if n % 80 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"{n}/{len(todo)}  {el/60:.1f} min elapsed, ~{el/n*(len(todo)-n)/60:.0f} min left", flush=True)
    res = pd.read_csv(out_csv)
    print("PARSE_FAIL counts:", {f: int((res[f'llm_{f}'] == 'PARSE_FAIL').sum()) for f in LABELS})
    return res


def run_info(stage, n):
    info = {"stage": stage, "n": n, "model": MODEL_ID, "quantization": "4-bit NF4 (bitsandbytes), fp16 compute",
            "decoding": "greedy (do_sample=False)", "max_new_tokens": MAX_NEW_TOKENS, "max_chars": MAX_CHARS,
            "batch": BATCH, "prompt_sha256_12": PROMPT_SHA, "python": platform.python_version(),
            "date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    try:
        import torch, transformers, bitsandbytes
        info.update(torch=torch.__version__, transformers=transformers.__version__, bitsandbytes=bitsandbytes.__version__,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
        from huggingface_hub import model_info
        info["model_revision"] = model_info(MODEL_ID).sha
    except Exception as e:
        info["version_error"] = str(e)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="reviews_for_classification.csv")
    ap.add_argument("--stage", choices=["validation", "all"], required=True)
    a = ap.parse_args()
    import torch; torch.manual_seed(SEED)
    df = pd.read_csv(a.input)
    if a.stage == "validation":
        df = df[df["Code"].notna()].sort_values("Code")
    out = f"llm_labels_{a.stage}.csv"
    tok, model = load_model()
    res = run(df, make_generator(tok, model), out)
    json.dump(run_info(a.stage, len(res)), open(f"run_info_{a.stage}.json", "w"), indent=1)
    print("Saved", out)
