#!/usr/bin/env python3
"""Engine-neutral benchmark client: the same requests against any OpenAI-compatible server (llama.cpp or Strata).

  1. three greedy prompts (answers kept for a quality check)
  2. a seeded warm-up, then two seeded 1200-token generations          -> decode tok/s
  3. a ~54k-token prompt (llama.cpp source) + 400-token answer, then a ~2.5k-token follow-up turn
                                                                      -> prompt tok/s, decode at depth
Reports the server's own timings (llama.cpp field names; Strata returns the same) and the lowest free VRAM.
usage: LLAMA_CPP_SRC=/path/to/llama.cpp bench_client.py TAG URL [short|long|all]
The VRAM watch reads /sys/class/drm/card1 (an AMD card on Linux; adjust for another card)."""
import json, os, sys, threading, time, urllib.request

TAG, URL = sys.argv[1], sys.argv[2].rstrip("/")
MODE = sys.argv[3] if len(sys.argv) > 3 else "all"
ROOT = os.environ.get("LLAMA_CPP_SRC", "../llama.cpp")   # a llama.cpp checkout: the prompts are its source
LOW = {"reasoning_effort": "low"}
TIMED = ["Write a detailed technical essay (at least 1200 words) on how a garbage collector works, covering mark-and-sweep, generational collection, write barriers and concurrent marking.",
         "Write a detailed technical essay (at least 1200 words) on TCP congestion control, covering slow start, AIMD, fast retransmit, CUBIC and BBR, with trade-offs."]
GREEDY = ["What is 17 * 23? Then list three prime numbers between 50 and 70. Answer briefly.",
          "Write a Python function is_palindrome(s) that ignores case and non-alphanumerics. Code only.",
          "Explain in three sentences why the sky is blue."]
src = "".join(open(f"{ROOT}/{p}", errors="ignore").read() for p in ["src/llama-context.cpp", "src/llama-graph.cpp"])
DOC = src[:190000]
EXTRA = open(f"{ROOT}/src/llama-sampler.cpp", errors="ignore").read()[:8000]
Q1 = "\n\nAbove is part of llama.cpp's source. Explain how llama_context::decode splits a batch into ubatches and where the graph is built and computed. Name the functions involved."
Q2 = "Here is another file excerpt:\n\n" + EXTRA + "\n\nHow does this sampler code relate to what you described? Answer briefly, naming functions."


def vram_free():
    d = "/sys/class/drm/card1/device/"
    return (int(open(d + "mem_info_vram_total").read()) - int(open(d + "mem_info_vram_used").read())) // 2**20


st = {"min": 10**9, "stop": False}
def watch():
    while not st["stop"]:
        st["min"] = min(st["min"], vram_free())
        time.sleep(0.25)
threading.Thread(target=watch, daemon=True).start()


def model_id():
    try:
        return json.load(urllib.request.urlopen(URL + "/v1/models", timeout=10))["data"][0]["id"]
    except Exception:
        return "default"


MODEL = model_id()


SAMPLING = {"temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05}  # llama.cpp's defaults, sent explicitly


def post(messages, n, **kw):
    body = {"model": MODEL, "messages": messages, "max_tokens": n, "chat_template_kwargs": LOW, **SAMPLING, **kw}
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    out = json.load(urllib.request.urlopen(r, timeout=3600))
    out["_wall"] = time.time() - t0
    return out


def text(r):
    m = r["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "\n---\n" + (m.get("content") or "")


res = {"tag": TAG, "url": URL, "model": MODEL}
if MODE in ("short", "all"):
    res["greedy"] = [text(post([{"role": "user", "content": p}], 300, temperature=0, top_k=1, seed=1)) for p in GREEDY]
    post([{"role": "user", "content": "Write a short essay about rivers."}], 400, seed=7)
    res["decode"] = []
    for i, p in enumerate(TIMED):
        t = post([{"role": "user", "content": p}], 1200, seed=11 + i)["timings"]
        res["decode"].append(t)
        print(f"{TAG:12s} decode run {i + 1}: {t['predicted_per_second']:6.2f} tok/s over {t['predicted_n']} tokens", flush=True)
if MODE in ("long", "all"):
    msgs = [{"role": "user", "content": DOC + Q1}]
    r1 = post(msgs, 400, seed=21)
    msgs += [{"role": "assistant", "content": r1["choices"][0]["message"].get("content") or ""}, {"role": "user", "content": Q2}]
    r2 = post(msgs, 400, seed=22)
    res["long"] = [r1["timings"], r2["timings"]]
    res["long_answer"] = text(r1)
    for k, r in (("54k prompt", r1), ("follow-up", r2)):
        t = r["timings"]
        print(f"{TAG:12s} {k:10s}: prompt {t['prompt_n']:6d} tok at {t['prompt_per_second'] or 0:7.1f} tok/s | decode {t['predicted_per_second'] or 0:6.2f} tok/s | wall {r['_wall']:6.1f} s", flush=True)
st["stop"] = True
res["min_vram_free_mib"] = st["min"]
print(f"{TAG:12s} lowest free VRAM during the run: {st['min']} MiB", flush=True)
if "greedy" in res:
    print(f"{TAG:12s} answer 1: " + res["greedy"][0].split("---")[-1].strip()[:140].replace("\n", " | "), flush=True)
json.dump(res, open(f"bench-{TAG}.json", "w"), indent=1)
