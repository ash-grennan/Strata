#!/usr/bin/env python3
"""bench_client.py (greedy x3, decode 2x1200 prose, ~54k prompt + follow-up) plus an agent-style edit: a ~9k-character
source file, re-emitted whole with one function renamed and a parameter added - most of the answer is a verbatim copy
of the context, as in omp's edit tool, which is what prompt lookup (--suffix-draft, --lookup-chain) drafts.
Writes bench-TAG.json (bench_client's) and bench-edit-TAG.json.  usage: LLAMA_CPP_SRC=/path/to/llama.cpp bench_tune.py TAG URL"""
import json, os, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TAG, URL = sys.argv[1], sys.argv[2].rstrip("/")
subprocess.run([sys.executable, os.path.join(HERE, "bench_client.py"), TAG, URL, "all"], cwd=HERE)

SRC = open(os.path.join(os.environ.get("LLAMA_CPP_SRC", "../llama.cpp"), "src/llama-sampler.cpp"), errors="ignore").read()
i = SRC.index("static void llama_sampler_top_k_impl")
CODE = SRC[i:i + 9000]
ASK = ("Here is part of a C++ file:\n\n```cpp\n" + CODE + "\n```\n\nRename the function `llama_sampler_top_k_impl` to "
       "`llama_sampler_apply_top_k` everywhere it appears, and give it an extra last parameter `bool sorted_hint` that "
       "it ignores. Output the COMPLETE modified code above in one ```cpp block, everything else unchanged, no comments "
       "or explanation.")
out = []
for rep in range(2):
    body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": f"[edit {rep}]\n" + ASK}],
            "max_tokens": 4096, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 31 + rep,
            "chat_template_kwargs": {"reasoning_effort": "low"}}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=1800))
    t = r["timings"]
    txt = r["choices"][0]["message"].get("content") or ""
    ok = "llama_sampler_apply_top_k" in txt and "sorted_hint" in txt
    out.append({"timings": t, "wall": time.time() - t0, "ok": ok, "chars": len(txt)})
    print(f"{TAG:14s} edit {rep + 1}: {t['predicted_n']} tok at {t['predicted_per_second']:.1f} tok/s, wall {time.time() - t0:.1f} s,"
          f" {'correct' if ok else 'INCOMPLETE'} ({len(txt)} chars)", flush=True)
json.dump(out, open(os.path.join(HERE, f"bench-edit-{TAG}.json"), "w"), indent=1)
