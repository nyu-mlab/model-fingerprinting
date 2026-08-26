#!/usr/bin/env python3
"""
Phase 2 only: probe all downloaded models that aren't fully probed yet.
Reuses run_all.py infrastructure (memory gate, crash isolation, ollama lifecycle).
"""
import json, os, subprocess, time, urllib.request, sys

ROOT = os.path.expanduser("~/workspace/model-fingerprinting")
MANIFEST = f"{ROOT}/models/zoo_manifest.json"
GGUF_DIR = f"{ROOT}/models/gguf"
MF_DIR   = f"{ROOT}/models/modelfiles"
RAW_DIR  = f"{ROOT}/raw_responses"
LOGS     = f"{ROOT}/logs"
LOG      = f"{LOGS}/probe_remaining.log"
OLLAMA   = os.path.expanduser("~/.local/bin/ollama")
STORE    = f"{ROOT}/models/ollama_store"
ENV = dict(os.environ,
           OLLAMA_MODELS=STORE,
           OLLAMA_MAX_LOADED_MODELS="1",
           OLLAMA_NUM_PARALLEL="1",
           OLLAMA_KEEP_ALIVE="0",
           PATH=os.path.expanduser("~/.local/bin")+":"+os.environ.get("PATH",""))
TRIALS = 5
N_PROBES = 16
EXPECTED_ROWS = TRIALS * N_PROBES
MEM_FLOOR_MB = 1500

for d in (MF_DIR, RAW_DIR, LOGS, STORE):
    os.makedirs(d, exist_ok=True)

def log(m):
    line=f"[{time.strftime('%H:%M:%S')}] {m}"; print(line,flush=True)
    with open(LOG,"a") as f: f.write(line+"\n")

def free_mb():
    with open("/proc/meminfo") as f:
        mi={l.split(':')[0]:int(l.split()[1]) for l in f}
    return mi.get("MemAvailable",0)//1024

_serve=None
def start_ollama():
    global _serve
    stop_ollama()
    sl=open(f"{LOGS}/ollama-serve.log","a")
    _serve=subprocess.Popen([OLLAMA,"serve"],env=ENV,stdout=sl,stderr=subprocess.STDOUT)
    for _ in range(30):
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags",timeout=3)
            log("  ollama up"); return True
        except Exception:
            time.sleep(1)
    log("  ollama failed to come up"); return False

def stop_ollama():
    global _serve
    try: subprocess.run(["pkill","-f","ollama serve"],capture_output=True)
    except Exception: pass
    _serve=None; time.sleep(2)

def ollama_alive():
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags",timeout=3); return True
    except Exception: return False

def already_done(name):
    p=f"{RAW_DIR}/{name}.jsonl"
    if not os.path.exists(p): return False
    rows=[l for l in open(p) if l.strip()]
    if len(rows)<EXPECTED_ROWS: return False
    return sum(1 for l in rows if json.loads(l).get("error"))==0

def import_model(name):
    mf=f"{MF_DIR}/{name}.Modelfile"
    open(mf,"w").write(f"FROM {GGUF_DIR}/{name}.gguf\n")
    r=subprocess.run([OLLAMA,"create",name,"-f",mf],env=ENV,capture_output=True,text=True,timeout=900)
    ok=r.returncode==0
    if not ok: log(f"  import FAIL: {(r.stdout+r.stderr)[-200:]}")
    return ok

def probe_model(name):
    r=subprocess.run(["python3",f"{ROOT}/scripts/probe.py",name,"--trials",str(TRIALS)],
                     env=ENV,capture_output=True,text=True,timeout=7200)
    log(f"  probe rc={r.returncode}: {(r.stdout+r.stderr).strip()[-180:]}")
    return r.returncode==0

def main():
    t0=time.time()
    man=json.load(open(MANIFEST))
    targets=[m for m in man if m.get("ok") and os.path.exists(m["local"])]
    todo=[m for m in targets if not already_done(m["name"])]
    log(f"##### PROBE REMAINING: {len(todo)} of {len(targets)} OK models need probing  free={free_mb()}MB #####")
    for m in todo:
        log(f"  -> {m['name']} ({m['params_b']}B)")

    if not todo:
        log("Nothing to probe, all done"); return

    if not start_ollama():
        log("FATAL: ollama won't start"); return

    done=set()
    for m in todo:
        name=m["name"]
        # memory gate
        for _ in range(15):
            if not ollama_alive():
                log(f"  ollama down before {name}; restarting"); start_ollama()
            if free_mb()>=MEM_FLOOR_MB: break
            log(f"  low mem {free_mb()}MB, waiting")
            subprocess.run([OLLAMA,"stop",name],env=ENV,capture_output=True); time.sleep(5)

        log(f"=== PROCESS {name} ({m['family']}, {m['params_b']}B)  free={free_mb()}MB ===")
        try:
            if not import_model(name):
                log(f"  -> import failed, skip"); continue
            ok=probe_model(name)
            subprocess.run([OLLAMA,"stop",name],env=ENV,capture_output=True)
            if ok and already_done(name):
                done.add(name); log(f"  -> DONE {name} ({len(done)}/{len(todo)})")
            else:
                log(f"  -> {name} incomplete; ollama_alive={ollama_alive()}")
                if not ollama_alive(): start_ollama()
        except Exception as e:
            log(f"  -> EXCEPTION on {name}: {e}; recovering")
            if not ollama_alive(): start_ollama()

    stop_ollama()
    elapsed=int(time.time()-t0)
    total_probed=sum(1 for m in targets if already_done(m["name"]))
    log(f"##### PROBE COMPLETE in {elapsed}s: {len(done)} new + {total_probed - len(done)} prior = {total_probed}/{len(targets)} models probed #####")

    # write summary
    summary=dict(ts=time.time(), elapsed_s=elapsed, newly_probed=list(done),
                 total_probed=total_probed, total_ok=len(targets),
                 all_probed=[m["name"] for m in targets if already_done(m["name"])])
    json.dump(summary,open(f"{ROOT}/probe_summary.json","w"),indent=2)

if __name__=="__main__":
    main()
