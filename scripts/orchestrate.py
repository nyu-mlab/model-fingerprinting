#!/usr/bin/env python3
"""
Orchestrator: wait for each zoo model to finish downloading, import into Ollama,
run the full probe battery, then unload. Idempotent & resumable: skips models
whose raw_responses/<name>.jsonl already has the expected row count.
"""
import json, os, subprocess, time, sys

ROOT = os.path.expanduser("~/workspace/model-fingerprinting")
MANIFEST = f"{ROOT}/models/zoo_manifest.json"
GGUF_DIR = f"{ROOT}/models/gguf"
MF_DIR   = f"{ROOT}/models/modelfiles"
RAW_DIR  = f"{ROOT}/raw_responses"
LOG      = f"{ROOT}/logs/orchestrate.log"
OLLAMA_BIN = os.path.expanduser("~/.local/bin/ollama")
ENV = dict(os.environ, OLLAMA_MODELS=f"{ROOT}/models/ollama_store",
           PATH=os.path.expanduser("~/.local/bin")+":"+os.environ.get("PATH",""))
TRIALS = 5
N_PROBES = 16
EXPECTED_ROWS = TRIALS * N_PROBES

def log(m):
    line=f"[{time.strftime('%H:%M:%S')}] {m}"; print(line,flush=True)
    with open(LOG,"a") as f: f.write(line+"\n")

def already_done(name):
    p=f"{RAW_DIR}/{name}.jsonl"
    if not os.path.exists(p): return False
    with open(p) as f:
        rows=[l for l in f if l.strip()]
    if len(rows)<EXPECTED_ROWS: return False
    # ensure no error rows / empty responses dominating
    errs=sum(1 for l in rows if json.loads(l).get("error"))
    return errs==0

def imported(name):
    try:
        out=subprocess.run([OLLAMA_BIN,"list"],env=ENV,capture_output=True,text=True,timeout=30)
        return any(line.split(":")[0].split()[0]==name or line.startswith(name+" ") or line.startswith(name+":")
                   for line in out.stdout.splitlines())
    except Exception:
        return False

def import_model(name):
    mf=f"{MF_DIR}/{name}.Modelfile"
    with open(mf,"w") as f:
        f.write(f"FROM {GGUF_DIR}/{name}.gguf\n")
    r=subprocess.run([OLLAMA_BIN,"create",name,"-f",mf],env=ENV,
                     capture_output=True,text=True,timeout=600)
    ok = r.returncode==0 and "success" in (r.stdout+r.stderr).lower()
    if not ok:
        log(f"  import FAIL {name}: {(r.stdout+r.stderr)[-200:]}")
    return ok

def probe_model(name):
    r=subprocess.run(["python3",f"{ROOT}/scripts/probe.py",name,"--trials",str(TRIALS)],
                     env=ENV,capture_output=True,text=True,timeout=7200)
    log(f"  probe rc={r.returncode}: {r.stdout.strip()[-160:]} {r.stderr.strip()[-160:]}")
    return r.returncode==0

def unload(name):
    # free RAM: set keep_alive 0
    try:
        subprocess.run([OLLAMA_BIN,"stop",name],env=ENV,capture_output=True,timeout=30)
    except Exception: pass

def load_manifest():
    if not os.path.exists(MANIFEST): return []
    with open(MANIFEST) as f: return json.load(f)

def main():
    # Wait loop: keep going until download log says ALL DOWNLOADS DONE AND all processed
    done=set()
    deadline=time.time()+3*3600
    while time.time()<deadline:
        man=load_manifest()
        all_dl_done=os.path.exists(f"{ROOT}/logs/zoo_download.log") and \
            any("ALL DOWNLOADS DONE" in l for l in open(f"{ROOT}/logs/zoo_download.log"))
        for m in man:
            name=m["name"]
            if name in done: continue
            if already_done(name):
                log(f"SKIP {name} (already probed)"); done.add(name); continue
            # ready only if fully downloaded
            if not (m.get("ok") and os.path.exists(m["local"]) and
                    m.get("expected") and os.path.getsize(m["local"])==m["expected"]):
                continue
            log(f"=== PROCESS {name} ({m['family']}, {m['params_b']}B) ===")
            if not imported(name):
                if not import_model(name):
                    log(f"  -> import failed, skip for now"); continue
            ok=probe_model(name)
            unload(name)
            if ok and already_done(name):
                done.add(name); log(f"  -> DONE {name} ({len(done)}/{len(man)})")
        if all_dl_done and man and len(done)>=sum(1 for m in man if m.get("ok")):
            log("ALL MODELS PROCESSED"); break
        time.sleep(20)
    log(f"orchestrator exit: {len(done)} models done")

if __name__=="__main__":
    main()
