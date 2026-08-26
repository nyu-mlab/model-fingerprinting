#!/usr/bin/env python3
"""
Hardened trimmed runner for the model-fingerprinting study.

Fixes from the Jun 25 stall:
  * Owns and supervises the Ollama server. If it dies mid-batch, we restart it
    and retry the current model instead of letting the whole run collapse.
  * Each model is isolated: a crash/timeout on one model is logged and the
    runner moves on to the next.
  * Resumable: skips any model whose raw_responses/<name>.jsonl already has the
    full, error-free expected row count.
  * Only operates on models in zoo_manifest_trim.json (verified-good, on disk).
"""
import json, os, subprocess, time, sys, urllib.request, signal

ROOT = os.path.expanduser("~/workspace/model-fingerprinting")
MANIFEST = f"{ROOT}/models/zoo_manifest_trim.json"
GGUF_DIR = f"{ROOT}/models/gguf"
MF_DIR   = f"{ROOT}/models/modelfiles"
RAW_DIR  = f"{ROOT}/raw_responses"
LOG      = f"{ROOT}/logs/run_trimmed.log"
SERVE_LOG= f"{ROOT}/logs/ollama-serve-trim.log"
OLLAMA_BIN = os.path.expanduser("~/.local/bin/ollama")
ENV = dict(os.environ,
           OLLAMA_MODELS=f"{ROOT}/models/ollama_store",
           OLLAMA_HOST="127.0.0.1:11434",
           OLLAMA_KEEP_ALIVE="0",
           OLLAMA_CONTEXT_LENGTH="2048",
           PATH=os.path.expanduser("~/.local/bin")+":"+os.environ.get("PATH",""))
TRIALS = 5
N_PROBES = 16
EXPECTED_ROWS = TRIALS * N_PROBES
OLLAMA = "http://127.0.0.1:11434"

os.makedirs(f"{ROOT}/logs", exist_ok=True)
os.makedirs(MF_DIR, exist_ok=True)

def log(m):
    line=f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG,"a") as f: f.write(line+"\n")

_serve_proc=None

def server_up():
    try:
        with urllib.request.urlopen(OLLAMA+"/api/tags", timeout=5) as r:
            return r.status==200
    except Exception:
        return False

def start_server():
    global _serve_proc
    if server_up(): return True
    log("starting ollama serve")
    sf=open(SERVE_LOG,"a")
    _serve_proc=subprocess.Popen([OLLAMA_BIN,"serve"],env=ENV,
                                 stdout=sf,stderr=subprocess.STDOUT,
                                 start_new_session=True)
    for _ in range(60):
        if server_up(): log("ollama serve is up"); return True
        time.sleep(1)
    log("ERROR: ollama serve did not come up in 60s")
    return False

def ensure_server():
    """Guarantee a live server; restart if needed."""
    for attempt in range(3):
        if server_up(): return True
        log(f"server down -> (re)start attempt {attempt+1}")
        start_server()
        if server_up(): return True
        time.sleep(3)
    return False

def already_done(name):
    p=f"{RAW_DIR}/{name}.jsonl"
    if not os.path.exists(p): return False
    rows=[l for l in open(p) if l.strip()]
    if len(rows)<EXPECTED_ROWS: return False
    return sum(1 for l in rows if json.loads(l).get("error"))==0

def imported(name):
    try:
        out=subprocess.run([OLLAMA_BIN,"list"],env=ENV,capture_output=True,text=True,timeout=30)
        return any(line.split()[0].split(":")[0]==name for line in out.stdout.splitlines() if line.strip())
    except Exception:
        return False

def import_model(name):
    mf=f"{MF_DIR}/{name}.Modelfile"
    with open(mf,"w") as f:
        f.write(f"FROM {GGUF_DIR}/{name}.gguf\n")
    r=subprocess.run([OLLAMA_BIN,"create",name,"-f",mf],env=ENV,
                     capture_output=True,text=True,timeout=600)
    ok = r.returncode==0
    if not ok:
        log(f"  import FAIL {name}: {(r.stdout+r.stderr)[-200:]}")
    return ok

def probe_model(name):
    r=subprocess.run(["python3",f"{ROOT}/scripts/probe.py",name,"--trials",str(TRIALS)],
                     env=ENV,capture_output=True,text=True,timeout=7200)
    log(f"  probe rc={r.returncode}: {(r.stdout.strip()[-160:])} {(r.stderr.strip()[-120:])}")
    return r.returncode==0

def unload(name):
    try:
        subprocess.run([OLLAMA_BIN,"stop",name],env=ENV,capture_output=True,timeout=30)
    except Exception: pass

def process_model(m):
    name=m["name"]
    if already_done(name):
        log(f"SKIP {name} (already fully probed)"); return True
    # up to 2 attempts per model, restarting the server between tries
    for attempt in range(2):
        if not ensure_server():
            log(f"  {name}: no server, deferring"); return False
        log(f"=== PROCESS {name} ({m['family']}, {m['params_b']}B) attempt {attempt+1} ===")
        if not imported(name):
            if not import_model(name):
                log(f"  {name}: import failed"); 
                if not server_up(): continue  # server may have died; retry
                return False
        probe_model(name)
        unload(name)
        if already_done(name):
            log(f"  -> DONE {name}"); return True
        log(f"  {name}: incomplete after attempt {attempt+1} (server_up={server_up()})")
    return already_done(name)

def main():
    man=json.load(open(MANIFEST))
    log(f"trimmed run start: {len(man)} models -> {[m['name'] for m in man]}")
    if not ensure_server():
        log("FATAL: could not start ollama server"); sys.exit(1)
    done=[]
    for m in man:
        ok=process_model(m)
        if ok: done.append(m["name"])
    log(f"trimmed run complete: {len(done)}/{len(man)} done -> {done}")

if __name__=="__main__":
    main()
