#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import joblib
import requests
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from datetime import datetime

# --------- CONFIG -------------
MODEL_NAME = "microsoft/codebert-base"
MODEL_FILE = "model/iforest_model.pkl"
SCALER_FILE = "model/commit_scaler.pkl"
LOG_FILE = "anomaly_log.txt"
MAX_TOKENS = 512
# ------------------------------

# Clear old logs at start
open(LOG_FILE, "w").close()

def load_event():
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.exists(event_path):
        with open(event_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_commit_shas(event):
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    if event_name == "push" and "commits" in event:
        return [c.get("id") for c in event.get("commits", []) if c.get("id")]
    elif event_name == "pull_request" and "pull_request" in event:
        base_ref = event["pull_request"]["base"]["ref"]
        try:
            subprocess.run(["git", "fetch", "origin", base_ref], check=False)
            out = subprocess.check_output(["git", "rev-list", f"origin/{base_ref}..HEAD"])
            shas = out.decode().strip().splitlines()
            return shas if shas else [os.environ.get("GITHUB_SHA")]
        except Exception:
            return [os.environ.get("GITHUB_SHA")]
    else:
        sha = os.environ.get("GITHUB_SHA")
        return [sha] if sha else []

def git_show_diff(sha):
    try:
        out = subprocess.check_output(
            ["git", "show", "--pretty=format:%B", "--no-color", sha],
            stderr=subprocess.DEVNULL
        )
        return out.decode(errors="ignore")
    except Exception:
        return ""

# Load model + tokenizer
print("Loading tokenizer/model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def embed_text_truncate(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
    return emb.cpu().numpy()

def embed_text_chunked(text):
    ids = tokenizer.encode(text, add_special_tokens=True)
    chunks = [ids[i:i+MAX_TOKENS] for i in range(0, len(ids), MAX_TOKENS)]
    vecs = []
    for chunk in chunks:
        input_ids = torch.tensor([chunk], dtype=torch.long).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            vec = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            vecs.append(vec)
    if not vecs:
        return np.zeros((1, model.config.hidden_size))
    return np.mean(np.vstack(vecs), axis=0, keepdims=True)

def get_embedding_for_text(text):
    toks = tokenizer.tokenize(text)
    if len(toks) <= MAX_TOKENS:
        return embed_text_truncate(text)
    return embed_text_chunked(text)

# Load anomaly model
if not os.path.exists(MODEL_FILE):
    print(f"ERROR: model file not found at {MODEL_FILE}", file=sys.stderr)
    sys.exit(2)

clf = joblib.load(MODEL_FILE)
print("Anomaly model loaded.")

scaler = joblib.load(SCALER_FILE) if os.path.exists(SCALER_FILE) else None
if scaler is not None:
    print("Scaler loaded.")

# Process commits
event = load_event()
shas = get_commit_shas(event)
repo = os.environ.get("GITHUB_REPOSITORY", "")
token = os.environ.get("GITHUB_TOKEN", "")
event_name = os.environ.get("GITHUB_EVENT_NAME", "")

if not shas:
    print("No commits found. Exiting.")
    sys.exit(0)

found_anomaly = False
for sha in shas:
    if not sha:
        continue
    print(f"Processing commit {sha} ...")
    diff = git_show_diff(sha)[:20000]
    if not diff.strip():
        continue

    embedding = get_embedding_for_text(diff)
    X = embedding
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception:
            X = np.asarray(X).reshape(1, -1)

    try:
        pred = clf.predict(X)[0]
        score = float(clf.decision_function(X)[0]) if hasattr(clf, "decision_function") else None
    except Exception:
        X = np.asarray(X).reshape(1, -1)
        pred = clf.predict(X)[0]
        score = float(clf.decision_function(X)[0]) if hasattr(clf, "decision_function") else None

    is_anom = (pred == -1)

    log_line = {
        "time": datetime.utcnow().isoformat()+"Z",
        "commit": sha,
        "score": score,
        "anomaly": bool(is_anom)
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_line) + "\n")

    if is_anom:
        found_anomaly = True
        body = (
            f"⚠️ **Anomalous commit detected**\n\n"
            f"- **Commit**: `{sha}`\n"
            f"- **Score**: `{score}`\n\n"
            f"Please review this commit.\n\n"
            f"---\n\n"
            f"```diff\n{diff}\n```\n"
        )

        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"} if token else {}
        try:
            if event_name == "pull_request" and "pull_request" in event:
                pr_number = event["pull_request"]["number"]
                url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
                requests.post(url, json={"body": body}, headers=headers, timeout=15)
            else:
                title = f"[security] anomalous commit detected: {sha}"
                url = f"https://api.github.com/repos/{repo}/issues"
                requests.post(url, json={"title": title, "body": body}, headers=headers, timeout=15)
        except Exception as ex:
            print("Warning: failed to send GitHub notification:", ex)

# Fail workflow if anomalies found
if found_anomaly:
    print("One or more anomalous commits detected. See anomaly_log.txt.")
    sys.exit(1)

print("No anomalies found.")
sys.exit(0)
