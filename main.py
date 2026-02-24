import json, re, requests, time, logging, os
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:14b-instruct"
STATE_FILE = "world_state.json"
RUN_FILE = "run_checkpoint.json"
LOG_DIR = "log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)

SYSTEM = """You are a semantic geometry engine. Given a world state (entity positions in XYZ idea space) and a sentence, output ONLY valid JSON with:
1. Updated/new entity positions (x,y,z 0-1, relative to existing entities)
2. Relations between entities
3. "next": declarative statement continuing current theme
4. "jump": declarative statement from a completely different domain

Format:
{"entities":{"name":{"x":0.0,"y":0.0,"z":0.0}},"relations":[{"f":"from","t":"to","r":"relation_type"}],"next":"...","jump":"..."}

Rules: physical concepts low Z, abstract high Z. Cause/effect close together. Opposites far apart. Both must be short declarative factual statements, not questions. Next must introduce at least one concept not present in current entities. Jump must be a factual statement from a domain unrelated to current entities. Output raw JSON only, no markdown, no fences, no explanation."""

def call_ollama(prompt):
    r = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 800}
    })
    return r.json()["response"].strip()

def extract_json(text):
    text = re.sub(r"```\w*\n?", "", text)
    text = text.replace("```", "")
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    chunk = text[start:end].strip()
    lines = chunk.split("\n")
    while lines and lines[0].strip().startswith("`"):
        lines.pop(0)
    while lines and lines[-1].strip().startswith("`"):
        lines.pop()
    return json.loads("\n".join(lines))

def build_prompt(state, sentence):
    # Keep state compact - only positions, truncate if huge
    state_str = json.dumps(state, separators=(',', ':'))
    if len(state_str) > 2000:
        # Keep only entities, drop relations from state to save context
        compact = {"entities": state.get("entities", {})}
        state_str = json.dumps(compact, separators=(',', ':'))
    return f"STATE:{state_str}\nSENTENCE:{sentence}\nOUTPUT:"

def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except:
        return {"entities": {}, "relations": []}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def load_checkpoint():
    try:
        with open(RUN_FILE) as f:
            c = json.load(f)
        return c.get("sentence"), c.get("turn", 1)
    except Exception:
        return None, None

def save_checkpoint(sentence, turn):
    with open(RUN_FILE, "w") as f:
        json.dump({"sentence": sentence, "turn": turn}, f)

def delete_checkpoint():
    try:
        os.remove(RUN_FILE)
    except Exception:
        pass

def merge(state, result):
    for name, pos in result.get("entities", {}).items():
        state["entities"][name] = pos
    state.setdefault("relations", [])
    for rel in result.get("relations", []):
        if rel not in state["relations"]:
            state["relations"].append(rel)
    return state

def run(iterations=500, sentence="The ball falls because gravity pulls it down."):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{int(time.time())}.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)

    state = load_state()
    fallback = "Water flows downhill."
    start_turn = 1
    checkpoint_sentence, checkpoint_turn = load_checkpoint()
    if checkpoint_sentence is not None and checkpoint_turn is not None and checkpoint_turn <= iterations:
        sentence = checkpoint_sentence
        start_turn = checkpoint_turn
        print(f"Resuming from turn {start_turn} with sentence: {sentence}")

    for turn in range(start_turn, iterations + 1):
        use_jump = turn % 2 == 0
        print(f"\n[{turn}] {'jump' if use_jump else 'next'} {sentence}")
        prompt = build_prompt(state, sentence)
        try:
            raw = call_ollama(SYSTEM + "\n" + prompt)
            result = extract_json(raw)
            state = merge(state, result)
            save_state(state)
            next_sentence = result.get("jump", result.get("next", fallback)) if use_jump else result.get("next", result.get("jump", fallback))
            entities = result.get("entities", {})
            relations = result.get("relations", [])
            entity_str = " ".join(f"{n}=({e['x']:.2f},{e['y']:.2f},{e['z']:.2f})" for n, e in entities.items())
            rel_str = " ".join(f"{r.get('f','')}->{r.get('t','')}:{r.get('r','')}" for r in relations)
            print(f"    entities: {list(entities.keys())}")
            print(f"    relations: {len(relations)}")
            logging.info(
                "turn=%d %s sentence=%r | committed entities: %s | committed relations: %s | next_sentence=%r",
                turn, "jump" if use_jump else "next", sentence, entity_str, rel_str, next_sentence,
            )
            sentence = next_sentence
            save_checkpoint(sentence, turn + 1)
        except Exception as e:
            logging.exception("turn=%d failed: %s", turn, e)
            print(f"    ERROR: {e}\n    raw: {raw[:200]}")
            time.sleep(1)
        time.sleep(0.5)

    delete_checkpoint()
    print(f"\nDone. World state has {len(state['entities'])} entities, {len(state['relations'])} relations.")
    print(f"Saved to {STATE_FILE}")

if __name__ == "__main__":
    run()