import itertools, json, re, requests, time, logging, os
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:14b-instruct"
STATE_FILE = "world_state.json"
RUN_FILE = "run_checkpoint.json"
LOG_DIR = "log"
VISUALIZE_DIR = "visualize"
VISUALIZE_EVERY = 50

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
5. "jump_domain": short domain tag for the jump sentence (e.g. biology, chemistry, astronomy)

Format:
{"entities":{"name":{"x":0.0,"y":0.0,"z":0.0}},"relations":[{"f":"from","t":"to","r":"relation_type"}],"next":"...","jump":"...","jump_domain":"..."}

Rules: physical concepts low Z, abstract high Z. Cause/effect close together. Opposites far apart. Both must be short declarative factual statements, not questions. Next must introduce at least one concept not present in current entities. Jump must be a factual statement from a domain not in domains_covered. Jump must never use entities already seen in conversation history. Next and jump must not restate any banned sentence. Avoid sentences using "largest", "tallest", "most famous". Prefer causal, mechanistic, or relational statements. Output only entities and relations for this sentence (new or updated), not the full state. Output raw JSON only, no markdown, no fences, no explanation."""

SYSTEM_REFINE = """You are a semantic geometry engine. Given the current world state and an entity name, output ONLY valid JSON with the corrected position for that entity: {"entities":{"name":{"x":0.0,"y":0.0,"z":0.0}}}. Rules: physical/concrete low Z, abstract high Z. Cause/effect and related concepts close together. Output raw JSON only, no markdown, no fences."""

REFINE_EVERY = 5
DIRTY_RADIUS = 0.3

def call_ollama(prompt, num_predict=2048):
    r = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": num_predict}
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

BANNED_SENTENCES_SIZE = 20

def build_prompt(state, sentence, banned_sentences=None):
    seen = list(state.get("entities", {}).keys())
    domains = state.get("domains_covered", ["physics", "geography"])
    banned = (banned_sentences or [])[-BANNED_SENTENCES_SIZE:]
    payload = {"entities": state.get("entities", {}), "relations": state.get("relations", [])}
    state_str = json.dumps(payload, separators=(',', ':'))
    if len(state_str) > 2000:
        payload = {"entities": state.get("entities", {})}
        state_str = json.dumps(payload, separators=(',', ':'))
    return f"seen_entities:{json.dumps(seen)}\ndomains_covered:{json.dumps(domains)}\nbanned_sentences:{json.dumps(banned)}\nSTATE:{state_str}\nSENTENCE:{sentence}\nOUTPUT:"

def load_state():
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except Exception:
        return {"entities": {}, "relations": []}
    if "dirty" in state and isinstance(state["dirty"], list):
        state["dirty"] = set(state["dirty"])
    state.setdefault("dirty", set())
    state.setdefault("domains_covered", ["physics", "geography"])
    return state

def save_state(state):
    to_save = dict(state)
    if "dirty" in to_save and isinstance(to_save["dirty"], set):
        to_save["dirty"] = list(to_save["dirty"])
    with open(STATE_FILE, "w") as f:
        json.dump(to_save, f, indent=2)

def mark_dirty(state, new_entity, radius=DIRTY_RADIUS):
    pos = state["entities"].get(new_entity)
    if not pos:
        return
    state.setdefault("dirty", set())
    for name, epos in state["entities"].items():
        if name == new_entity:
            continue
        dist = ((pos["x"] - epos["x"]) ** 2 + (pos["y"] - epos["y"]) ** 2 + (pos["z"] - epos["z"]) ** 2) ** 0.5
        if dist < radius:
            state["dirty"].add(name)

def load_checkpoint():
    try:
        with open(RUN_FILE) as f:
            c = json.load(f)
        return c.get("sentence"), c.get("turn", 1), c.get("banned_sentences", [])
    except Exception:
        return None, None, []

def save_checkpoint(sentence, turn, banned_sentences=None):
    with open(RUN_FILE, "w") as f:
        json.dump({"sentence": sentence, "turn": turn, "banned_sentences": banned_sentences or []}, f)

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
    state.setdefault("domains_covered", ["physics", "geography"])
    domain = (result.get("jump_domain") or "").strip().lower()
    if domain and domain not in state["domains_covered"]:
        state["domains_covered"].append(domain)
    return state

def build_refinement_prompt(state, entity):
    payload = {"entities": state.get("entities", {}), "relations": state.get("relations", [])}
    state_str = json.dumps(payload, separators=(",", ":"))
    if len(state_str) > 2000:
        state_str = json.dumps({"entities": state.get("entities", {})}, separators=(",", ":"))
    return f"STATE:{state_str}\nENTITY_TO_REPOSITION:{entity}\nOutput corrected position only (raw JSON):"

def run(iterations=None, sentence="The ball falls because gravity pulls it down."):
    os.makedirs(LOG_DIR, exist_ok=True)
    run_ts = int(time.time())
    log_path = os.path.join(LOG_DIR, f"{run_ts}.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)

    try:
        from visualize import render_to_dir
    except Exception:
        render_to_dir = None

    state = load_state()
    fallback = "Water flows downhill."
    start_turn = 1
    sentence_turn = 0
    banned_sentences = []
    checkpoint_sentence, checkpoint_turn, checkpoint_banned = load_checkpoint()
    if checkpoint_sentence is not None and checkpoint_turn is not None and (iterations is None or checkpoint_turn <= iterations):
        sentence = checkpoint_sentence
        start_turn = checkpoint_turn
        banned_sentences = checkpoint_banned or []
        print(f"Resuming from turn {start_turn} with sentence: {sentence}")

    turn_range = range(start_turn, iterations + 1) if iterations is not None else itertools.count(start_turn)
    try:
        for turn in turn_range:
            do_refine = state.get("dirty") and (turn % REFINE_EVERY == 0)
            if do_refine:
                entity = state["dirty"].pop()
                print(f"\n[{turn}] refine {entity}")
                prompt = build_refinement_prompt(state, entity)
                try:
                    raw = call_ollama(SYSTEM_REFINE + "\n" + prompt)
                    result = extract_json(raw)
                    state = merge(state, result)
                    save_state(state)
                    pos = result.get("entities", {}).get(entity, {})
                    pos_str = f"({pos.get('x',0):.2f},{pos.get('y',0):.2f},{pos.get('z',0):.2f})" if pos else "?"
                    print(f"    repositioned: {entity} {pos_str}")
                    logging.info(
                        "turn=%d refine entity=%r new_position=%s",
                        turn, entity, pos_str,
                    )
                    save_checkpoint(sentence, turn + 1, banned_sentences)
                except Exception as e:
                    logging.exception("turn=%d refine failed: %s", turn, e)
                    try:
                        print(f"    ERROR: {e}\n    raw: {raw[:200]}")
                    except NameError:
                        print(f"    ERROR: {e}")
                    state["dirty"].add(entity)
                    time.sleep(1)
            else:
                use_jump = (sentence_turn + 1) % 2 == 0
                print(f"\n[{turn}] {'jump' if use_jump else 'next'} {sentence}")
                prompt = build_prompt(state, sentence, banned_sentences)
                try:
                    raw = call_ollama(SYSTEM + "\n" + prompt)
                    result = extract_json(raw)
                    old_entities = set(state["entities"].keys())
                    state = merge(state, result)
                    for name in result.get("entities", {}):
                        if name not in old_entities:
                            mark_dirty(state, name, DIRTY_RADIUS)
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
                    for s in (sentence, result.get("next"), result.get("jump")):
                        if s and s not in banned_sentences:
                            banned_sentences.append(s)
                    banned_sentences = banned_sentences[-BANNED_SENTENCES_SIZE:]
                    sentence = next_sentence
                    sentence_turn += 1
                    save_checkpoint(sentence, turn + 1, banned_sentences)
                except Exception as e:
                    logging.exception("turn=%d failed: %s", turn, e)
                    try:
                        print(f"    ERROR: {e}\n    raw: {raw[:200]}")
                    except NameError:
                        print(f"    ERROR: {e}")
                    sentence = fallback
                    save_checkpoint(sentence, turn + 1, banned_sentences)
                    time.sleep(1)
            time.sleep(0.5)

            if turn % VISUALIZE_EVERY == 0 and render_to_dir:
                out_dir = os.path.join(VISUALIZE_DIR, str(run_ts))
                try:
                    render_to_dir(out_dir, turn, state)
                    print(f"    saved viz to {out_dir}/turn_{turn}_xy.png, turn_{turn}_3d.png")
                except Exception as ex:
                    logging.warning("visualize failed: %s", ex)
    except KeyboardInterrupt:
        print("\nStopped by user. Checkpoint saved; run again to resume.")
    finally:
        if iterations is not None:
            delete_checkpoint()
            print(f"\nDone. World state has {len(state['entities'])} entities, {len(state['relations'])} relations.")
        print(f"Saved to {STATE_FILE}")

if __name__ == "__main__":
    run()