#!/usr/bin/env python3
"""Synteticky dataset otazek a odpovedi pro maly model (styl TinyStories): rozhovory User/Assistant
jednoduchou anglictinou, generovane vetsim modelem pres OpenAI-kompatibilni API (vLLM).

Rozmanitost jako u TinyStories: kazde zadani kombinuje nahodne tema, typ otazky, slovo ze slovniku
TinyStories, delku a naladu. Vystup JSONL (id, zadani, text); beh lze prerusit a navazat (hotova id se preskoci).

Pouziti: gen_qa.py --url https://<pod>-8000.proxy.runpod.net --key-file KEY --words words.txt \
                   --n 120000 --out qa.jsonl [--workers 128]
         gen_qa.py --clean qa.jsonl --out qa.txt      (filtr a prevod na text "User: ...\\nAssistant: ...")
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request

TOPICS = {
    "animals": ["dogs", "cats", "birds", "fish", "bees", "ants", "cows", "horses", "pigs", "sheep", "ducks", "frogs",
                "lions", "elephants", "giraffes", "monkeys", "bears", "owls", "rabbits", "snakes", "turtles", "whales",
                "dolphins", "sharks", "butterflies", "spiders", "penguins", "zebras", "chickens", "squirrels", "worms",
                "bats", "kangaroos", "camels", "octopuses", "crabs", "snails", "ladybugs", "foxes", "wolves"],
    "body": ["eyes", "ears", "teeth", "hair", "hands", "feet", "the heart", "the nose", "bones", "skin", "sleep",
             "yawning", "sneezing", "hiccups", "growing up", "being sick", "washing hands", "brushing teeth"],
    "nature": ["the sun", "the moon", "stars", "rain", "snow", "wind", "clouds", "rainbows", "thunder", "lightning",
               "trees", "flowers", "leaves", "seeds", "grass", "rivers", "the ocean", "mountains", "rocks", "sand",
               "mud", "fire", "ice", "puddles", "the sky", "night and day", "seasons", "spring", "summer", "autumn",
               "winter", "volcanoes", "caves", "forests", "the desert", "islands"],
    "space": ["planets", "the earth", "rockets", "astronauts", "the stars at night", "the moon's shape"],
    "food": ["apples", "bananas", "carrots", "bread", "milk", "eggs", "cheese", "honey", "cookies", "soup",
             "vegetables", "fruit", "water", "breakfast", "cooking", "pizza", "ice cream", "juice", "rice", "candy"],
    "home": ["the kitchen", "beds", "doors", "windows", "toys", "books", "clocks", "lights", "the bathtub", "chairs",
             "cleaning up", "the garden", "a fridge", "a phone", "a computer", "the TV", "stairs", "keys"],
    "things that go": ["cars", "trains", "boats", "planes", "bikes", "buses", "trucks", "fire trucks", "tractors",
                       "wheels", "roads", "traffic lights", "bridges"],
    "people and jobs": ["doctors", "teachers", "farmers", "firefighters", "police officers", "bakers", "builders",
                        "nurses", "mail carriers", "pilots", "vets", "grandparents", "babies", "friends", "neighbors"],
    "feelings": ["being happy", "being sad", "being angry", "being scared", "being shy", "being lonely",
                 "being excited", "being bored", "being proud", "being jealous", "being nervous", "missing someone"],
    "being good": ["sharing", "saying sorry", "saying thank you", "being kind", "telling the truth", "helping others",
                   "taking turns", "being patient", "listening", "being brave", "making friends", "being polite"],
    "safety": ["crossing the street", "hot stoves", "strangers", "swimming safely", "wearing a helmet",
               "getting lost", "sharp things", "staying warm"],
    "learning": ["counting", "colors", "shapes", "letters", "reading", "drawing", "music", "singing", "dancing",
                 "big and small", "days of the week", "telling time", "left and right", "opposites"],
    "play": ["playing outside", "the park", "games", "balls", "kites", "swings", "building blocks", "puzzles",
             "hide and seek", "the beach", "birthdays", "holidays", "pets", "camping", "a picnic"],
}

QTYPES = [
    ("why", "The child asks a 'why' question about {t}."),
    ("how", "The child asks a 'how' question about {t} (how something works or how to do something)."),
    ("what", "The child asks 'what is' or 'what does' about {t}."),
    ("where", "The child asks a 'where' question about {t}."),
    ("can", "The child asks whether something about {t} is possible ('Can ...?' or 'Do ...?')."),
    ("fact", "The child asks the assistant to tell a fun, true fact about {t}."),
    ("advice", "The child has a small problem or worry related to {t} and asks what to do."),
    ("story", "The child asks for a very short story about {t}. The assistant tells a story of 4 to 6 simple sentences."),
    ("explain", "The child says they do not understand something about {t} and asks for a simple explanation."),
    ("feeling", "The child tells the assistant how they feel about {t}, and the assistant answers kindly and asks or explains something."),
]

MOODS = ["curious", "excited", "a little worried", "happy", "puzzled", "sleepy", "playful", "shy"]

SYSTEM = ("You write training data for a tiny language model that only knows the words a 4 to 5 year old child "
          "understands. You follow the format exactly and output nothing else.")

PROMPT = """Write a short conversation between a {mood} young child (User) and a kind, helpful assistant (Assistant).
Topic: {topic}.
{qtype}
{wordline}It has {turns}.

Rules:
- The child asks something real young children often wonder about; the question must make sense.
- Use only very simple words that a 4 or 5 year old knows. Short sentences.
- The Assistant answers truthfully and clearly, in 1 to 3 short sentences (a story may be longer), and is warm and friendly.
- No names for the Assistant. Do not mention being an AI, a computer or a program.
- Plain text only: no emojis, no lists, no bold, no headings.
- Every line starts with "User: " or "Assistant: ". The User speaks first."""


def make_task(i, words, seed):
    r = random.Random(seed * 1_000_003 + i)
    cat = r.choice(list(TOPICS))
    topic = r.choice(TOPICS[cat])
    qk, qt = r.choice(QTYPES)
    turns = r.choices(["one question and one answer", "two exchanges (the child asks a follow-up question)",
                       "three exchanges (the child asks follow-up questions)"], weights=[3, 4, 2])[0]
    word, mood = r.choice(words), r.choice(MOODS)
    use_word = r.random() < 0.3
    word = word if use_word else ""
    return {"id": i, "topic": topic, "qtype": qk, "word": word, "mood": mood, "turns": turns,
            "prompt": PROMPT.format(mood=mood, topic=topic, qtype=qt.format(t=topic), turns=turns,
                                    wordline=(f'If it fits naturally, use the word "{word}".\n' if use_word else ""))}


def call(url, key, model, prompt, temp):
    body = json.dumps({"model": model, "temperature": temp, "top_p": 0.95, "max_tokens": 450,
                       "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
    req = urllib.request.Request(url + "/v1/chat/completions", data=body.encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key,
                                          "User-Agent": "ppu-gen-qa/1.0"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as f:
                d = json.load(f)
            return d["choices"][0]["message"]["content"], d["usage"]["completion_tokens"]
        except Exception as e:  # noqa: BLE001  (sit/proxy: zkusit znovu)
            err = e
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"pozadavek selhal: {err}")


def generate(a):
    key = open(a.key_file).read().strip()
    words = [w.strip() for w in open(a.words) if w.strip()]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["id"])
            except (ValueError, KeyError):
                pass
    todo = [i for i in range(a.n) if i not in done]
    print(f"hotovo {len(done)}, zbyva {len(todo)}", file=sys.stderr, flush=True)
    lock = threading.Lock()
    out = open(a.out, "a")
    stats = {"n": 0, "tok": 0, "t0": time.time()}

    def job(i):
        t = make_task(i, words, a.seed)
        text, ntok = call(a.url, key, a.model, t["prompt"], a.temp)
        rec = {k: v for k, v in t.items() if k != "prompt"}
        rec["text"] = text
        with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["n"] += 1
            stats["tok"] += ntok
            if stats["n"] % 500 == 0:
                out.flush()
                dt = time.time() - stats["t0"]
                print(f"{stats['n']}/{len(todo)}  {stats['tok'] / dt:.0f} tok/s  {stats['n'] / dt * 3600:.0f}/h",
                      file=sys.stderr, flush=True)

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for f in cf.as_completed([ex.submit(job, i) for i in todo]):
            if f.exception():
                print("chyba:", f.exception(), file=sys.stderr, flush=True)
    out.close()


LINE = re.compile(r"^(User|Assistant): (\S.*)$")
BAD = re.compile(r"\b(AI|artificial|language model|chatbot|computer program|I'm just|as an assistant)\b", re.I)


def clean(a):
    """Prevod na text pro prepare: kontrola formatu, ASCII, delek a odmitnuti; duplicitni prvni otazky omezene."""
    keep, seen = [], {}
    rej = {"format": 0, "ascii": 0, "bad": 0, "dup": 0}
    for line in open(a.clean):
        rec = json.loads(line)
        t = rec["text"].replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        t = t.replace("—", " - ").replace("–", "-").replace("…", "...").strip()
        lines = [x.strip() for x in t.split("\n") if x.strip()]
        if not lines or not all(LINE.match(x) for x in lines) or not lines[0].startswith("User: ") \
                or not lines[-1].startswith("Assistant: ") or len(lines) > 8:
            rej["format"] += 1
            continue
        if any(ord(c) > 126 or ord(c) < 32 for c in "".join(lines)):
            rej["ascii"] += 1
            continue
        if BAD.search(t) or any(len(x) > 700 for x in lines):
            rej["bad"] += 1
            continue
        q = lines[0].lower()
        seen[q] = seen.get(q, 0) + 1
        if seen[q] > 3:
            rej["dup"] += 1
            continue
        keep.append("\n".join(lines))
    with open(a.out, "w") as f:
        for k in keep:
            f.write(k + "\n<|endoftext|>\n")
    print(f"ponechano {len(keep)}, vyrazeno {rej}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--key-file")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--words")
    ap.add_argument("--n", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--clean")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    clean(a) if a.clean else generate(a)


if __name__ == "__main__":
    main()
