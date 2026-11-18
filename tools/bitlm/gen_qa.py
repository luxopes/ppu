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


# --- styl "adult": klasicky asistent (dospely uzivatel, bezne dotazy), stale jednoducha anglictina ---
TOPICS_A = {
    "everyday life": ["sleeping better", "waking up early", "cleaning the house", "doing laundry", "shopping for food",
                      "saving money", "planning a day", "being on time", "moving to a new home", "taking care of plants",
                      "fixing small things at home", "getting ready for winter", "packing for a trip"],
    "food and cooking": ["boiling eggs", "making pancakes", "cooking rice", "making a salad", "baking bread",
                         "making soup", "healthy breakfast", "storing fruit", "making tea", "making coffee",
                         "cooking pasta", "a simple dinner idea", "what to eat after sport"],
    "health": ["drinking enough water", "a cold", "headaches", "walking every day", "stretching", "stress",
               "eating vegetables", "washing hands", "sunburn", "staying healthy in winter", "resting after work"],
    "science": ["why the sky is blue", "how plants grow", "gravity", "magnets", "electricity", "the water cycle",
                "why ice floats", "how rainbows form", "the moon", "the sun", "thunder and lightning", "volcanoes",
                "how the heart works", "why leaves change color", "how sound travels", "what air is made of"],
    "geography": ["the biggest ocean", "continents", "mountains", "deserts", "rivers", "the capital of France",
                  "the capital of Japan", "the capital of Italy", "the North Pole", "rain forests", "islands",
                  "the tallest mountain", "the longest river"],
    "animals": ["dogs", "cats", "elephants", "whales", "bees", "owls", "penguins", "sharks", "horses", "ants",
                "how birds fly", "why cats purr", "what pandas eat", "how fish breathe", "animals in winter"],
    "technology": ["the internet", "email", "a good password", "phones", "computers", "charging a phone",
                   "taking photos", "a slow computer", "what a website is", "staying safe online"],
    "words and language": ["the meaning of a word", "a word that means happy", "the opposite of a word",
                           "spelling a word", "learning a new language", "writing neatly", "reading more books"],
    "math": ["adding two numbers", "counting money", "half of a number", "how many days are in a week",
             "how many minutes are in an hour", "even and odd numbers", "simple shapes", "telling time"],
    "writing help": ["a short birthday message", "a thank you note", "a short poem about the sea",
                     "a short poem about the moon", "a tiny story about a cat", "a funny joke", "a name for a dog",
                     "a short message to a friend", "a good title for a story", "a kind message for a sick friend"],
    "advice": ["making new friends", "being nervous before a test", "a job interview", "learning faster",
               "staying calm", "saying sorry", "being kind to people", "starting a new hobby", "getting motivated",
               "feeling lonely", "talking to a neighbor"],
    "hobbies": ["gardening", "learning to draw", "playing guitar", "running", "swimming", "chess", "reading",
                "taking walks", "riding a bike", "camping", "fishing", "painting"],
    "small talk": ["saying hello", "asking how the assistant is", "asking who the assistant is",
                   "asking what the assistant can do", "saying thank you", "saying good morning",
                   "saying good night", "talking about the weather", "having a good day", "telling a joke"],
}

QTYPES_A = [
    ("fact", "The user asks a simple factual question about {t}."),
    ("how", "The user asks how to do something related to {t}."),
    ("why", "The user asks why something is the way it is, about {t}."),
    ("explain", "The user asks the assistant to explain {t} in simple words."),
    ("advice", "The user asks for advice or tips about {t}."),
    ("request", "The user asks the assistant to write or make something about {t}."),
    ("chat", "The user starts a friendly chat about {t}."),
    ("compare", "The user asks about the difference between two things related to {t}."),
]

SYSTEM_A = ("You write training data for a small language model. It knows only simple, everyday English. "
            "You follow the format exactly and output nothing else.")

PROMPT_A = """Write a short conversation between a user (User) and a helpful assistant (Assistant).
Topic: {topic}.
{qtype}
It has {turns}.

Rules:
- The user writes like a normal adult asking a chat assistant a real, sensible question.
- Use simple, common English words and short sentences. No hard or technical words.
- The Assistant is correct, clear and friendly. Each answer is 1 to 4 short sentences, all on one line.
- Facts must be true. If the user asks something the assistant cannot know, it says so kindly.
- If asked who it is, the Assistant says it is a helpful assistant that can answer questions and help with ideas. Do not mention AI, computers or programs.
- Plain text only: no emojis, no lists, no bullet points, no bold, no headings, no line breaks inside an answer.
- Every line starts with "User: " or "Assistant: ". The User speaks first."""


def make_task_adult(i, words, seed):
    r = random.Random(seed * 1_000_003 + i)
    cat = r.choice(list(TOPICS_A))
    topic = r.choice(TOPICS_A[cat])
    qk, qt = ("chat", "The user greets the assistant or makes small talk about {t}.") if cat == "small talk" \
        else r.choice(QTYPES_A)
    turns = r.choices(["one question and one answer", "two exchanges (the user asks a follow-up question)",
                       "three exchanges (the user asks follow-up questions)"], weights=[4, 4, 2])[0]
    return {"id": i, "topic": topic, "qtype": qk, "turns": turns,
            "prompt": PROMPT_A.format(topic=topic, qtype=qt.format(t=topic), turns=turns)}


# --- styl "adult2": tisice temat (seznam vytvori LLM pres --make-subjects), persona uzivatele, pozdravy ---
CATEGORIES = [
    "computers and software", "the internet and websites", "phones and apps", "operating systems like Linux and Windows",
    "programming basics", "computer parts", "online safety", "gadgets and electronics", "science basics", "physics in daily life",
    "chemistry in the kitchen", "biology and living things", "the human body", "health and medicine basics", "food and nutrition",
    "cooking and recipes", "baking", "drinks", "space and planets", "weather and climate", "earth and nature", "oceans and rivers",
    "plants and gardening", "pets", "wild animals", "insects", "birds", "world geography", "countries and cities", "famous landmarks",
    "history basics", "inventions and inventors", "famous people from history", "money and saving", "shopping", "jobs and work",
    "school and learning", "studying tips", "english words and grammar", "learning languages", "math ideas", "measurements and units",
    "time and calendars", "sports", "games and puzzles", "music and instruments", "art and drawing", "books and reading", "movies and tv",
    "travel and holidays", "cars and driving", "trains, planes and boats", "home and cleaning", "fixing things at home", "tools",
    "clothes and fashion", "sleep and rest", "exercise and fitness", "feelings and emotions", "friends and family", "manners and etiquette",
    "celebrations and holidays", "hobbies", "energy and electricity", "recycling and the environment", "jobs people do", "the city and town",
    "farms and farming", "safety and first aid", "materials like wood, metal and plastic", "colors, shapes and patterns", "music styles",
]

PERSONAS = ["a student", "a parent", "an older person", "a teenager", "a busy office worker", "a curious adult", "a home cook",
            "a traveler", "someone learning English", "a new computer user", "a teacher", "a retired person", "a young adult"]

QTYPES_A2 = [
    ("what", "The user asks what {t} is or what it means."),
    ("how", "The user asks how to do something with {t}, or how {t} works."),
    ("why", "The user asks why something about {t} is true."),
    ("fact", "The user asks a simple factual question about {t}."),
    ("advice", "The user asks for simple advice or tips about {t}."),
    ("compare", "The user asks about the difference between {t} and a similar thing."),
    ("request", "The user asks the assistant to write something short related to {t}, like a short note, a few ideas or a tiny story."),
    ("explain", "The user asks the assistant to explain {t} in very simple words."),
]

SUBJECT_PROMPT = """List 40 different, specific things that people often ask a helpful assistant about in the area: {cat}.
Each item is a short noun phrase of 1 to 5 words (for example: "Linux", "a Wi-Fi router", "why bread rises").
Make them varied, from very common to a bit less common. One item per line, no numbers, no extra text."""


def make_subjects(a):
    key = open(a.key_file).read().strip()
    subs = set()

    def job(k):
        cat = CATEGORIES[k % len(CATEGORIES)]
        text, _ = call(a.url, key, a.model, SUBJECT_PROMPT.format(cat=cat), 1.0, SYSTEM_A)
        return cat, [x.strip(" -*0123456789.\t").strip() for x in text.split("\n")]

    with cf.ThreadPoolExecutor(64) as ex:
        for cat, items in ex.map(job, range(len(CATEGORIES) * 4)):
            for x in items:
                if 2 <= len(x) <= 40 and all(32 <= ord(c) < 127 for c in x):
                    subs.add((cat, x))
    with open(a.out, "w") as f:
        for cat, x in sorted(subs):
            f.write(f"{cat}\t{x}\n")
    print(f"temat {len(subs)}", file=sys.stderr)


def make_task_adult2(i, subjects, seed):
    r = random.Random(seed * 1_000_003 + i)
    cat, topic = r.choice(subjects)
    qk, qt = r.choice(QTYPES_A2)
    turns = r.choices(["one question and one answer", "two exchanges (the user asks a follow-up question)",
                       "three exchanges (the user asks follow-up questions)"], weights=[4, 4, 2])[0]
    extra = f"The user is {r.choice(PERSONAS)}."
    if r.random() < 0.2:
        extra += ' The user starts with a short greeting (like "Hi!" or "Hello,") before the question, on the same line.'
    prompt = PROMPT_A.format(topic=f"{topic} ({cat})", qtype=qt.format(t=topic) + "\n" + extra, turns=turns)
    return {"id": i, "topic": topic, "cat": cat, "qtype": qk, "turns": turns, "prompt": prompt}


def call(url, key, model, prompt, temp, system=SYSTEM):
    body = json.dumps({"model": model, "temperature": temp, "top_p": 0.95, "max_tokens": 450,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
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
    words = [w.strip() for w in open(a.words) if w.strip()] if a.words else []
    if a.style == "adult2":
        words = [tuple(w.split("\t", 1)) for w in words]
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
        if a.style == "adult2":
            t = make_task_adult2(i, words, a.seed)
            text, ntok = call(a.url, key, a.model, t["prompt"], a.temp, SYSTEM_A)
        elif a.style == "adult":
            t = make_task_adult(i, words, a.seed)
            text, ntok = call(a.url, key, a.model, t["prompt"], a.temp, SYSTEM_A)
        else:
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
        if seen[q] > a.max_dup:
            rej["dup"] += 1
            continue
        keep.append("\n".join(lines))
    random.Random(0).shuffle(keep)
    va, tr = keep[:a.val], keep[a.val:]
    base = a.out[:-4] if a.out.endswith(".txt") else a.out
    for path, part in ((a.out, tr), (base + "_val.txt", va)):
        with open(path, "w") as f:
            for k in part:
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
    ap.add_argument("--make-subjects", action="store_true", help="vytvorit seznam temat (kategorie\ttema) do --out")
    ap.add_argument("--style", default="child", choices=["child", "adult", "adult2"])
    ap.add_argument("--max-dup", type=int, default=5, help="max. rozhovoru se stejnou prvni otazkou")
    ap.add_argument("--val", type=int, default=1000, help="pocet rozhovoru do <out>_val (posledni)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.make_subjects:
        make_subjects(a)
    elif a.clean:
        clean(a)
    else:
        generate(a)


if __name__ == "__main__":
    main()
