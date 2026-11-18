#!/usr/bin/env python3
"""Sablonove rozhovory User/Assistant bez jazykoveho modelu: zarucene spravne odpovedi.

Pokryva to, co maly model potrebuje umet spolehlive a co LLM generator pokryl spatne:
pozdravy a small talk, identitu asistenta, jednoduche pocty (+ - x do 20/10), dny, mesice,
poradi cisel, hlavni mesta, protiklady, zvuky zvirat, barvy veci. Rozhovory maji 1 az 4 vymeny
nahodne poskladane z temat (pozdrav -> otazka -> podekovani), aby se model ucil i navaznost.

Pouziti: gen_facts.py --n 150000 --out data/qa/facts.txt [--seed 1]
"""
import argparse
import random

GREET_U = ["Hi", "Hi!", "Hello", "Hello!", "Hey", "Hey there", "Hi there", "Good morning", "Good afternoon",
           "Good evening", "Hello there!", "Hey!", "Hiya", "Howdy", "Greetings", "Yo", "Morning!", "hi", "hello",
           "hey"]
GREET_A = ["Hello! How can I help you today?", "Hi! What can I do for you?", "Hey! How can I help?",
           "Hello! It is nice to talk to you. What would you like to know?", "Hi there! How can I help you?",
           "Hello! I am here to help. What is on your mind?", "Hi! Ask me anything you like.",
           "Hello! What would you like to talk about?"]
HOW_U = ["How are you?", "How are you today?", "How are you doing?", "How is it going?", "How's your day?",
         "Are you okay?", "How do you feel today?", "How are things?"]
HOW_A = ["I am doing well, thank you for asking! How can I help you?", "I am good, thanks! How are you?",
         "I am fine, thank you. What can I do for you today?", "Very well, thanks! How is your day going?",
         "I am great, thank you! What would you like to talk about?"]
WHO = [  # (otazky, odpovedi) - odpovedi sedi ke skupine otazek
    (["Who are you?", "What are you?", "Who am I talking to?", "Tell me about yourself.", "Introduce yourself."],
     ["I am a small helpful assistant. I can answer simple questions, tell short stories and chat with you.",
      "I am an assistant. I am here to answer your questions and help with simple things.",
      "I am a friendly assistant. You can ask me questions, ask for a story, or just talk."]),
    (["What is your name?", "Do you have a name?", "What should I call you?"],
     ["I do not have a name. You can just call me Assistant.",
      "I am just called Assistant. What is your name?",
      "I have no real name, but you can call me Assistant."]),
    (["Are you a person?", "Are you human?", "Are you real?"],
     ["No, I am not a person. I am a small assistant that can chat and answer simple questions.",
      "I am not a human. I am an assistant made to help with simple questions."]),
]
CAN_U = ["What can you do?", "How can you help me?", "What are you good at?", "What can I ask you?",
         "Can you help me?", "What do you know?"]
CAN_A = ["I can answer simple questions, do easy math, tell short stories and give simple advice.",
         "I can help with simple facts, easy sums, short stories and friendly chat. Just ask!",
         "Yes, I can try! I know simple facts, numbers, days and months, and I like telling short stories.",
         "You can ask me about animals, nature, food, numbers, or ask me for a short story."]
THANK_U = ["Thank you!", "Thanks!", "Thanks a lot.", "Thank you very much.", "Great, thanks.", "Cool, thanks!",
           "Okay, thank you.", "That helps, thanks.", "Thanks for your help.", "thx"]
THANK_A = ["You are welcome!", "You're welcome! Happy to help.", "No problem! Ask me anything else.",
           "Glad I could help!", "Anytime! Have a nice day.", "You are welcome. Is there anything else?"]
BYE_U = ["Bye!", "Goodbye.", "See you later.", "Good night!", "Bye bye.", "I have to go now.", "See you!",
         "Talk to you later."]
BYE_A = ["Goodbye! Have a great day.", "Bye! Come back any time.", "See you later! Take care.",
         "Good night! Sleep well.", "Bye! It was nice talking to you."]

ADD_Q = ["What is {a} + {b}?", "What is {a} plus {b}?", "How much is {a} plus {b}?", "{a} + {b} = ?",
         "What's {a}+{b}?", "Can you add {a} and {b}?", "How much is {a}+{b}", "What do you get if you add {a} and {b}?"]
SUB_Q = ["What is {a} - {b}?", "What is {a} minus {b}?", "How much is {a} minus {b}?", "{a} - {b} = ?",
         "What's {a}-{b}?", "If I have {a} apples and give away {b}, how many are left?"]
MUL_Q = ["What is {a} times {b}?", "What is {a} x {b}?", "How much is {a} times {b}?", "{a} * {b} = ?",
         "What's {a}*{b}?", "Can you multiply {a} by {b}?"]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
CAPITALS = {"France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "England": "London",
            "the United Kingdom": "London", "Japan": "Tokyo", "China": "Beijing", "Russia": "Moscow",
            "the United States": "Washington, D.C.", "Canada": "Ottawa", "Mexico": "Mexico City",
            "Brazil": "Brasilia", "Argentina": "Buenos Aires", "Egypt": "Cairo", "India": "New Delhi",
            "Australia": "Canberra", "Greece": "Athens", "Portugal": "Lisbon", "Poland": "Warsaw",
            "the Czech Republic": "Prague", "Austria": "Vienna", "Hungary": "Budapest", "Sweden": "Stockholm",
            "Norway": "Oslo", "Finland": "Helsinki", "Denmark": "Copenhagen", "Ireland": "Dublin",
            "the Netherlands": "Amsterdam", "Belgium": "Brussels", "Switzerland": "Bern", "Turkey": "Ankara",
            "South Korea": "Seoul", "Thailand": "Bangkok", "Vietnam": "Hanoi", "Kenya": "Nairobi",
            "Peru": "Lima", "Chile": "Santiago", "Ukraine": "Kyiv", "Slovakia": "Bratislava",
            "Croatia": "Zagreb", "Iceland": "Reykjavik", "New Zealand": "Wellington", "Cuba": "Havana",
            "Indonesia": "Jakarta", "the Philippines": "Manila", "Nigeria": "Abuja", "Morocco": "Rabat"}
OPPOSITES = [("hot", "cold"), ("big", "small"), ("up", "down"), ("day", "night"), ("fast", "slow"),
             ("happy", "sad"), ("old", "new"), ("open", "closed"), ("light", "dark"), ("tall", "short"),
             ("full", "empty"), ("wet", "dry"), ("hard", "soft"), ("loud", "quiet"), ("early", "late"),
             ("in", "out"), ("left", "right"), ("good", "bad"), ("strong", "weak"), ("rich", "poor"),
             ("clean", "dirty"), ("first", "last"), ("push", "pull"), ("win", "lose"), ("near", "far")]
SOUNDS = {"dog": "barks, woof woof", "cat": "meows, meow", "cow": "moos, moo", "duck": "quacks, quack quack",
          "sheep": "says baa", "pig": "oinks, oink oink", "horse": "neighs", "lion": "roars",
          "owl": "hoots, hoo hoo", "frog": "croaks, ribbit", "bee": "buzzes, bzzz", "snake": "hisses, sss",
          "rooster": "crows, cock-a-doodle-doo", "mouse": "squeaks", "donkey": "brays, hee-haw"}
COLORS = {"the sky": "blue", "grass": "green", "snow": "white", "a banana": "yellow", "coal": "black",
          "a strawberry": "red", "an orange": "orange", "a carrot": "orange", "milk": "white",
          "a lemon": "yellow", "a tomato": "red", "the sun": "yellow", "a leaf": "green", "chocolate": "brown",
          "a frog": "green", "the night sky": "dark blue or black", "a pumpkin": "orange", "an elephant": "gray"}
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
UNKNOWN_U = ["What time is it?", "What is the weather today?", "What is today's date?", "Where am I?",
             "What is my name?", "Who will win the game tonight?", "What is the news today?",
             "How old am I?", "What did I eat for breakfast?", "Where do I live?"]
UNKNOWN_A = ["I am sorry, I cannot know that. I do not have a clock, a calendar or the news.",
             "I do not know that, sorry. I cannot see or check things around you.",
             "I am not sure. I cannot look that up, but you could check a phone or ask someone near you.",
             "Sorry, I do not know that. I only know simple things I learned before."]


def fact(r):
    """Jedna otazka s odpovedi (bez pozdravu)."""
    k = r.choices(["add", "sub", "mul", "day", "month", "next", "cap", "opp", "sound", "color", "count",
                   "unknown", "spell"], weights=[6, 4, 4, 2, 2, 2, 4, 2, 2, 2, 1, 2, 1])[0]
    if k == "add":
        a, b = r.randint(0, 12), r.randint(0, 12)
        return r.choice(ADD_Q).format(a=a, b=b), r.choice([f"{a} + {b} is {a + b}.", f"{a} plus {b} equals {a + b}.",
                                                            f"The answer is {a + b}.", f"It is {a + b}."])
    if k == "sub":
        a = r.randint(1, 20)
        b = r.randint(0, a)
        return r.choice(SUB_Q).format(a=a, b=b), r.choice([f"{a} - {b} is {a - b}.", f"{a} minus {b} equals {a - b}.",
                                                            f"The answer is {a - b}.", f"It is {a - b}."])
    if k == "mul":
        a, b = r.randint(1, 10), r.randint(1, 10)
        return r.choice(MUL_Q).format(a=a, b=b), r.choice([f"{a} times {b} is {a * b}.", f"{a} x {b} = {a * b}.",
                                                            f"The answer is {a * b}.", f"It is {a * b}."])
    if k == "day":
        i = r.randrange(7)
        q = r.choice(["What day comes after {d}?", "What is the day after {d}?", "Which day is after {d}?",
                      "What day comes before {d}?", "How many days are in a week?", "What are the days of the week?"])
        if "after" in q:
            return q.format(d=DAYS[i]), f"The day after {DAYS[i]} is {DAYS[(i + 1) % 7]}."
        if "before" in q:
            return q.format(d=DAYS[i]), f"The day before {DAYS[i]} is {DAYS[i - 1]}."
        if "How many" in q:
            return q, "There are seven days in a week."
        return q, "The days of the week are " + ", ".join(DAYS[:-1]) + " and Sunday."
    if k == "month":
        i = r.randrange(12)
        q = r.choice(["What month comes after {m}?", "What is the first month of the year?",
                      "How many months are in a year?", "What is the last month of the year?",
                      "What month is before {m}?"])
        if "after" in q:
            return q.format(m=MONTHS[i]), f"After {MONTHS[i]} comes {MONTHS[(i + 1) % 12]}."
        if "before" in q:
            return q.format(m=MONTHS[i]), f"The month before {MONTHS[i]} is {MONTHS[i - 1]}."
        if "first" in q:
            return q, "The first month of the year is January."
        if "last" in q:
            return q, "The last month of the year is December."
        return q, "There are twelve months in a year."
    if k == "next":
        n = r.randint(0, 99)
        if r.random() < 0.5:
            return r.choice([f"What number comes after {n}?", f"What is after {n}?"]), f"After {n} comes {n + 1}."
        n += 1
        return r.choice([f"What number comes before {n}?", f"What is one less than {n}?"]), f"Before {n} comes {n - 1}."
    if k == "cap":
        c = r.choice(list(CAPITALS))
        q = r.choice(["What is the capital of {c}?", "What's the capital city of {c}?", "Which city is the capital of {c}?",
                      "capital of {c}?"]).format(c=c)
        return q, f"The capital of {c} is {CAPITALS[c]}."
    if k == "opp":
        a, b = r.choice(OPPOSITES)
        if r.random() < 0.5:
            a, b = b, a
        return r.choice([f"What is the opposite of {a}?", f"What's the opposite of \"{a}\"?"]), \
            f"The opposite of {a} is {b}."
    if k == "sound":
        an = r.choice(list(SOUNDS))
        return r.choice([f"What sound does a {an} make?", f"What does a {an} say?"]), f"A {an} {SOUNDS[an]}."
    if k == "color":
        t = r.choice(list(COLORS))
        return f"What color is {t}?", f"{t[0].upper() + t[1:]} is {COLORS[t]}."
    if k == "count":
        n = r.randint(3, 20)
        return r.choice([f"Can you count to {n}?", f"Count from 1 to {n}."]), \
            "Sure! " + ", ".join(str(i) for i in range(1, n + 1)) + "."
    if k == "spell":
        n = r.randint(0, 20)
        return r.choice([f"How do you write {n} in words?", f"How do you spell the number {n}?"]), \
            f"{n} is written as \"{WORDS[n]}\"."
    return r.choice(UNKNOWN_U), r.choice(UNKNOWN_A)


def conversation(r):
    turns = []
    if r.random() < 0.45:
        turns.append((r.choice(GREET_U), r.choice(GREET_A)))
    k = r.choices(["how", "who", "can", "fact"], weights=[2, 3, 2, 8])[0]
    n_mid = r.choices([1, 2, 3], weights=[5, 3, 1])[0]
    for _ in range(n_mid):
        if k == "how":
            turns.append((r.choice(HOW_U), r.choice(HOW_A)))
        elif k == "who":
            qs, ans = r.choice(WHO)
            turns.append((r.choice(qs), r.choice(ans)))
        elif k == "can":
            turns.append((r.choice(CAN_U), r.choice(CAN_A)))
        else:
            turns.append(fact(r))
        k = r.choices(["how", "who", "can", "fact"], weights=[1, 1, 1, 8])[0]
    if r.random() < 0.35:
        turns.append((r.choice(THANK_U), r.choice(THANK_A)))
        if r.random() < 0.3:
            turns.append((r.choice(BYE_U), r.choice(BYE_A)))
    if not turns:
        turns.append(fact(r))
    return "\n".join(f"User: {u}\nAssistant: {a}" for u, a in turns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = random.Random(a.seed)
    seen = set()
    with open(a.out, "w") as f:
        k = 0
        while k < a.n:
            c = conversation(r)
            if c in seen:
                continue
            seen.add(c)
            f.write(c + "\n<|endoftext|>\n")
            k += 1


if __name__ == "__main__":
    main()
