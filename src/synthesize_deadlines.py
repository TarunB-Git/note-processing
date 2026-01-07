#!/usr/bin/env python3
"""
synth_deadlines_rich.py

High-variation synthetic deadline sentence generator.
Outputs CSV with columns: sentence,label,label_source

Usage:
    python src/synth_deadlines_rich.py --num 2000 --out data/processed/synth_deadlines_rich.csv --seed 123

Options:
    --num      Number of sentences to generate (default 2000)
    --out      Output CSV path
    --seed     Random seed for reproducibility
    --dedupe   Remove duplicates in final output (default True)
    --verbose  Print a few samples after generation
"""
import csv
import os
import random
import argparse
from datetime import datetime

# -------------------------
# Vocabulary / phrase lists
# -------------------------
OBJECTS = [
    "report", "presentation", "expense report", "invoice", "draft", "slides",
    "proposal", "assignment", "timesheet", "application", "budget", "summary",
    "minutes", "project plan", "code fix", "documentation", "article", "contract",
    "schedule", "form", "summary memo", "meeting notes", "design doc", "test plan",
    "release notes", "research notes", "evaluation", "dataset", "analysis"
]

ACTIONS = [
    "submit", "finish", "complete", "turn in", "hand in", "finalize", "prepare",
    "email", "send", "deliver", "upload", "post", "share", "provide", "return",
    "produce", "compile", "resolve", "implement"
]

PEOPLE = [
    "Alice", "Bob", "Ruthik", "Tarun", "team", "QA", "dev", "manager", "PM",
    "engineering", "ops", "finance", "HR"
]

WEEKDAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]
TIMES_OF_DAY = ["9am","10am","11am","noon","1pm","2pm","3pm","4pm","5pm","6pm","7pm","8pm","11:59pm"]

STRONG_CUES = ["deadline","due","no later than","due on","due by","deadline for"]
WEAK_CUES = ["by","before","until","on or before","expected by","required by"]
ALL_CUES = STRONG_CUES + WEAK_CUES

ABBREV = ["EOD","COB","EOW","ASAP","ASAP please","ASAP.", "by EOD", "by COB"]

REMINDER_PREFIXES = ["Reminder:", "Note:", "FYI -", "Heads up:", "Quick reminder:"]
SIGNOFFS = ["Thanks,", "Thanks!", "Regards,", "- Ruthik", "- Tarun", "- Team", "Best,"]
POLITE_PREFIX = ["Please", "Kindly", "Please make sure to", "Do try to", "Pls"]

# additional small clause tails to make notes more natural
TAILS = [
    "", "if possible.", "if you can.", "thanks.", "— thanks", "please.", "asap.", "by then, please.",
    "so we can proceed.", "to meet the deadline.", "for the review."
]

SHORT_FORMS = [
    "{obj} by {time}",
    "do {obj} by {time}",
    "finish {obj} by {time}",
    "{obj} - due {time}",
    "due: {obj} {time}"
]

BULLET_PREFIXES = ["-","*","•","• "]

# --------------------------------
# Time expression generation logic
# --------------------------------
def random_static_date():
    """Return a static date string like 'August 10' or 'August 10, 2024' and optional time."""
    month = random.choice(MONTHS)
    day = random.randint(1, 28)  # keep simple, avoid month-length issues
    s = f"{month} {day}"
    if random.random() < 0.4:
        s += f", {datetime.now().year}"
    if random.random() < 0.3:
        if random.random() < 0.6:
            s += f" at {random.choice(TIMES_OF_DAY)}"
        else:
            s += f" by {random.choice(TIMES_OF_DAY)}"
    return s

def random_weekday_expr():
    """Return 'this Friday', 'next Tuesday', 'Friday', etc., perhaps with time."""
    typ = random.random()
    if typ < 0.4:
        return random.choice(["this", "next"]) + " " + random.choice(WEEKDAYS)
    elif typ < 0.7:
        return random.choice(WEEKDAYS)
    else:
        # include time
        return random.choice(WEEKDAYS) + " at " + random.choice(TIMES_OF_DAY)

def random_relative_expr():
    """Return relative time expressions including very short and very long spans."""
    r = random.random()
    # immediate / ultra-short responses
    if r < 0.12:
        return random.choice(["now", "immediately", "ASAP", "one hour tops", "one hour max", "one minute", "next second", "right away"])
    # seconds-level (very rare, for "do next second" style)
    if r < 0.18:
        sec = random.choice([1,2,3,5,10,30])
        return f"in {sec} second{'s' if sec!=1 else ''}"
    # hours (1..240)
    if r < 0.45:
        h = random.choice([1,2,3,4,6,8,12,24,48,72,120,240])
        return f"in {h} hour{'s' if h!=1 else ''}"
    # days (1..365)
    if r < 0.75:
        d = random.choice([1,2,3,4,5,7,10,14,30,60,90,180,300,365])
        return f"in {d} day{'s' if d!=1 else ''}"
    # weeks/months / end-of-period expressions
    if r < 0.88:
        return random.choice(["next week", "in 2 weeks", "in 3 weeks", "in 4 weeks", "in 6 months", "in 3 months"])
    if r < 0.96:
        return random.choice(["by end of month", "by the end of the week", "by the end of the quarter", "by EOD", "by COB"])
    # fallback to an abbreviation
    return random.choice(ABBREV)


def random_time_expr():
    """Return a time expression mixing static, weekday, relative, or explicit time-of-day."""
    r = random.random()
    if r < 0.28:
        return random_static_date()
    if r < 0.58:
        return random_weekday_expr()
    if r < 0.86:
        return random_relative_expr()
    # include a specific time-of-day phrase
    return random.choice(TIMES_OF_DAY)

# -----------------------
# Template constructions
# -----------------------
def template_variants(obj, action, time_expr, cue=None):
    """Return a list of natural templates for given object/action/time - exhaustive variations."""
    action_cap = action.capitalize()
    variants = []

    # canonical imperative / polite forms
    variants.append(f"{action_cap} the {obj} {cue or 'by'} {time_expr}.")
    variants.append(f"Please {action} the {obj} {cue or 'by'} {time_expr}.")
    variants.append(f"{action_cap} {obj} {cue or 'by'} {time_expr}.")
    variants.append(f"Can you {action} the {obj} {cue or 'by'} {time_expr}?")
    variants.append(f"{obj.capitalize()} {cue or 'by'} {time_expr}.")
    variants.append(f"{obj.capitalize()} — {cue or 'due'} {time_expr}.")
    variants.append(f"{obj.capitalize()} - due {time_expr}.")
    variants.append(f"Due {time_expr}: {obj}.")
    variants.append(f"{cue or 'Due'}: {obj} {time_expr}.")
    variants.append(f"{action_cap} the {obj} {cue or 'on or before'} {time_expr}.")
    variants.append(f"Make sure the {obj} is {cue or 'completed by'} {time_expr}.")
    variants.append(f"Ensure the {obj} is {cue or 'completed by'} {time_expr}.")
    variants.append(f"Target: {obj} {cue or 'by'} {time_expr}.")
    variants.append(f"ETA: {obj} {cue or 'by'} {time_expr}.")
    variants.append(f"Reminder: {obj} {cue or 'due'} {time_expr}.")
    variants.append(f"Don't forget to {action} the {obj} {cue or 'by'} {time_expr}.")
    variants.append(f"Needs to be {cue or 'finished by'} {time_expr}: {obj}.")
    variants.append(f"{action_cap} the {obj} — {cue or 'by'} {time_expr}.")
    variants.append(f"{action_cap} the {obj} {cue or 'no later than'} {time_expr}.")
    variants.append(f"{obj.capitalize()} needs to be {cue or 'done by'} {time_expr}.")
    variants.append(f"{obj.capitalize()} should be {cue or 'done by'} {time_expr}.")
    # short forms and casual
    variants.append(f"{action} {obj} by {time_expr}.")
    variants.append(f"do {obj} by {time_expr}")
    variants.append(f"finish {obj} by {time_expr}")
    variants.append(f"make {obj} by {time_expr}")
    variants.append(f"create {obj} before {time_expr}")
    variants.append(f"{obj} by {time_expr}")
    variants.append(f"{obj} due {time_expr}")
    variants.append(f"due: {obj} {time_expr}")
    variants.append(f"{action_cap} {obj} — due {time_expr}")
    # question-like/confirmations
    variants.append(f"Is the {obj} due {time_expr}?")
    variants.append(f"When is the {obj} due? (Preferably {time_expr})")
    variants.append(f"Can we have the {obj} {cue or 'by'} {time_expr}?")
    # bullet list forms
    for bullet in BULLET_PREFIXES:
        variants.append(f"{bullet} {action_cap} the {obj} {cue or 'by'} {time_expr}")
        variants.append(f"{bullet} {obj} - due {time_expr}")
    # email-like short
    variants.append(f"{action_cap} {obj} {cue or 'by'} {time_expr} - {random.choice(SIGNOFFS)}")
    # extra polite variants
    variants.append(f"{random.choice(POLITE_PREFIX)} {action} the {obj} {cue or 'by'} {time_expr}, {random.choice(TAILS)}")
    # variation with person/assignee
    variants.append(f"{random.choice(PEOPLE)}: {action_cap} the {obj} {cue or 'by'} {time_expr}.")
    variants.append(f"{random.choice(PEOPLE)}, please {action} the {obj} {cue or 'by'} {time_expr}.")

    # in/within forms (explicit "in N days/hours" style)
    variants.append(f"{action_cap} the {obj} in {time_expr}.")
    variants.append(f"Please {action} the {obj} in {time_expr}.")
    variants.append(f"{action_cap} the {obj} within {time_expr}.")
    variants.append(f"Make sure to {action} the {obj} within {time_expr}.")
    variants.append(f"{obj.capitalize()} — {action} in {time_expr}.")
    variants.append(f"{action_cap} {obj} within {time_expr}.")

    # final sanitization: ensure single spaces, strip double punctuation
    clean_variants = []
    for v in variants:
        v2 = " ".join(v.split())
        v2 = v2.replace(" .", ".").replace(" ,", ",")
        clean_variants.append(v2)
    return clean_variants


# -----------------------
# Sentence assembly logic
# -----------------------
def generate_one_sentence():
    # choose object and action
    obj = random.choice(OBJECTS)
    action = random.choice(ACTIONS)
    # time expression
    time_expr = random_time_expr()
    # choose whether to use a strong or weak cue; strong cues allowed even without time
    if random.random() < 0.32:
        cue = random.choice(STRONG_CUES)
    else:
        cue = random.choice(ALL_CUES)
    # create variants for the given triplet
    cand_variants = template_variants(obj, action, time_expr, cue)
    sent = random.choice(cand_variants)
    # occasionally add small noise: parentheses, inline notes, abbreviations
    if random.random() < 0.12:
        if not sent.endswith(".") and not sent.endswith("?"):
            sent = sent.strip() + "."
    if random.random() < 0.08:
        tail = random.choice(TAILS)
        if tail:
            # avoid duplicate punctuation
            if not sent.endswith(".") and not sent.endswith("!"):
                sent = sent + " " + tail
            else:
                sent = sent[:-1] + ", " + tail
    # occasionally add explicit time-of-day suffix
    if random.random() < 0.07 and "at" not in time_expr and "by" not in time_expr:
        sent = sent.replace(time_expr, time_expr + " at " + random.choice(TIMES_OF_DAY))
    # sometimes uppercase or remove polite prefix to mimic notes
    if random.random() < 0.05:
        sent = sent.lower()
    if random.random() < 0.03:
        # add a signoff line (simulate note handwriting)
        sent = sent + " " + random.choice(SIGNOFFS)
    # sometimes produce very short note-like form
    if random.random() < 0.06:
        short_tpl = random.choice(SHORT_FORMS)
        sent = short_tpl.format(obj=obj, time=time_expr)
    # strip extra whitespace
    sent = " ".join(sent.split())
    return sent

# -----------------------
# Main generation function
# -----------------------
def generate_sentences(num=2000, dedupe=True, seed=None):
    if seed is not None:
        random.seed(seed)
    out = []
    attempts = 0
    max_attempts = num * 6  # avoid infinite loops
    while len(out) < num and attempts < max_attempts:
        attempts += 1
        s = generate_one_sentence()
        # optional simple filter: avoid sentences that are too short or non-sensical
        if len(s) < 8:
            continue
        # disallow sentences where 'by' is used without a time phrase (heuristic)
        if " by " in s.lower() and not any(t.lower() in s.lower() for t in WEEKDAYS + [m.lower() for m in MONTHS] + [w.lower() for w in ["today","tomorrow","tonight","eod","cob","asap","end of week","end of the month"]]):
            # allow if contains explicit time tokens
            if not any(tok in s.lower() for tok in ["am", "pm", "noon", "11:59", "eod", "cob", "by end", "end of"]):
                # still allow if strong cue present
                if not any(sc in s.lower() for sc in STRONG_CUES):
                    continue
        out.append(s)
    # dedupe optionally while preserving order
    if dedupe:
        seen = set()
        unique = []
        for s in out:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        out = unique[:num]
    # return list of tuples (sentence, label, label_source)
    return [(s, "deadline", "synthetic") for s in out]

# -----------------------
# CSV write helper
# -----------------------
def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sentence","label","label_source"])
        for r in rows:
            writer.writerow(r)

# -----------------------
# CLI entrypoint
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Generate rich synthetic deadline sentences")
    ap.add_argument("--num", type=int, default=2000, help="Number of sentences to generate")
    ap.add_argument("--out", type=str, default="synthetic_deadlines_rich.csv", help="Output CSV path")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--dedupe", action="store_true", default=True, help="Remove duplicates in output (default True)")
    ap.add_argument("--no-dedupe", dest="dedupe", action="store_false", help="Do not dedupe")
    ap.add_argument("--verbose", action="store_true", help="Print sample output after generation")
    args = ap.parse_args()

    rows = generate_sentences(num=args.num, dedupe=args.dedupe, seed=args.seed)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} sentences to {args.out}")
    if args.verbose:
        print("Sample sentences:")
        for s,_,_ in rows[:20]:
            print("-", s)

if __name__ == "__main__":
    main()

