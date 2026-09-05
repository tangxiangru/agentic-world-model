#!/usr/bin/env python3
"""Generate generic 'executable-style' single-call function-calling examples.

These are common math/string/date/unit/finance utility functions with concrete
randomized argument values. They match the general STYLE/DOMAIN of executable
function-calling benchmarks but are NOT derived from any specific test item.
"""
import json, random, math
random.seed(1234)

def P(type_, desc, **extra):
    d = {"type": type_, "description": desc}
    d.update(extra)
    return d

def tool(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props,
                           "required": required, "additionalProperties": False}}}

def rint(a, b): return random.randint(a, b)
def rflt(a, b, nd=2): return round(random.uniform(a, b), nd)

EXAMPLES = []
def add(tool_def, query, name, arguments, distractors=None):
    tools = [tool_def]
    if distractors:
        tools = tools + distractors
        random.shuffle(tools)
    EXAMPLES.append({
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "tool_calls": [
                {"type": "function", "function": {"name": name, "arguments": arguments}}]},
        ],
        "tools": tools,
        "query": query,
        "answer_str": f"{name}(" + ", ".join(f"{k}={v!r}" for k, v in arguments.items()) + ")",
    })

# ---- a pool of distractor tools to force selection ----
DISTRACTOR_POOL = [
    tool("get_weather", "Get current weather for a city.",
         {"city": P("string", "City name.")}, ["city"]),
    tool("send_email", "Send an email message.",
         {"to": P("string", "Recipient."), "subject": P("string", "Subject.")}, ["to", "subject"]),
    tool("convert_currency", "Convert an amount between currencies.",
         {"amount": P("number", "Amount."), "from_currency": P("string", "From."),
          "to_currency": P("string", "To.")}, ["amount", "from_currency", "to_currency"]),
    tool("random_number", "Generate a random number in a range.",
         {"low": P("integer", "Low."), "high": P("integer", "High.")}, ["low", "high"]),
]
def distractors(k=2):
    return random.sample(DISTRACTOR_POOL, k=min(k, len(DISTRACTOR_POOL)))

# ---- generators ----
def gen_geometry(n):
    for _ in range(n):
        kind = random.choice(["circle_area", "rectangle_area", "triangle_area", "cylinder_volume", "sphere_volume"])
        if kind == "circle_area":
            r = rflt(1, 50); t = tool("circle_area", "Calculate the area of a circle given its radius.",
                {"radius": P("number", "Radius of the circle.")}, ["radius"])
            q = random.choice([f"What is the area of a circle with radius {r}?",
                               f"Compute the area of a circle whose radius is {r}.",
                               f"Find the area of a circle of radius {r} units."])
            add(t, q, "circle_area", {"radius": r}, distractors())
        elif kind == "rectangle_area":
            w = rint(1, 100); h = rint(1, 100)
            t = tool("rectangle_area", "Calculate the area of a rectangle.",
                {"width": P("number", "Width."), "height": P("number", "Height.")}, ["width", "height"])
            q = random.choice([f"What's the area of a rectangle {w} wide and {h} tall?",
                               f"Calculate the area of a rectangle with width {w} and height {h}."])
            add(t, q, "rectangle_area", {"width": w, "height": h}, distractors())
        elif kind == "triangle_area":
            b = rint(1, 100); h = rint(1, 100)
            t = tool("triangle_area", "Calculate the area of a triangle given base and height.",
                {"base": P("number", "Base length."), "height": P("number", "Height.")}, ["base", "height"])
            q = random.choice([f"Find the area of a triangle with base {b} and height {h}.",
                               f"What is the area of a triangle whose base is {b} and height is {h}?"])
            add(t, q, "triangle_area", {"base": b, "height": h}, distractors())
        elif kind == "cylinder_volume":
            r = rflt(1, 20); h = rflt(1, 40)
            t = tool("cylinder_volume", "Calculate the volume of a cylinder.",
                {"radius": P("number", "Radius."), "height": P("number", "Height.")}, ["radius", "height"])
            q = f"Compute the volume of a cylinder with radius {r} and height {h}."
            add(t, q, "cylinder_volume", {"radius": r, "height": h}, distractors())
        else:
            r = rflt(1, 30)
            t = tool("sphere_volume", "Calculate the volume of a sphere.",
                {"radius": P("number", "Radius.")}, ["radius"])
            q = f"What is the volume of a sphere with radius {r}?"
            add(t, q, "sphere_volume", {"radius": r}, distractors())

def gen_math(n):
    for _ in range(n):
        kind = random.choice(["factorial", "gcd", "power", "is_prime", "quadratic", "mean", "hypotenuse"])
        if kind == "factorial":
            k = rint(1, 12); t = tool("factorial", "Compute the factorial of a non-negative integer.",
                {"n": P("integer", "The number.")}, ["n"])
            add(t, random.choice([f"Calculate the factorial of {k}.", f"What is {k} factorial?"]),
                "factorial", {"n": k}, distractors())
        elif kind == "gcd":
            a = rint(2, 500); b = rint(2, 500)
            t = tool("gcd", "Compute the greatest common divisor of two integers.",
                {"a": P("integer", "First integer."), "b": P("integer", "Second integer.")}, ["a", "b"])
            add(t, f"Find the greatest common divisor of {a} and {b}.", "gcd", {"a": a, "b": b}, distractors())
        elif kind == "power":
            base = rint(2, 15); exp = rint(2, 6)
            t = tool("power", "Raise a base to an exponent.",
                {"base": P("number", "Base."), "exponent": P("number", "Exponent.")}, ["base", "exponent"])
            add(t, f"What is {base} raised to the power of {exp}?", "power", {"base": base, "exponent": exp}, distractors())
        elif kind == "is_prime":
            k = rint(2, 999); t = tool("is_prime", "Check whether a number is prime.",
                {"number": P("integer", "The number to check.")}, ["number"])
            add(t, f"Is {k} a prime number?", "is_prime", {"number": k}, distractors())
        elif kind == "quadratic":
            a = rint(1, 9); b = rint(-9, 9); c = rint(-9, 9)
            t = tool("solve_quadratic", "Solve a quadratic equation ax^2+bx+c=0.",
                {"a": P("number", "Coefficient a."), "b": P("number", "Coefficient b."),
                 "c": P("number", "Coefficient c.")}, ["a", "b", "c"])
            add(t, f"Solve the quadratic equation with a={a}, b={b}, c={c}.", "solve_quadratic",
                {"a": a, "b": b, "c": c}, distractors())
        elif kind == "mean":
            nums = [rint(1, 100) for _ in range(random.randint(3, 6))]
            t = tool("mean", "Compute the arithmetic mean of a list of numbers.",
                {"numbers": P("array", "The numbers.", items={"type": "number"})}, ["numbers"])
            add(t, f"What is the average of {nums}?", "mean", {"numbers": nums}, distractors())
        else:
            a = rint(1, 50); b = rint(1, 50)
            t = tool("hypotenuse", "Compute the hypotenuse length of a right triangle given two legs.",
                {"a": P("number", "First leg."), "b": P("number", "Second leg.")}, ["a", "b"])
            add(t, f"Find the hypotenuse of a right triangle with legs {a} and {b}.", "hypotenuse",
                {"a": a, "b": b}, distractors())

def gen_convert(n):
    units = [("celsius_to_fahrenheit", "temperature in Celsius to Fahrenheit", "celsius", "Temperature in Celsius."),
             ("km_to_miles", "kilometers to miles", "km", "Distance in kilometers."),
             ("kg_to_pounds", "kilograms to pounds", "kg", "Mass in kilograms."),
             ("meters_to_feet", "meters to feet", "meters", "Length in meters.")]
    for _ in range(n):
        name, desc, arg, argdesc = random.choice(units)
        v = rflt(1, 500)
        t = tool(name, f"Convert {desc}.", {arg: P("number", argdesc)}, [arg])
        add(t, random.choice([f"Convert {v} using {name}.", f"Please convert {v} {arg}."]),
            name, {arg: v}, distractors())

def gen_finance(n):
    for _ in range(n):
        kind = random.choice(["simple_interest", "compound_interest", "tip", "discount"])
        if kind == "simple_interest":
            p = rint(100, 10000); r = rflt(1, 15); yy = rint(1, 10)
            t = tool("simple_interest", "Compute simple interest.",
                {"principal": P("number", "Principal amount."), "rate": P("number", "Annual rate percent."),
                 "years": P("number", "Number of years.")}, ["principal", "rate", "years"])
            add(t, f"Calculate the simple interest on a principal of {p} at {r}% for {yy} years.",
                "simple_interest", {"principal": p, "rate": r, "years": yy}, distractors())
        elif kind == "compound_interest":
            p = rint(100, 10000); r = rflt(1, 15); yy = rint(1, 10)
            t = tool("compound_interest", "Compute compound interest.",
                {"principal": P("number", "Principal."), "rate": P("number", "Annual rate percent."),
                 "years": P("number", "Years."), "n": P("integer", "Compounds per year.", default=1)},
                ["principal", "rate", "years"])
            add(t, f"What is the compound interest on {p} at {r}% over {yy} years?",
                "compound_interest", {"principal": p, "rate": r, "years": yy}, distractors())
        elif kind == "tip":
            amt = rflt(10, 300); pct = random.choice([10, 15, 18, 20, 25])
            t = tool("calculate_tip", "Calculate a tip amount for a bill.",
                {"bill": P("number", "Bill amount."), "percentage": P("number", "Tip percentage.")}, ["bill", "percentage"])
            add(t, f"How much is a {pct}% tip on a bill of {amt}?", "calculate_tip",
                {"bill": amt, "percentage": pct}, distractors())
        else:
            price = rflt(10, 500); pct = random.choice([5, 10, 20, 25, 30, 50])
            t = tool("apply_discount", "Apply a percentage discount to a price.",
                {"price": P("number", "Original price."), "discount": P("number", "Discount percent.")}, ["price", "discount"])
            add(t, f"What is the final price after a {pct}% discount on {price}?", "apply_discount",
                {"price": price, "discount": pct}, distractors())

def gen_string(n):
    for _ in range(n):
        kind = random.choice(["reverse", "count_vowels", "wordcount", "upper", "palindrome"])
        word = random.choice(["algorithm", "benchmark", "california", "function", "keyboard",
                               "mountain", "notebook", "elephant", "umbrella", "javascript"])
        if kind == "reverse":
            t = tool("reverse_string", "Reverse the characters of a string.",
                {"text": P("string", "The input string.")}, ["text"])
            add(t, f"Reverse the string '{word}'.", "reverse_string", {"text": word}, distractors())
        elif kind == "count_vowels":
            t = tool("count_vowels", "Count the number of vowels in a string.",
                {"text": P("string", "The input string.")}, ["text"])
            add(t, f"How many vowels are in '{word}'?", "count_vowels", {"text": word}, distractors())
        elif kind == "wordcount":
            sent = "the quick brown fox jumps"
            t = tool("word_count", "Count the number of words in a text.",
                {"text": P("string", "The text.")}, ["text"])
            add(t, f"How many words are in the sentence '{sent}'?", "word_count", {"text": sent}, distractors())
        elif kind == "upper":
            t = tool("to_uppercase", "Convert a string to uppercase.",
                {"text": P("string", "The input string.")}, ["text"])
            add(t, f"Convert '{word}' to uppercase.", "to_uppercase", {"text": word}, distractors())
        else:
            t = tool("is_palindrome", "Check whether a string is a palindrome.",
                {"text": P("string", "The input string.")}, ["text"])
            w = random.choice(["racecar", "level", "hello", "python", "noon"])
            add(t, f"Is '{w}' a palindrome?", "is_palindrome", {"text": w}, distractors())

def gen_datetime(n):
    for _ in range(n):
        kind = random.choice(["days_between", "add_days", "weekday"])
        if kind == "days_between":
            t = tool("days_between", "Compute the number of days between two dates (YYYY-MM-DD).",
                {"start_date": P("string", "Start date."), "end_date": P("string", "End date.")}, ["start_date", "end_date"])
            add(t, "How many days are there between 2023-01-01 and 2023-03-15?",
                "days_between", {"start_date": "2023-01-01", "end_date": "2023-03-15"}, distractors())
        elif kind == "add_days":
            d = rint(1, 100)
            t = tool("add_days", "Add a number of days to a date (YYYY-MM-DD).",
                {"date": P("string", "The base date."), "days": P("integer", "Days to add.")}, ["date", "days"])
            add(t, f"What date is {d} days after 2024-06-01?", "add_days",
                {"date": "2024-06-01", "days": d}, distractors())
        else:
            t = tool("weekday", "Return the day of the week for a date (YYYY-MM-DD).",
                {"date": P("string", "The date.")}, ["date"])
            add(t, "What day of the week is 2025-12-25?", "weekday", {"date": "2025-12-25"}, distractors())

def main():
    gen_geometry(900); gen_math(1400); gen_convert(600); gen_finance(800)
    gen_string(700); gen_datetime(600)
    random.shuffle(EXAMPLES)
    with open("synth_records.jsonl", "w") as f:
        for r in EXAMPLES:
            f.write(json.dumps({k: r[k] for k in ("messages", "tools", "query", "answer_str")}) + "\n")
    with open("synth_decon.jsonl", "w") as f:
        for r in EXAMPLES:
            f.write(json.dumps({"question": r["query"], "answer": r["answer_str"]}) + "\n")
    print("generated", len(EXAMPLES), "synthetic examples")

if __name__ == "__main__":
    main()
