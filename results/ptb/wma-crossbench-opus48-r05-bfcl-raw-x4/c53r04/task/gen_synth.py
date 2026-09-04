"""Generate synthetic single-call function-calling examples in the computational/
scientific/utility domain (BFCL exec_simple *style*, not derived from test items).
All functions/templates are original. Output = normalized JSONL matching prep_xlam."""
import json, random, sys, math

random.seed(1234)

def rf(a, b, nd=2):
    return round(random.uniform(a, b), nd)
def ri(a, b):
    return random.randint(a, b)

# Each family: dict with name, description, params (name->(type,desc)),
# required list, a sampler() -> args dict, and templates: list of format strings.
FAMILIES = []
def fam(**kw):
    FAMILIES.append(kw); return kw

fam(
 name="compute_rectangle_area", description="Compute the area of a rectangle given its length and width.",
 params={"length":("float","Length of the rectangle."),"width":("float","Width of the rectangle.")},
 required=["length","width"],
 sampler=lambda: {"length":rf(1,200),"width":rf(1,200)},
 templates=[
   "I have a rectangular plot that is {length} meters long and {width} meters wide. What is its area?",
   "Could you find the area of a rectangle whose length is {length} and width is {width}?",
   "Calculate the area of a rectangle measuring {length} by {width}.",
   "A garden bed is {length} m long and {width} m across. How large is its area?",
 ])

fam(
 name="triangle_area_base_height", description="Compute the area of a triangle from its base and height.",
 params={"base":("float","The base length."),"height":("float","The height.")},
 required=["base","height"],
 sampler=lambda: {"base":rf(1,100),"height":rf(1,100)},
 templates=[
   "What's the area of a triangle with a base of {base} and a height of {height}?",
   "Find the triangle area given base {base} and height {height}.",
   "I need the area of a triangular sail with base {base} m and height {height} m.",
 ])

fam(
 name="circle_area_from_radius", description="Compute the area of a circle from its radius.",
 params={"radius":("float","Radius of the circle.")},
 required=["radius"],
 sampler=lambda: {"radius":rf(0.5,50)},
 templates=[
   "How much area does a circle with radius {radius} cover?",
   "Compute the area of a circle whose radius is {radius}.",
   "A round table has a radius of {radius} meters; what is its surface area?",
 ])

fam(
 name="cylinder_volume", description="Compute the volume of a cylinder from radius and height.",
 params={"radius":("float","Radius of the base."),"height":("float","Height of the cylinder.")},
 required=["radius","height"],
 sampler=lambda: {"radius":rf(1,30),"height":rf(1,80)},
 templates=[
   "What is the volume of a cylinder with radius {radius} and height {height}?",
   "Find the cylinder volume for a can of radius {radius} cm and height {height} cm.",
 ])

fam(
 name="compute_density", description="Compute density given mass and volume.",
 params={"mass":("float","Mass of the object."),"volume":("float","Volume of the object.")},
 required=["mass","volume"],
 sampler=lambda: {"mass":rf(1,500),"volume":rf(1,100)},
 templates=[
   "An object weighs {mass} kg and occupies {volume} cubic meters. What is its density?",
   "Determine the density of a material with mass {mass} and volume {volume}.",
   "I measured a rock at {mass} kilograms filling {volume} cubic meters of space; find its density.",
 ])

fam(
 name="kinetic_energy", description="Compute the kinetic energy of a moving object.",
 params={"mass":("float","Mass in kilograms."),"velocity":("float","Velocity in meters per second.")},
 required=["mass","velocity"],
 sampler=lambda: {"mass":rf(0.5,100),"velocity":rf(1,60)},
 templates=[
   "A {mass} kg object moves at {velocity} m/s. What is its kinetic energy?",
   "Compute the kinetic energy for mass {mass} and velocity {velocity}.",
 ])

fam(
 name="final_velocity_kinematics", description="Compute final velocity given initial velocity, acceleration, and time.",
 params={"initial_velocity":("float","Initial velocity."),"acceleration":("float","Acceleration."),"time":("float","Elapsed time.")},
 required=["initial_velocity","acceleration","time"],
 sampler=lambda: {"initial_velocity":rf(0,30),"acceleration":rf(-10,15),"time":rf(1,20)},
 templates=[
   "An object starts at {initial_velocity} m/s and accelerates at {acceleration} m/s^2 for {time} seconds. What's its final velocity?",
   "Find the final velocity with initial velocity {initial_velocity}, acceleration {acceleration}, and time {time}.",
 ])

fam(
 name="ohms_law_voltage", description="Compute voltage from current and resistance using Ohm's law.",
 params={"current":("float","Current in amperes."),"resistance":("float","Resistance in ohms.")},
 required=["current","resistance"],
 sampler=lambda: {"current":rf(0.1,20),"resistance":rf(1,500)},
 templates=[
   "If a current of {current} amps flows through a {resistance} ohm resistor, what is the voltage?",
   "Compute the voltage for current {current} and resistance {resistance}.",
 ])

fam(
 name="simple_interest", description="Compute simple interest given principal, annual rate, and time in years.",
 params={"principal":("float","Principal amount."),"rate":("float","Annual interest rate as a percentage."),"time":("float","Time in years.")},
 required=["principal","rate","time"],
 sampler=lambda: {"principal":rf(100,50000),"rate":rf(1,15),"time":rf(1,30)},
 templates=[
   "How much simple interest accrues on ${principal} at {rate}% per year over {time} years?",
   "Calculate simple interest for a principal of {principal}, rate {rate}, and time {time}.",
 ])

fam(
 name="compound_interest_amount", description="Compute the final amount for compound interest.",
 params={"principal":("float","Principal."),"rate":("float","Annual rate as a percentage."),"years":("integer","Number of years."),"n":("integer","Compounding periods per year.")},
 required=["principal","rate","years","n"],
 sampler=lambda: {"principal":rf(500,20000),"rate":rf(1,12),"years":ri(1,25),"n":random.choice([1,2,4,12])},
 templates=[
   "Invest ${principal} at {rate}% compounded {n} times a year for {years} years. What's the final amount?",
   "Find the compound amount: principal {principal}, rate {rate}, {years} years, compounded {n} times yearly.",
 ])

fam(
 name="binomial_probability", description="Compute the probability of exactly k successes in n independent trials.",
 params={"n":("integer","Number of trials."),"k":("integer","Number of successes."),"p":("float","Probability of success on a single trial.")},
 required=["n","k","p"],
 sampler=lambda: (lambda n:{"n":n,"k":ri(0,n),"p":rf(0.05,0.95)})(ri(2,40)),
 templates=[
   "In {n} trials with success probability {p}, what's the chance of exactly {k} successes?",
   "Compute the binomial probability for n={n}, k={k}, p={p}.",
   "If I flip a biased coin {n} times with heads probability {p}, what's the probability of exactly {k} heads?",
 ])

fam(
 name="mean_of_values", description="Compute the arithmetic mean of a list of numbers.",
 params={"values":("array","List of numbers.")},
 required=["values"],
 sampler=lambda: {"values":[rf(0,100) for _ in range(ri(3,7))]},
 templates=[
   "What is the average of these numbers: {values}?",
   "Compute the mean of {values}.",
   "Find the arithmetic mean for the dataset {values}.",
 ])

fam(
 name="standard_deviation", description="Compute the standard deviation of a list of numbers.",
 params={"data":("array","List of numeric data points.")},
 required=["data"],
 sampler=lambda: {"data":[ri(1,100) for _ in range(ri(4,8))]},
 templates=[
   "Calculate the standard deviation of {data}.",
   "What's the spread (standard deviation) for these measurements: {data}?",
 ])

fam(
 name="euclidean_distance", description="Compute the Euclidean distance between two points in 2D.",
 params={"point_a":("array","First point [x, y]."),"point_b":("array","Second point [x, y].")},
 required=["point_a","point_b"],
 sampler=lambda: {"point_a":[rf(-50,50,1),rf(-50,50,1)],"point_b":[rf(-50,50,1),rf(-50,50,1)]},
 templates=[
   "How far apart are the points {point_a} and {point_b}?",
   "Compute the Euclidean distance between {point_a} and {point_b}.",
 ])

fam(
 name="cosine_similarity_vectors", description="Compute the cosine similarity between two vectors.",
 params={"vector_a":("array","First vector."),"vector_b":("array","Second vector.")},
 required=["vector_a","vector_b"],
 sampler=lambda: (lambda d:{"vector_a":[rf(0,1,1) for _ in range(d)],"vector_b":[rf(0,1,1) for _ in range(d)]})(ri(3,5)),
 templates=[
   "Find the cosine similarity between {vector_a} and {vector_b}.",
   "How similar are the feature vectors {vector_a} and {vector_b}? Use cosine similarity.",
 ])

fam(
 name="dot_product", description="Compute the dot product of two equal-length vectors.",
 params={"a":("array","First vector."),"b":("array","Second vector.")},
 required=["a","b"],
 sampler=lambda: (lambda d:{"a":[ri(-9,9) for _ in range(d)],"b":[ri(-9,9) for _ in range(d)]})(ri(2,5)),
 templates=[
   "What is the dot product of {a} and {b}?",
   "Compute the dot product between vectors {a} and {b}.",
 ])

fam(
 name="celsius_to_fahrenheit", description="Convert a temperature from Celsius to Fahrenheit.",
 params={"celsius":("float","Temperature in degrees Celsius.")},
 required=["celsius"],
 sampler=lambda: {"celsius":rf(-40,120)},
 templates=[
   "Convert {celsius} degrees Celsius to Fahrenheit.",
   "What is {celsius}C in Fahrenheit?",
 ])

fam(
 name="km_to_miles", description="Convert a distance from kilometers to miles.",
 params={"kilometers":("float","Distance in kilometers.")},
 required=["kilometers"],
 sampler=lambda: {"kilometers":rf(1,1000)},
 templates=[
   "How many miles is {kilometers} kilometers?",
   "Convert {kilometers} km into miles.",
 ])

fam(
 name="bmi_calculator", description="Compute body mass index from weight and height.",
 params={"weight_kg":("float","Weight in kilograms."),"height_m":("float","Height in meters.")},
 required=["weight_kg","height_m"],
 sampler=lambda: {"weight_kg":rf(40,140),"height_m":rf(1.4,2.1)},
 templates=[
   "Compute the BMI for someone weighing {weight_kg} kg and {height_m} m tall.",
   "What's the body mass index at weight {weight_kg} and height {height_m}?",
 ])

fam(
 name="factorial", description="Compute the factorial of a non-negative integer.",
 params={"n":("integer","The number to compute the factorial of.")},
 required=["n"],
 sampler=lambda: {"n":ri(0,12)},
 templates=[
   "What is {n} factorial?",
   "Compute the factorial of {n}.",
 ])

fam(
 name="gcd_two_numbers", description="Compute the greatest common divisor of two integers.",
 params={"a":("integer","First integer."),"b":("integer","Second integer.")},
 required=["a","b"],
 sampler=lambda: {"a":ri(2,999),"b":ri(2,999)},
 templates=[
   "What's the greatest common divisor of {a} and {b}?",
   "Find the GCD of {a} and {b}.",
 ])

fam(
 name="quadratic_discriminant", description="Compute the discriminant of a quadratic ax^2+bx+c.",
 params={"a":("float","Coefficient a."),"b":("float","Coefficient b."),"c":("float","Coefficient c.")},
 required=["a","b","c"],
 sampler=lambda: {"a":rf(1,10),"b":rf(-10,10),"c":rf(-10,10)},
 templates=[
   "For the quadratic with a={a}, b={b}, c={c}, what is the discriminant?",
   "Compute the discriminant of {a}x^2 + {b}x + {c}.",
 ])

fam(
 name="future_value_annuity", description="Compute the future value of a series of equal payments.",
 params={"payment":("float","Payment per period."),"rate":("float","Interest rate per period as a decimal."),"periods":("integer","Number of periods.")},
 required=["payment","rate","periods"],
 sampler=lambda: {"payment":rf(50,2000),"rate":rf(0.01,0.1,3),"periods":ri(2,40)},
 templates=[
   "If I deposit {payment} each period at a periodic rate of {rate} for {periods} periods, what's the future value?",
   "Compute the future value of an annuity with payment {payment}, rate {rate}, and {periods} periods.",
 ])

fam(
 name="percentage_change", description="Compute the percentage change from an old value to a new value.",
 params={"old_value":("float","The original value."),"new_value":("float","The new value.")},
 required=["old_value","new_value"],
 sampler=lambda: {"old_value":rf(1,1000),"new_value":rf(1,1000)},
 templates=[
   "What is the percentage change from {old_value} to {new_value}?",
   "Compute the percent change when a value goes from {old_value} to {new_value}.",
 ])

fam(
 name="pythagorean_hypotenuse", description="Compute the hypotenuse of a right triangle from the two legs.",
 params={"a":("float","Length of one leg."),"b":("float","Length of the other leg.")},
 required=["a","b"],
 sampler=lambda: {"a":rf(1,100),"b":rf(1,100)},
 templates=[
   "A right triangle has legs {a} and {b}. What is the hypotenuse?",
   "Find the hypotenuse given legs of {a} and {b}.",
 ])

fam(
 name="speed_from_distance_time", description="Compute average speed given distance and time.",
 params={"distance":("float","Distance traveled."),"time":("float","Time taken.")},
 required=["distance","time"],
 sampler=lambda: {"distance":rf(1,1000),"time":rf(0.5,20)},
 templates=[
   "If I travel {distance} km in {time} hours, what's my average speed?",
   "Compute the average speed for a distance of {distance} over {time} hours.",
 ])

fam(
 name="convert_currency", description="Convert an amount of money using a given exchange rate.",
 params={"amount":("float","Amount in the source currency."),"rate":("float","Exchange rate to the target currency.")},
 required=["amount","rate"],
 sampler=lambda: {"amount":rf(1,10000),"rate":rf(0.5,150,3)},
 templates=[
   "Convert {amount} units of currency using an exchange rate of {rate}.",
   "How much is {amount} at an exchange rate of {rate}?",
 ])

fam(
 name="monthly_loan_payment", description="Compute the monthly payment for a fixed-rate loan.",
 params={"principal":("float","Loan principal."),"annual_rate":("float","Annual interest rate as a percentage."),"months":("integer","Loan term in months.")},
 required=["principal","annual_rate","months"],
 sampler=lambda: {"principal":rf(1000,300000),"annual_rate":rf(2,12),"months":random.choice([12,24,36,48,60,120,240,360])},
 templates=[
   "What's the monthly payment on a ${principal} loan at {annual_rate}% APR over {months} months?",
   "Compute the monthly loan payment for principal {principal}, annual rate {annual_rate}, term {months} months.",
 ])

fam(
 name="probability_at_least_one", description="Compute the probability of at least one success over independent trials.",
 params={"p":("float","Probability of success in a single trial."),"trials":("integer","Number of trials.")},
 required=["p","trials"],
 sampler=lambda: {"p":rf(0.01,0.5,3),"trials":ri(2,30)},
 templates=[
   "With a per-trial success probability of {p}, what's the chance of at least one success in {trials} trials?",
   "Compute the probability of at least one success given p={p} and {trials} trials.",
 ])

fam(
 name="convert_hours_to_minutes", description="Convert a duration in hours to minutes.",
 params={"hours":("float","Number of hours.")},
 required=["hours"],
 sampler=lambda: {"hours":rf(0.5,48,1)},
 templates=[
   "How many minutes are in {hours} hours?",
   "Convert {hours} hours to minutes.",
 ])

fam(
 name="weighted_average", description="Compute a weighted average of values with corresponding weights.",
 params={"values":("array","List of values."),"weights":("array","List of weights.")},
 required=["values","weights"],
 sampler=lambda: (lambda d:{"values":[ri(1,100) for _ in range(d)],"weights":[ri(1,10) for _ in range(d)]})(ri(3,5)),
 templates=[
   "Compute the weighted average of values {values} with weights {weights}.",
   "What is the weighted mean for data {values} and weights {weights}?",
 ])

fam(
 name="temperature_kelvin_to_celsius", description="Convert a temperature from Kelvin to Celsius.",
 params={"kelvin":("float","Temperature in Kelvin.")},
 required=["kelvin"],
 sampler=lambda: {"kelvin":rf(200,400)},
 templates=[
   "Convert {kelvin} Kelvin to Celsius.",
   "What is {kelvin} K expressed in degrees Celsius?",
 ])

fam(
 name="area_of_trapezoid", description="Compute the area of a trapezoid from its two parallel sides and height.",
 params={"base1":("float","Length of the first parallel side."),"base2":("float","Length of the second parallel side."),"height":("float","Distance between the parallel sides.")},
 required=["base1","base2","height"],
 sampler=lambda: {"base1":rf(1,50),"base2":rf(1,50),"height":rf(1,30)},
 templates=[
   "Find the area of a trapezoid with parallel sides {base1} and {base2} and height {height}.",
   "Compute the trapezoid area given bases {base1} and {base2} with height {height}.",
 ])

fam(
 name="power_of_number", description="Raise a base to an exponent.",
 params={"base":("float","The base value."),"exponent":("integer","The exponent.")},
 required=["base","exponent"],
 sampler=lambda: {"base":rf(1,12),"exponent":ri(2,6)},
 templates=[
   "What is {base} raised to the power of {exponent}?",
   "Compute {base} to the {exponent} power.",
 ])

fam(
 name="convert_pounds_to_kg", description="Convert a weight from pounds to kilograms.",
 params={"pounds":("float","Weight in pounds.")},
 required=["pounds"],
 sampler=lambda: {"pounds":rf(1,500)},
 templates=[
   "Convert {pounds} pounds to kilograms.",
   "How many kilograms is {pounds} lbs?",
 ])

fam(
 name="grade_point_average", description="Compute the average of a list of exam scores.",
 params={"scores":("array","List of exam scores.")},
 required=["scores"],
 sampler=lambda: {"scores":[ri(50,100) for _ in range(ri(3,6))]},
 templates=[
   "Compute the average of these exam scores: {scores}.",
   "What's the mean score given {scores}?",
 ])

fam(
 name="net_force", description="Compute net force from mass and acceleration (Newton's second law).",
 params={"mass":("float","Mass in kilograms."),"acceleration":("float","Acceleration in m/s^2.")},
 required=["mass","acceleration"],
 sampler=lambda: {"mass":rf(0.5,200),"acceleration":rf(0.1,30)},
 templates=[
   "What net force is needed to accelerate a {mass} kg object at {acceleration} m/s^2?",
   "Compute the force from mass {mass} and acceleration {acceleration}.",
 ])

fam(
 name="discount_price", description="Compute the final price after applying a percentage discount.",
 params={"price":("float","Original price."),"discount_percent":("float","Discount percentage.")},
 required=["price","discount_percent"],
 sampler=lambda: {"price":rf(5,2000),"discount_percent":rf(5,70)},
 templates=[
   "An item costs ${price} and is {discount_percent}% off. What's the final price?",
   "Apply a {discount_percent}% discount to a price of {price}.",
 ])

fam(
 name="tip_amount", description="Compute the tip for a bill given a tip percentage.",
 params={"bill":("float","Total bill amount."),"tip_percent":("float","Tip percentage.")},
 required=["bill","tip_percent"],
 sampler=lambda: {"bill":rf(10,400),"tip_percent":random.choice([10,15,18,20,22])*1.0},
 templates=[
   "How much should I tip on a ${bill} bill at {tip_percent}%?",
   "Compute a {tip_percent}% tip for a bill of {bill}.",
 ])

def fmt_val(v):
    if isinstance(v, list):
        return "[" + ", ".join(fmt_val(x) for x in v) + "]"
    return str(v)

def render(template, args):
    d = {k: fmt_val(v) for k, v in args.items()}
    return template.format(**d)

def build_tool_schema(f, params_subset=None):
    props = {}
    for pname,(ptype,desc) in f["params"].items():
        props[pname] = {"type":ptype,"description":desc}
    return {"name":f["name"],"description":f["description"],
            "parameters":{"type":"dict","properties":props,"required":list(f["required"])}}

def main(out_path, per_family=220):
    others = [f for f in FAMILIES]
    with open(out_path,"w") as out:
        n=0
        for f in FAMILIES:
            seen=set()
            attempts=0
            made=0
            while made<per_family and attempts<per_family*4:
                attempts+=1
                args=f["sampler"]()
                key=json.dumps(args,sort_keys=True)
                if key in seen: continue
                seen.add(key)
                tmpl=random.choice(f["templates"])
                query=render(tmpl,args)
                tool=build_tool_schema(f)
                # sometimes add 1-2 distractor tools
                toollist=[tool]
                if random.random()<0.3:
                    distractors=random.sample([g for g in others if g["name"]!=f["name"]], k=random.choice([1,2]))
                    for g in distractors:
                        toollist.append(build_tool_schema(g))
                    random.shuffle(toollist)
                rec={"tools":toollist,"query":query,"answer":{"name":f["name"],"arguments":args}}
                out.write(json.dumps(rec)+"\n")
                n+=1; made+=1
    print("wrote",n,"synthetic examples across",len(FAMILIES),"families to",out_path)

if __name__=="__main__":
    out=sys.argv[1] if len(sys.argv)>1 else "synth_norm.jsonl"
    pf=int(sys.argv[2]) if len(sys.argv)>2 else 220
    main(out,pf)
