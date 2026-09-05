"""Round-2 synthetic generator: adds natural-language phrasing variety that
requires light extraction/conversion (percentages, worded units), plus more
paraphrases. Original functions in the computational/scientific/utility domain.
Emits normalized JSONL (tools, query, answer)."""
import json, random, sys

random.seed(77)

def rf(a,b,nd=2): return round(random.uniform(a,b),nd)
def ri(a,b): return random.randint(a,b)

FAMILIES=[]
def fam(**kw): FAMILIES.append(kw)

# helper to format a probability sometimes as decimal, sometimes as percent text
def prob_phrasings(pname):
    def f(args):
        p=args[pname]
        return [("{v}", str(p)), ("{v}", str(p)), ("{pct}%", ("{:g}".format(round(p*100,4))))]
    return f

fam(
 name="binomial_probability", description="Compute the probability of exactly k successes in n independent Bernoulli trials.",
 params={"n":("integer","Number of trials."),"k":("integer","Number of successes."),"p":("float","Probability of success per trial (as a decimal).")},
 required=["n","k","p"],
 sampler=lambda:(lambda n:{"n":n,"k":ri(0,n),"p":rf(0.05,0.95)})(ri(3,40)),
 templates=[
   "Out of {n} attempts where each succeeds with probability {p}, what's the chance of exactly {k} successes?",
   "A biased coin lands heads {p_pct}% of the time. In {n} flips, what's the probability of exactly {k} heads?",
   "Rolling a special die where a six comes up {p_pct}% of the time, in {n} rolls what are the odds of exactly {k} sixes?",
   "Compute the binomial probability with n={n}, k={k}, and success probability {p}.",
   "If the success rate is {p_pct}%, over {n} independent trials what is the probability of getting exactly {k} successes?",
 ],
 special_pct=["p"],
)

fam(
 name="probability_at_least_one", description="Probability of at least one success across independent trials.",
 params={"p":("float","Per-trial success probability as a decimal."),"trials":("integer","Number of trials.")},
 required=["p","trials"],
 sampler=lambda:{"p":rf(0.02,0.6,3),"trials":ri(2,25)},
 templates=[
   "With a per-attempt success chance of {p_pct}%, what's the probability of at least one success in {trials} attempts?",
   "If each try works with probability {p}, what is the chance of at least one success over {trials} tries?",
 ],
 special_pct=["p"],
)

fam(
 name="expected_value_discrete", description="Compute the expected value given values and their probabilities.",
 params={"values":("array","List of outcome values."),"probabilities":("array","Matching list of probabilities.")},
 required=["values","probabilities"],
 sampler=lambda:(lambda d:{"values":[ri(1,50) for _ in range(d)],"probabilities":[round(1.0/d,3)]*d})(ri(3,4)),
 templates=[
   "Given outcomes {values} with probabilities {probabilities}, what is the expected value?",
   "Compute the expected value for values {values} and probabilities {probabilities}.",
 ],
)

fam(
 name="calculate_density", description="Compute density from mass and volume.",
 params={"mass":("float","Mass of the object."),"volume":("float","Volume occupied.")},
 required=["mass","volume"],
 sampler=lambda:{"mass":rf(1,500),"volume":rf(1,120)},
 templates=[
   "An object weighing {mass} kilograms takes up {volume} cubic meters. Find its density.",
   "It's pretty heavy at {mass} kg and occupies about {volume} cubic meters of space; what's the density?",
   "Determine the density given a mass of {mass} and a volume of {volume}.",
 ],
)

fam(
 name="displacement_kinematics", description="Compute displacement given initial velocity, acceleration, and time.",
 params={"initial_velocity":("float","Initial velocity."),"acceleration":("float","Acceleration."),"time":("float","Time elapsed.")},
 required=["initial_velocity","acceleration","time"],
 sampler=lambda:{"initial_velocity":rf(0,30),"acceleration":rf(0,12),"time":rf(1,15)},
 templates=[
   "An object initially moving at {initial_velocity} m/s accelerates at {acceleration} m/s^2 for {time} seconds. How far does it travel?",
   "Find the displacement with initial velocity {initial_velocity}, acceleration {acceleration}, over {time} seconds.",
 ],
)

fam(
 name="investment_growth", description="Compute the future value of an investment with annual compounding.",
 params={"principal":("float","Initial amount."),"annual_rate":("float","Annual growth rate as a decimal."),"years":("integer","Number of years.")},
 required=["principal","annual_rate","years"],
 sampler=lambda:{"principal":rf(500,50000),"annual_rate":rf(0.01,0.12,3),"years":ri(1,30)},
 templates=[
   "If I invest ${principal} growing at {rate_pct}% per year for {years} years, what will it be worth?",
   "Compute the future value of {principal} at an annual rate of {annual_rate} over {years} years.",
 ],
 special_pct=["annual_rate"],
)

fam(
 name="sales_tax_total", description="Compute the total price including sales tax.",
 params={"price":("float","Pre-tax price."),"tax_rate":("float","Sales tax rate as a decimal.")},
 required=["price","tax_rate"],
 sampler=lambda:{"price":rf(5,3000),"tax_rate":rf(0.03,0.12,3)},
 templates=[
   "What's the total cost of a ${price} item with a {tax_pct}% sales tax?",
   "Add a sales tax rate of {tax_rate} to a price of {price} and give the total.",
 ],
 special_pct=["tax_rate"],
)

fam(
 name="convert_meters_to_feet", description="Convert a length from meters to feet.",
 params={"meters":("float","Length in meters.")},
 required=["meters"],
 sampler=lambda:{"meters":rf(0.5,500)},
 templates=[
   "How many feet is {meters} meters?",
   "Convert {meters} m to feet.",
 ],
)

fam(
 name="wavelength_from_frequency", description="Compute the wavelength of a wave from its frequency.",
 params={"frequency":("float","Frequency in hertz."),"wave_speed":("float","Wave propagation speed.")},
 required=["frequency","wave_speed"],
 sampler=lambda:{"frequency":rf(20,20000),"wave_speed":random.choice([343.0,1500.0,299792458.0])},
 templates=[
   "A wave travels at {wave_speed} m/s with a frequency of {frequency} Hz. What is its wavelength?",
   "Compute the wavelength for frequency {frequency} and wave speed {wave_speed}.",
 ],
)

fam(
 name="ideal_gas_pressure", description="Compute pressure using the ideal gas law given moles, temperature, and volume.",
 params={"moles":("float","Amount of gas in moles."),"temperature":("float","Temperature in Kelvin."),"volume":("float","Volume in liters.")},
 required=["moles","temperature","volume"],
 sampler=lambda:{"moles":rf(0.1,10),"temperature":rf(200,500),"volume":rf(1,50)},
 templates=[
   "Using the ideal gas law, find the pressure for {moles} moles at {temperature} K in {volume} liters.",
   "Compute the pressure of {moles} mol of gas held at {temperature} Kelvin in a {volume} L container.",
 ],
)

fam(
 name="slope_between_points", description="Compute the slope of the line through two points.",
 params={"point1":("array","First point as [x, y]."),"point2":("array","Second point as [x, y].")},
 required=["point1","point2"],
 sampler=lambda:{"point1":[ri(-20,20),ri(-20,20)],"point2":[ri(-20,20),ri(-20,20)]},
 templates=[
   "What's the slope of the line through {point1} and {point2}?",
   "Find the slope between the points {point1} and {point2}.",
 ],
)

fam(
 name="matrix_vector_multiply", description="Multiply a 2x2 matrix by a 2D vector.",
 params={"matrix":("array","2x2 matrix as a list of rows."),"vector":("array","2D vector.")},
 required=["matrix","vector"],
 sampler=lambda:{"matrix":[[ri(-5,5),ri(-5,5)],[ri(-5,5),ri(-5,5)]],"vector":[ri(-5,5),ri(-5,5)]},
 templates=[
   "Multiply the matrix {matrix} by the vector {vector}.",
   "Compute the product of matrix {matrix} and vector {vector}.",
 ],
)

fam(
 name="median_of_list", description="Compute the median of a list of numbers.",
 params={"numbers":("array","List of numbers.")},
 required=["numbers"],
 sampler=lambda:{"numbers":[ri(1,100) for _ in range(ri(4,9))]},
 templates=[
   "What is the median of {numbers}?",
   "Find the median value in the list {numbers}.",
 ],
)

fam(
 name="permutations_count", description="Compute the number of permutations of n items taken r at a time.",
 params={"n":("integer","Total number of items."),"r":("integer","Number of items to arrange.")},
 required=["n","r"],
 sampler=lambda:(lambda n:{"n":n,"r":ri(1,n)})(ri(2,12)),
 templates=[
   "How many ways can {r} items be arranged from a set of {n}?",
   "Compute the number of permutations of {n} items taken {r} at a time.",
 ],
)

fam(
 name="combinations_count", description="Compute the number of combinations of n items taken r at a time.",
 params={"n":("integer","Total number of items."),"r":("integer","Number of items to choose.")},
 required=["n","r"],
 sampler=lambda:(lambda n:{"n":n,"r":ri(0,n)})(ri(2,15)),
 templates=[
   "In how many ways can I choose {r} items from {n}?",
   "Compute {n} choose {r}.",
 ],
)

fam(
 name="compound_annual_growth_rate", description="Compute CAGR from beginning value, ending value, and number of years.",
 params={"begin_value":("float","Starting value."),"end_value":("float","Ending value."),"years":("integer","Number of years.")},
 required=["begin_value","end_value","years"],
 sampler=lambda:{"begin_value":rf(100,10000),"end_value":rf(100,30000),"years":ri(1,20)},
 templates=[
   "An investment grew from {begin_value} to {end_value} over {years} years. What's the CAGR?",
   "Compute the compound annual growth rate given a start of {begin_value}, an end of {end_value}, and {years} years.",
 ],
)

fam(
 name="heat_energy", description="Compute heat energy from mass, specific heat, and temperature change.",
 params={"mass":("float","Mass in grams."),"specific_heat":("float","Specific heat capacity."),"delta_temp":("float","Temperature change.")},
 required=["mass","specific_heat","delta_temp"],
 sampler=lambda:{"mass":rf(10,1000),"specific_heat":rf(0.1,4.2,3),"delta_temp":rf(1,100)},
 templates=[
   "How much heat is needed to change {mass} g of a substance (specific heat {specific_heat}) by {delta_temp} degrees?",
   "Compute the heat energy for mass {mass}, specific heat {specific_heat}, and a temperature change of {delta_temp}.",
 ],
)

def fmt_val(v):
    if isinstance(v,list): return "["+", ".join(fmt_val(x) for x in v)+"]"
    return str(v)

def build_tool_schema(f):
    props={p:{"type":t,"description":d} for p,(t,d) in f["params"].items()}
    return {"name":f["name"],"description":f["description"],
            "parameters":{"type":"dict","properties":props,"required":list(f["required"])}}

def render(template,args,f):
    d={k:fmt_val(v) for k,v in args.items()}
    # add percent variants
    for pk in f.get("special_pct",[]):
        d[pk.replace("_rate","")+"_pct" if False else pk+"_pct"]="{:g}".format(round(args[pk]*100,6))
    # convenience aliases used in templates
    if "annual_rate" in args: d["rate_pct"]="{:g}".format(round(args["annual_rate"]*100,6))
    if "tax_rate" in args: d["tax_pct"]="{:g}".format(round(args["tax_rate"]*100,6))
    if "p" in args: d["p_pct"]="{:g}".format(round(args["p"]*100,6))
    try:
        return template.format(**d)
    except KeyError:
        return None

def main(out_path,per_family=260):
    others=list(FAMILIES)
    n=0
    with open(out_path,"w") as out:
        for f in FAMILIES:
            seen=set(); made=0; att=0
            while made<per_family and att<per_family*5:
                att+=1
                args=f["sampler"]()
                key=json.dumps(args,sort_keys=True)
                if key in seen: continue
                seen.add(key)
                tmpl=random.choice(f["templates"])
                q=render(tmpl,args,f)
                if q is None: continue
                toollist=[build_tool_schema(f)]
                if random.random()<0.3:
                    for g in random.sample([x for x in others if x["name"]!=f["name"]],k=random.choice([1,2])):
                        toollist.append(build_tool_schema(g))
                    random.shuffle(toollist)
                out.write(json.dumps({"tools":toollist,"query":q,"answer":{"name":f["name"],"arguments":args}})+"\n")
                n+=1; made+=1
    print("wrote",n,"examples across",len(FAMILIES),"families to",out_path)

if __name__=="__main__":
    out=sys.argv[1] if len(sys.argv)>1 else "synth2_norm.jsonl"
    pf=int(sys.argv[2]) if len(sys.argv)>2 else 260
    main(out,pf)
