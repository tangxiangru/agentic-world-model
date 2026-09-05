"""Round-3 synthetic data targeting GENERAL skills (allowed: matching general
style/format/difficulty). Original function names, NOT copied from any test item:
  (a) optional params present in schema but OMITTED unless mentioned;
  (b) two-argument functions where argument ORDER must follow the query;
  (c) boolean flag extraction (true AND false);
  (d) exact string / string-list copying."""
import json, random, sys
random.seed(2024)

def rf(a,b,nd=2): return round(random.uniform(a,b),nd)
def ri(a,b): return random.randint(a,b)

recs=[]

def fmt_val(v):
    if isinstance(v,list): return "["+", ".join(fmt_val(x) for x in v)+"]"
    return str(v)

def schema_from(req, opt=None):
    props={}
    for p,(t,d) in req.items(): props[p]={"type":t,"description":d}
    required=list(req.keys())
    if opt:
        for p,(t,d,default) in opt.items(): props[p]={"type":t,"description":d,"default":default}
    return props, required

# ---------- (a) optional params omitted (original functions) ----------
OPT_FUNCS=[
 dict(name="project_portfolio_growth",
      desc="Project the future value of a portfolio.",
      req={"starting_balance":("float","Starting balance."),
           "growth_rate":("float","Annual growth rate as a decimal."),
           "horizon_years":("integer","Number of years.")},
      opt={"yearly_deposit":("float","Additional yearly deposit.",0.0),
           "reinvest_dividends":("boolean","Whether to reinvest dividends.",True),
           "frequency":("string","Compounding frequency.","annual")},
      sample=lambda:{"starting_balance":rf(1000,50000),"growth_rate":rf(0.02,0.12,3),"horizon_years":ri(1,30)},
      tmpl=["Project a portfolio starting at ${starting_balance} growing {growth_rate} per year over {horizon_years} years.",
            "What will a ${starting_balance} portfolio be worth at a {growth_rate} annual growth rate after {horizon_years} years?"]),
 dict(name="summarize_dataset",
      desc="Produce a statistical summary of a dataset.",
      req={"source":("string","Dataset source name."),"statistic":("string","Statistic to compute.")},
      opt={"output_format":("string","Output format.","json"),"include_plots":("boolean","Whether to include plots.",False)},
      sample=lambda:{"source":random.choice(["region_sales","signup_log","sensor_feed","catalog","poll_results"]),"statistic":random.choice(["mean","median","spread","total","variance"])},
      tmpl=["Summarize the {statistic} of the {source} dataset.",
            "Give me the {statistic} for the {source} dataset."]),
 dict(name="book_appointment",
      desc="Book an appointment on the calendar.",
      req={"subject":("string","Appointment subject."),"length_minutes":("integer","Length in minutes.")},
      opt={"venue":("string","Where it takes place.","virtual"),"notify_attendees":("boolean","Whether to notify attendees.",True)},
      sample=lambda:{"subject":random.choice(["Consultation","Follow-up","Planning","Interview","Demo"]),"length_minutes":random.choice([15,30,45,60,90])},
      tmpl=["Book a '{subject}' appointment lasting {length_minutes} minutes.",
            "Set up a {length_minutes}-minute '{subject}' appointment."]),
 dict(name="run_simulation",
      desc="Run a numerical simulation.",
      req={"model_name":("string","Name of the model."),"steps":("integer","Number of steps.")},
      opt={"seed":("integer","Random seed.",0),"verbose":("boolean","Verbose logging.",False),"tolerance":("float","Convergence tolerance.",0.001)},
      sample=lambda:{"model_name":random.choice(["diffusion","predator_prey","heat_flow","traffic","epidemic"]),"steps":ri(50,5000)},
      tmpl=["Run the {model_name} simulation for {steps} steps.",
            "Simulate the {model_name} model over {steps} steps."]),
]

for f in OPT_FUNCS:
    for _ in range(320):
        args=f["sample"]()
        props,required=schema_from(f["req"],f["opt"])
        tool={"name":f["name"],"description":f["desc"],
              "parameters":{"type":"dict","properties":props,"required":required}}
        q=random.choice(f["tmpl"]).format(**{k:fmt_val(v) for k,v in args.items()})
        recs.append({"tools":[tool],"query":q,"answer":{"name":f["name"],"arguments":args}})

# ---------- (b) order-sensitive two-argument functions (original names) ----------
ORDER_FUNCS=[
 ("least_common_multiple","Compute the least common multiple of two integers.","first","second","the least common multiple of {first} and {second}", lambda:(ri(2,200),ri(2,200))),
 ("greatest_common_factor","Compute the greatest common factor of two integers.","first","second","the greatest common factor of {first} and {second}", lambda:(ri(2,999),ri(2,999))),
 ("subtract_values","Subtract the second value from the first.","minuend","subtrahend","{minuend} minus {subtrahend}", lambda:(ri(1,1000),ri(1,1000))),
 ("divide_values","Divide the first value by the second.","numerator","denominator","{numerator} divided by {denominator}", lambda:(ri(1,1000),ri(1,100))),
 ("raise_to_power","Raise the first value to the power of the second.","base","exponent","{base} raised to the power {exponent}", lambda:(ri(2,20),ri(2,6))),
 ("remainder_of","Compute the remainder of the first divided by the second.","dividend","divisor","the remainder when {dividend} is divided by {divisor}", lambda:(ri(1,1000),ri(2,50))),
]
for name,desc,p1,p2,phr,samp in ORDER_FUNCS:
    for _ in range(260):
        v1,v2=samp()
        args={p1:v1,p2:v2}
        tool={"name":name,"description":desc,
              "parameters":{"type":"dict","properties":{p1:{"type":"integer","description":f"First operand ({p1})."},p2:{"type":"integer","description":f"Second operand ({p2})."}},"required":[p1,p2]}}
        pre=random.choice(["What is ","Compute ","Find ","Calculate "])
        q=pre+phr.format(**{p1:v1,p2:v2})+"?"
        recs.append({"tools":[tool],"query":q,"answer":{"name":name,"arguments":args}})

# ---------- (c) boolean flag extraction (both values, original functions) ----------
BOOL_FUNCS=[
 ("retrieve_price_series","Retrieve a historical price series.","ticker",["ZX","QQ","MB","TT","AZ","NV"],
   "granularity",["daily","weekly","monthly"],"adjusted","with dividend and split adjustments applied"),
 ("query_table","Query rows from a data table.","table",["members","transactions","items","audit","tickets"],
   "order",["ascending","descending"],"include_archived","including archived rows"),
]
for name,desc,k1,vals1,k2,vals2,boolkey,boolphr in BOOL_FUNCS:
    for _ in range(300):
        v1=random.choice(vals1); v2=random.choice(vals2); bv=random.choice([True,False])
        args={k1:v1,k2:v2,boolkey:bv}
        tool={"name":name,"description":desc,
              "parameters":{"type":"dict","properties":{
                 k1:{"type":"string","description":"Primary identifier."},
                 k2:{"type":"string","description":"An option."},
                 boolkey:{"type":"boolean","description":boolphr.capitalize()+"."}},"required":[k1,k2,boolkey]}}
        if bv:
            q=f"{desc[:-1]} for {v1} at {v2} granularity, {boolphr}." if k2=='granularity' else f"Query {v1} ordered {v2}, {boolphr}."
        else:
            q=f"{desc[:-1]} for {v1} at {v2} granularity, but not {boolphr}." if k2=='granularity' else f"Query {v1} ordered {v2}, but not {boolphr}."
        recs.append({"tools":[tool],"query":q,"answer":{"name":name,"arguments":args}})

# ---------- (d) exact string / list copying (original function) ----------
PRODUCTS=["widget","gadget","sprocket","bolt kit","cable reel","gasket","hex nut","valve","bracket","filter cartridge"]
for _ in range(500):
    n=ri(1,3)
    chosen=random.sample(PRODUCTS,n)
    qty=[ri(1,120) for _ in range(n)]
    unit_cost=[round(random.choice([0.1,0.5,1.0,2.5,5.0,10.0]),2) for _ in range(n)]
    tool={"name":"create_purchase_order","description":"Create a purchase order for parts.",
          "parameters":{"type":"dict","properties":{
             "part":{"type":"array","description":"List of part names."},
             "count":{"type":"array","description":"Counts for each part."},
             "unit_cost":{"type":"array","description":"Unit cost of each part."}},"required":["part","count","unit_cost"]}}
    items_txt=", ".join(f"{qty[i]} {chosen[i]} at ${unit_cost[i]}" for i in range(n))
    q=f"Create a purchase order for: {items_txt}."
    recs.append({"tools":[tool],"query":q,"answer":{"name":"create_purchase_order","arguments":{"part":chosen,"count":qty,"unit_cost":unit_cost}}})

random.shuffle(recs)
out=sys.argv[1] if len(sys.argv)>1 else "synth3_norm.jsonl"
with open(out,"w") as o:
    for r in recs: o.write(json.dumps(r)+"\n")
print("wrote",len(recs),"to",out)
