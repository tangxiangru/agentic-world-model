"""Teach the GENERAL skill: arguments must follow the ORDER they appear in the
query, never sorted by magnitude — even for commutative-looking operations.
Original function names; generic parameter names (a/b, x/y, num1/num2, first/second)."""
import json, random, sys
random.seed(555)
def ri(a,b): return random.randint(a,b)
recs=[]

# (function_name, description, phrase_template using {p1}/{p2}, is_commutative_concept)
TWO=[
 ("least_common_multiple","Compute the least common multiple of two integers.","the least common multiple of {v1} and {v2}"),
 ("greatest_common_divisor","Compute the greatest common divisor of two integers.","the greatest common divisor of {v1} and {v2}"),
 ("sum_two_integers","Add two integers together.","the sum of {v1} and {v2}"),
 ("product_two_integers","Multiply two integers.","the product of {v1} and {v2}"),
 ("difference_of","Subtract the second integer from the first.","{v1} minus {v2}"),
 ("quotient_of","Divide the first integer by the second.","{v1} divided by {v2}"),
 ("max_of_two","Return the larger of two integers.","the maximum of {v1} and {v2}"),
 ("min_of_two","Return the smaller of two integers.","the minimum of {v1} and {v2}"),
 ("power_of","Raise the first integer to the power of the second.","{v1} to the power of {v2}"),
 ("compare_ratio","Compute the ratio of the first integer to the second.","the ratio of {v1} to {v2}"),
]
PARAM_PAIRS=[("a","b"),("x","y"),("num1","num2"),("first","second"),("m","n"),("value1","value2")]
PREF=["What is ","Compute ","Find ","Calculate ","Please compute ","Give me "]

for name,desc,phr in TWO:
    for _ in range(220):
        # random values; deliberately vary so first can be larger OR smaller than second
        v1=ri(2,999); v2=ri(2,999)
        p1,p2=random.choice(PARAM_PAIRS)
        tool={"name":name,"description":desc,
              "parameters":{"type":"dict","properties":{
                 p1:{"type":"integer","description":f"The first number ({p1})."},
                 p2:{"type":"integer","description":f"The second number ({p2})."}},"required":[p1,p2]}}
        q=random.choice(PREF)+phr.format(v1=v1,v2=v2)+"?"
        recs.append({"tools":[tool],"query":q,"answer":{"name":name,"arguments":{p1:v1,p2:v2}}})

# three-argument order preservation
THREE=[
 ("blend_three","Combine three values in the given order.","{v1}, {v2}, and {v3}"),
 ("weighted_triplet","Process three numbers in order.","{v1}, then {v2}, then {v3}"),
]
TRIP=[("a","b","c"),("x","y","z"),("first","second","third"),("p","q","r")]
for name,desc,phr in THREE:
    for _ in range(220):
        v1,v2,v3=ri(1,500),ri(1,500),ri(1,500)
        p1,p2,p3=random.choice(TRIP)
        tool={"name":name,"description":desc,
              "parameters":{"type":"dict","properties":{
                 p1:{"type":"integer","description":"First."},
                 p2:{"type":"integer","description":"Second."},
                 p3:{"type":"integer","description":"Third."}},"required":[p1,p2,p3]}}
        q=random.choice(PREF)+f"the result for {phr.format(v1=v1,v2=v2,v3=v3)}?"
        recs.append({"tools":[tool],"query":q,"answer":{"name":name,"arguments":{p1:v1,p2:v2,p3:v3}}})

random.shuffle(recs)
out=sys.argv[1] if len(sys.argv)>1 else "order_norm.jsonl"
with open(out,"w") as o:
    for r in recs: o.write(json.dumps(r)+"\n")
print("wrote",len(recs),"to",out)
