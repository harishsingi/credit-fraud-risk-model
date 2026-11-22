import math
WEIGHTS={"delinquencies":1.1,"utilisation":1.6,"low_income":0.5}
def score(income, delinquencies, utilisation):
    z=-2.2 + WEIGHTS["delinquencies"]*delinquencies + WEIGHTS["utilisation"]*utilisation + (WEIGHTS["low_income"] if income<35000 else 0)
    p=1/(1+math.exp(-z))
    drivers=sorted([("delinquencies",delinquencies),("utilisation",utilisation)], key=lambda x:-x[1])
    return {"fraud_probability":round(p,4),"top_drivers":[d[0] for d in drivers],"review_required":p>0.25}
