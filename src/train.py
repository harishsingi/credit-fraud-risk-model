import csv
from src.model import score
rows=list(csv.DictReader(open("data/applications.csv")))
tp=fp=fn=tn=0
for r in rows:
    pred=score(int(r["income"]),int(r["delinquencies"]),float(r["utilisation"]))["review_required"]; y=int(r["fraud"])
    tp+=pred and y; fp+=pred and not y; fn+=(not pred) and y; tn+=(not pred) and not y
print({"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":round(tp/max(1,tp+fp),3),"recall":round(tp/max(1,tp+fn),3)})
