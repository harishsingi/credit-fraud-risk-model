import csv, random, pathlib
random.seed(11)
pathlib.Path("data").mkdir(exist_ok=True)
with open("data/applications.csv","w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["income","delinquencies","utilisation","fraud"])
    for _ in range(2000):
        inc=random.randint(25000,180000); dq=random.choice([0,0,0,1,2,3]); ut=round(random.random(),2)
        risk=(dq*0.35)+(ut*0.5)+(0.2 if inc<35000 else 0)
        wr.writerow([inc,dq,ut,1 if random.random()<risk*0.35 else 0])
print("ok")
