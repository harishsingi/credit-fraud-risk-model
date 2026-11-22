import argparse
from src.model import score
ap=argparse.ArgumentParser(); ap.add_argument("--income",type=int,required=True); ap.add_argument("--delinquencies",type=int,required=True); ap.add_argument("--utilisation",type=float,required=True); a=ap.parse_args()
print(score(a.income,a.delinquencies,a.utilisation))
