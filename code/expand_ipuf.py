"""Genisletme: Interpose-PUF (iPUF) modelleme-direnci + genisletilmis BER.
Mevcut DOGRULANMIS XORArbiterPUF building-block'larini kullanir (uydurma yok)."""
import numpy as np, csv, time
from puf_models import XORArbiterPUF, challenge_to_feature, random_challenges, bit_error_rate
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

class InterposePUF:
    """(x,y)-iPUF: ust XOR-PUF (n+1 stage), alt XOR-PUF (n stage);
    alt yanit ust challenge'in ortasina interpose edilir (Nguyen 2019)."""
    def __init__(self, n, x, y, rng):
        self.n=n; self.pos=n//2
        self.upper=XORArbiterPUF(n+1, x, rng)
        self.lower=XORArbiterPUF(n, y, rng)
    def response(self, C):
        r_low=self.lower.response(C)                      # (m,)
        Cup=np.insert(C, self.pos, r_low, axis=1)         # (m, n+1)
        return self.upper.response(Cup)

def attack(puf, n, n_crp, rng, seed):
    C=random_challenges(n_crp, n, rng); R=puf.response(C)
    Phi=challenge_to_feature(C)
    Xtr,Xte,ytr,yte=train_test_split(Phi,R,test_size=0.2,random_state=seed)
    out={}
    lr=LogisticRegression(max_iter=300).fit(Xtr,ytr); out["LR"]=lr.score(Xte,yte)
    ml=MLPClassifier(hidden_layer_sizes=(64,64),max_iter=120,random_state=seed).fit(Xtr,ytr); out["MLP"]=ml.score(Xte,yte)
    return out

n=64; rows=[]
for (x,y) in [(1,1),(2,2),(1,2)]:
    for tr in [20000,50000,200000]:
        for seed in [0,1,2]:
            rng=np.random.default_rng(1000*seed+x*10+y)
            puf=InterposePUF(n,x,y,rng)
            a=attack(puf,n,tr,rng,seed)
            for m,acc in a.items():
                rows.append(dict(ipuf=f"({x},{y})",n=n,train_size=tr,seed=seed,model=m,test_acc=acc))
            print(f"({x},{y}) tr={tr} seed={seed} LR={a['LR']:.3f} MLP={a['MLP']:.3f}",flush=True)
w=csv.DictWriter(open("../results/ipuf_attack.csv","w"),fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]

# genisletilmis BER: sigma x k
ber=[]
for k in [1,2,3,4,5,6]:
    for sg in [0.01,0.025,0.05,0.075,0.1]:
        for seed in [0,1,2]:
            rng=np.random.default_rng(7*seed+k)
            puf=XORArbiterPUF(n,k,rng)
            C=random_challenges(5000,n,rng)
            b=bit_error_rate(puf,C,sg,n_repeats=15,rng=rng)
            ber.append(dict(n=n,k=k,sigma=sg,seed=seed,ber=b))
    print(f"BER k={k} done",flush=True)
w=csv.DictWriter(open("../results/ber_sigma_k.csv","w"),fieldnames=list(ber[0])); w.writeheader(); [w.writerow(r) for r in ber]
print("DONE ipuf_attack.csv + ber_sigma_k.csv")
