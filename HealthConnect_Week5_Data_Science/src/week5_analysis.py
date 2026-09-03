from __future__ import annotations
import csv, json, math, random, shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "data" / "HealthConnect_Appointment_Data_model_ready.csv"
OUT = Path(__file__).resolve().parents[1]
DATA, FIG = OUT / "data", OUT / "figures"

def num(v):
    try: return float(v) if v not in ("", None) else None
    except ValueError: return None
def median(values):
    a=sorted(x for x in values if x is not None); n=len(a); return (a[(n-1)//2]+a[n//2])/2
def svg(path, title, labels, values, colour="#4c72b0", pct=False):
    w,h,L,B=820,470,80,110; top=max(values) or 1
    bars=[]; step=(w-L-30)/len(labels); bw=step*.62
    for i,(lab,val) in enumerate(zip(labels,values)):
        x=L+i*step+(step-bw)/2; bh=(h-B-70)*val/top; y=h-B-bh
        display=f"{val:.1f}%" if pct else f"{int(val):,}"
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{colour}"/><text x="{x+bw/2:.1f}" y="{y-8:.1f}" text-anchor="middle">{display}</text><text x="{x+bw/2:.1f}" y="{h-B+25}" text-anchor="middle" transform="rotate(25 {x+bw/2:.1f} {h-B+25})">{lab}</text>')
    body=''.join(bars)
    path.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><style>text{{font-family:Arial;font-size:14px;fill:#222}}.t{{font-size:20px;font-weight:bold}}</style><rect width="100%" height="100%" fill="white"/><text class="t" x="{L}" y="35">{title}</text><line x1="{L}" y1="{h-B}" x2="{w-25}" y2="{h-B}" stroke="#555"/>{body}</svg>''', encoding="utf-8")

def auc(y, score):
    ranked=sorted(zip(score,y)); ranks=[]; i=0
    while i<len(ranked):
        j=i
        while j+1<len(ranked) and ranked[j+1][0]==ranked[i][0]: j+=1
        ranks += [((i+j+2)/2)]*(j-i+1); i=j+1
    pos=sum(y); neg=len(y)-pos
    return (sum(r for r,(_,v) in zip(ranks,ranked) if v)-pos*(pos+1)/2)/(pos*neg)

def metrics(y,p):
    pred=[int(x>=.5) for x in p]; tp=sum(a==b==1 for a,b in zip(y,pred)); tn=sum(a==b==0 for a,b in zip(y,pred)); fp=sum(a==0 and b==1 for a,b in zip(y,pred)); fn=sum(a==1 and b==0 for a,b in zip(y,pred))
    precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0
    return {"accuracy":round((tp+tn)/len(y),4),"precision":round(precision,4),"recall":round(recall,4),"f1":round(2*precision*recall/(precision+recall),4) if precision+recall else 0,"roc_auc":round(auc(y,p),4),"confusion_matrix":[[tn,fp],[fn,tp]]}

def grouped_split(rows):
    patients=defaultdict(list)
    for i,r in enumerate(rows): patients[r["patient_id"]].append(i)
    items=list(patients.items()); random.Random(42).shuffle(items); items.sort(key=lambda x:len(x[1]),reverse=True)
    total=len(rows); rate=sum(r["target"] for r in rows)/total; folds=[[] for _ in range(5)]; stats=[[0,0] for _ in range(5)]
    for _,idxs in items:
        n=len(idxs); p=sum(rows[i]["target"] for i in idxs); scores=[]
        for size,pos in stats:
            ns,np=size+n,pos+p; scores.append(((ns-total/5)/(total/5))**2+4*((np/ns)-rate)**2)
        k=min(range(5),key=lambda z:scores[z]); folds[k]+=idxs; stats[k][0]+=n; stats[k][1]+=p
    test=set(folds[0]); return [i for i in range(total) if i not in test],sorted(test)

def build_matrix(rows, train, test):
    numeric=["age","booking_lead_days","previous_appointments","previous_no_shows","prior_no_show_rate","distance_to_clinic_km"]
    cat=["appointment_type","appointment_day","appointment_time","reminder_sent","reminder_channel","has_prior_no_show","lead_time_bucket","booking_month","appointment_month"]
    med={c:median(rows[i][c] for i in train) for c in numeric}; means={}; stds={}
    for c in numeric:
        vals=[rows[i][c] if rows[i][c] is not None else med[c] for i in train]; means[c]=sum(vals)/len(vals); stds[c]=max(math.sqrt(sum((x-means[c])**2 for x in vals)/len(vals)),1e-9)
    categories={c:sorted({rows[i][c] for i in train}) for c in cat}; names=[f"num__{c}" for c in numeric]+[f"cat__{c}={v}" for c in cat for v in categories[c]]
    def rowvec(r):
        x=[((r[c] if r[c] is not None else med[c])-means[c])/stds[c] for c in numeric]
        x += [1.0 if r[c]==v else 0.0 for c in cat for v in categories[c]]; return x
    return [rowvec(rows[i]) for i in train],[rowvec(rows[i]) for i in test],names

def fit_logistic(X,y,epochs=550,lr=.12,l2=.001):
    n=len(X); d=len(X[0]); w=[0.0]*(d+1)
    for epoch in range(epochs):
        grad=[0.0]*(d+1)
        for x,t in zip(X,y):
            z=w[0]+sum(a*b for a,b in zip(w[1:],x)); p=1/(1+math.exp(-max(-30,min(30,z)))); e=p-t; grad[0]+=e
            for j,v in enumerate(x): grad[j+1]+=e*v
        step=lr/math.sqrt(1+epoch*.01); w[0]-=step*grad[0]/n
        for j in range(d): w[j+1]-=step*(grad[j+1]/n+l2*w[j+1])
    return w
def predict(X,w): return [1/(1+math.exp(-max(-30,min(30,w[0]+sum(a*b for a,b in zip(w[1:],x)))))) for x in X]

def main():
    OUT.mkdir(parents=True,exist_ok=True); DATA.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True); shutil.copy2(SRC,DATA/SRC.name)
    with SRC.open(encoding="utf-8") as f: raw=list(csv.DictReader(f))
    dist_med=median(num(r["distance_to_clinic_km"]) for r in raw); rows=[]
    for r in raw:
        b=datetime.strptime(r["booking_date"],"%Y-%m-%d"); a=datetime.strptime(r["appointment_date"],"%Y-%m-%d"); pa=int(r["previous_appointments"]); pn=int(r["previous_no_shows"]); lead=int(r["booking_lead_days"])
        rows.append({"appointment_id":r["appointment_id"],"patient_id":r["patient_id"],"target":int(r["appointment_outcome"]=="No-Show"),"age":num(r["age"]),"booking_lead_days":num(r["booking_lead_days"]),"previous_appointments":num(r["previous_appointments"]),"previous_no_shows":num(r["previous_no_shows"]),"prior_no_show_rate":pn/pa if pa else 0.0,"distance_to_clinic_km":num(r["distance_to_clinic_km"]),"appointment_type":r["appointment_type"],"appointment_day":r["appointment_day"],"appointment_time":r["appointment_time"],"reminder_sent":r["reminder_sent"],"reminder_channel":r["reminder_channel"] or "None","has_prior_no_show":"Yes" if pn else "No","lead_time_bucket":"0-7 days" if lead<=7 else "8-14 days" if lead<=14 else "15-30 days" if lead<=30 else "31+ days","booking_month":b.strftime("%b"),"appointment_month":a.strftime("%b")})
    tr,te=grouped_split(rows); Xtr,Xte,names=build_matrix(rows,tr,te); ytr=[rows[i]["target"] for i in tr]; yte=[rows[i]["target"] for i in te]; w=fit_logistic(Xtr,ytr); p=predict(Xte,w); m=metrics(yte,p)
    # derived dataset
    with (DATA/"week5_feature_dataset.csv").open("w",newline="",encoding="utf-8") as f:
        fields=list(rows[0]); out=csv.DictWriter(f,fieldnames=fields); out.writeheader(); out.writerows(rows)
    counts=Counter(r["target"] for r in rows); leadlabs=["0-7 days","8-14 days","15-30 days","31+ days"]; lead=[100*sum(r["target"] for r in rows if r["lead_time_bucket"]==x)/sum(r["lead_time_bucket"]==x for r in rows) for x in leadlabs]; priorlabs=["No","Yes"]; prior=[100*sum(r["target"] for r in rows if r["has_prior_no_show"]==x)/sum(r["has_prior_no_show"]==x for r in rows) for x in priorlabs]
    svg(FIG/"01_target_distribution.svg","Target distribution after excluding cancellations",["Attended","No-Show"],[counts[0],counts[1]],"#4c72b0")
    svg(FIG/"02_lead_time_noshow_rate.svg","No-show rate by booking lead time",leadlabs,lead,"#dd8452",True)
    svg(FIG/"03_prior_noshow_rate.svg","No-show rate by prior no-show history",priorlabs,prior,"#55a868",True)
    cm=m["confusion_matrix"]; svg(FIG/"04_confusion_matrix.svg","Logistic Regression: held-out patient test set",["True attended / predicted attended","True attended / predicted no-show","True no-show / predicted attended","True no-show / predicted no-show"],[cm[0][0],cm[0][1],cm[1][0],cm[1][1]],"#8172b2")
    imp=sorted(zip(names,w[1:]),key=lambda x:abs(x[1]),reverse=True)[:12]; svg(FIG/"05_logistic_coefficients.svg","Largest absolute Logistic Regression coefficients",[x[0].replace("cat__"," ").replace("num__","") for x in imp][::-1],[abs(x[1]) for x in imp][::-1],"#c44e52")
    result={"model":"Logistic Regression (manual, L2-regularised)","train_rows":len(tr),"test_rows":len(te),"shared_patients_train_test":0,"metrics":m,"lead_time_no_show_rates":dict(zip(leadlabs,lead)),"prior_history_no_show_rates":dict(zip(priorlabs,prior))}; (OUT/"evaluation_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    md=f'''# HealthConnect - Week 5 Project Summary (Data Science)

## Completed work

Built and evaluated a binary no-show baseline using **{len(rows):,}** non-cancelled appointments. The original files were not modified. The target is `No-Show = 1` and `Attended = 0`; cancellations remain excluded as agreed in Week 4.

## Data preparation and feature engineering

- `reminder_channel` missingness is retained as the meaningful category `None`.
- Created `prior_no_show_rate`, `has_prior_no_show`, `lead_time_bucket`, `booking_month` and `appointment_month`.
- Excluded IDs, outcome fields, raw dates and `waiting_time_minutes`. The latter remains a leakage risk because it may not be known at booking time.
- Used a grouped train/test split by `patient_id`, with **zero** shared patients between partitions.

## Initial evaluation

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} |

## Findings and recommendations

- No-show rate rises from **{lead[0]:.1f}%** for 0-7 days to **{lead[-1]:.1f}%** for 31+ days lead time.
- Patients with a prior no-show have a rate of **{prior[1]:.1f}%**, compared with **{prior[0]:.1f}%** otherwise.
- Use any risk score for proportionate support (for example, an additional reminder or administrative call), never to deny care or penalise a patient.
- Data Analytics can monitor lead-time and prior-attendance segments as dashboard KPIs; Data Science provides the target and model metrics.

## Limitations and Week 6

This is synthetic data, one grouped hold-out split and a provisional 0.50 threshold. Week 6 should add grouped cross-validation, calibration, threshold selection based on intervention capacity, a tree-model comparison and fairness review.
'''
    (OUT/"Week5_Project_Summary.md").write_text(md,encoding="utf-8")
    srcdir=OUT/"src"; srcdir.mkdir(exist_ok=True); portable=srcdir/"week5_analysis.py"; shutil.copy2(Path(__file__),portable)
    portable_text=portable.read_text(encoding="utf-8")
    portable_text=portable_text.replace(f'SRC = Path(r"{SRC}")', 'SRC = Path(__file__).resolve().parents[1] / "data" / "HealthConnect_Appointment_Data_model_ready.csv"')
    portable_text=portable_text.replace(f'OUT = Path(r"{OUT}")', 'OUT = Path(__file__).resolve().parents[1]')
    portable.write_text(portable_text, encoding="utf-8")
    nb={"cells":[{"cell_type":"markdown","metadata":{},"source":["# HealthConnect Clinic - Week 5 Data Science Baseline Modelling\\n","**Prepared by:** M'bengue mama | **Track:** Data Science\\n\\nThis notebook implements and documents the Week 4 no-show prediction plan. The complete standard-library implementation is supplied in `src/week5_analysis.py`.\\n"]},{"cell_type":"markdown","metadata":{},"source":["## 1. Week 4 foundation review\\n","- **Target:** `target_no_show` (1 = No-Show, 0 = Attended); cancellations are outside the primary model.\\n- **Safety and leakage:** `reminder_channel` blanks are the meaningful category `None`; `waiting_time_minutes` is excluded because it may not be known at booking.\\n- **Split:** held-out patients are separated from training patients to prevent repeat-patient leakage.\\n"]},{"cell_type":"markdown","metadata":{},"source":["## 2. Data preparation and feature engineering\\n","The analysis uses the separate model-ready data copy in `data/`. It creates `prior_no_show_rate`, `has_prior_no_show`, `lead_time_bucket`, `booking_month` and `appointment_month`. Numeric distance missingness is median-imputed inside the model pipeline. IDs, raw dates, outcome fields and waiting time are excluded as model features.\\n"]},{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["# Re-run the full, dependency-free analysis from the package root.\\n# It reads data/HealthConnect_Appointment_Data_model_ready.csv and refreshes all outputs.\\nfrom src.week5_analysis import main\\nmain()"]},{"cell_type":"markdown","metadata":{},"source":["## 3. Train/test strategy and baseline model\\n","A deterministic grouped split assigns complete patient histories either to training or test data. The baseline is L2-regularised Logistic Regression, selected for interpretability. The 0.50 threshold is provisional.\\n"]},{"cell_type":"markdown","metadata":{},"source":["## 4. Initial evaluation\\n",f"Held-out patient results: **Accuracy {m['accuracy']:.3f} | Precision {m['precision']:.3f} | Recall {m['recall']:.3f} | F1 {m['f1']:.3f} | ROC-AUC {m['roc_auc']:.3f}**.\\n\\nThe evaluation uses {len(te):,} appointments and has zero patients shared with training. See `evaluation_metrics.json` and `figures/04_confusion_matrix.svg`.\\n"]},{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":["import json\\nfrom pathlib import Path\\nresults = json.loads(Path('evaluation_metrics.json').read_text())\\nresults"]},{"cell_type":"markdown","metadata":{},"source":["## 5. Interpretation, limitations and Week 6\\n","Longer lead times and prior no-shows are meaningful risk signals. This is synthetic data and a single grouped hold-out split, not a deployable clinical model. Week 6 should add grouped cross-validation, probability calibration, a tree-model comparison, threshold selection aligned to staff capacity, and fairness review.\\n\\n**Cross-track dependency:** Data Analytics can monitor lead-time and prior-attendance KPIs; Data Science supplies the target definition, risk-feature logic and evaluation metrics.\\n"]}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
    (OUT/"Week5_Data_Science_Baseline_Modelling.ipynb").write_text(json.dumps(nb,indent=2),encoding="utf-8")
    readme=f'''# HealthConnect Experience Lab - Week 5 Data Science

## Deliverables

- `Week5_Data_Science_Baseline_Modelling.ipynb` - baseline modelling notebook
- `Week5_Project_Summary.md` - required concise summary
- `data/` - a separate copy of the provided model-ready source and derived features
- `figures/` - five decision-supporting visualisations
- `evaluation_metrics.json` - reproducible initial metrics
- `src/week5_analysis.py` - dependency-free reproducible analysis implementation

## Model

Logistic Regression predicts no-show risk using booking-time and prior-attendance information. The test split has no patient overlap with the training split. Initial ROC-AUC: **{m['roc_auc']:.3f}**. This is a synthetic-data baseline, not a deployable clinical model.
'''
    (OUT/"README.md").write_text(readme,encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
