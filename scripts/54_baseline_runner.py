#!/usr/bin/env python3
"""Frozen pre-action selective retrieval baseline evaluation.

Expected CSV: query_id, split (train|val|test), baseline_ndcg,
  <action>_ndcg for requested actions (e.g. rerank_ndcg, prf_ndcg, qe_ndcg),
  z_* pre-action telemetry columns, emb_* frozen query embedding columns,
  optionally nqc_score (higher values assumed to indicate better retrieval).

All model fitting and hyperparameter selection happen on train/val ONLY.
Test relevance labels are used only to evaluate the already-frozen scores.
The output is exploratory unless the dataset/split/protocol were sealed in advance.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

SEED = 20271110
BUDGETS = (0.10, 0.25, 0.50, 0.75, 1.00)

def check_data(df, actions, zcols, ecols):
    required = {'query_id','split','baseline_ndcg'} | {f'{x}_ndcg' for x in actions}
    missing = required - set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    if not zcols: raise ValueError('Need pre-action telemetry columns named z_*')
    if not ecols: raise ValueError('Need frozen embedding columns named emb_*')
    if df.query_id.duplicated().any(): raise ValueError('query_id values must be unique')
    if set(df['split']) != {'train', 'val', 'test'}:
        raise ValueError('Require nonempty train, val, test splits, with no other split names')
    if df[list(required)].isna().any().any(): raise ValueError('Missing outcomes or IDs')
    for col in ['baseline_ndcg']+[f'{x}_ndcg' for x in actions]:
        if ((df[col] < -1e-8)|(df[col] > 1+1e-8)).any():
            raise ValueError(f'{col} outside [0,1]')

def cat_gbdt(xtr, ytr, xval, yval, xte):
    # Validation set only controls hyperparameter selection; no test-tuning.
    opts=[(3, 15, 1.0),(5, 31, 5.0)]
    best=None
    for depth, leaves, penalty in opts:
        model=LGBMRegressor(objective='regression', learning_rate=0.035,
                    n_estimators=180, max_depth=depth, num_leaves=leaves,
                    min_child_samples=18, reg_lambda=penalty, verbosity=-1,
                    random_state=SEED, n_jobs=1)
        model.fit(xtr,ytr)
        metric=mean_squared_error(yval, model.predict(xval))
        if best is None or metric < best[0]: best=(metric, model, {'depth':depth,'leaves':leaves,'l2':penalty})
    return best[1].predict(xte),best[2]

def quantile_lcb(xtr,ytr,xval,yval,xte):
    best=None
    for q in (0.10,0.20,0.30):
        m=LGBMRegressor(objective='quantile', alpha=q, learning_rate=0.035,
                        n_estimators=180, max_depth=3, num_leaves=15,
                        min_child_samples=18, reg_lambda=5.0, verbosity=-1,
                        random_state=SEED, n_jobs=1)
        m.fit(xtr,ytr)
        # Tune quantile solely by proper pinball loss on held-out validation.
        e=yval-m.predict(xval)
        pin=np.mean(np.maximum(q*e, (q-1)*e))
        if best is None or pin<best[0]: best=(pin,m,{'quantile':q})
    return best[1].predict(xte),best[2]

def semantic_knn(etr,ytr,eval_,yval,ete):
    # Frozen encoder embeddings; use cosine neighbors, weighted by similarity.
    etr=normalize(np.nan_to_num(etr,nan=0.0))
    eval_=normalize(np.nan_to_num(eval_,nan=0.0))
    ete=normalize(np.nan_to_num(ete,nan=0.0))
    best=None
    for k in (5,15,30,60):
        k=min(k,len(ytr))
        nn=NearestNeighbors(n_neighbors=k,metric='cosine',algorithm='brute',n_jobs=1)
        nn.fit(etr)
        def infer(features):
            distances, indices=nn.kneighbors(features)
            # Positive bounded weighting, without test-derived labels.
            weights=1.0/np.maximum(distances,0.025)
            return np.sum(weights*ytr[indices],axis=1)/np.sum(weights,axis=1)
        va=infer(eval_)
        metric=mean_squared_error(yval,va)
        if best is None or metric<best[0]: best=(metric,nn,k)
    nn=best[1]
    distances,indices=nn.kneighbors(ete,n_neighbors=best[2])
    weights=1.0/np.maximum(distances,0.025)
    return np.sum(weights*ytr[indices],axis=1)/np.sum(weights,axis=1),{'k':best[2]}

def fit_failure(xtr,base_tr,xte):
    fail=(base_tr<=1e-12).astype(int)
    if len(np.unique(fail))<2: return np.full(len(xte), float(fail[0]))
    m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                    LogisticRegression(max_iter=2000,random_state=SEED))
    m.fit(xtr,fail)
    return m.predict_proba(xte)[:,1]

def select_mask(score, frac, cap_only=False, cost=0.0):
    n=len(score)
    k=int(np.floor(frac*n+1e-12))
    order=np.argsort(-score,kind='mergesort')[:k]
    if cap_only: order=order[score[order] > cost]
    mask=np.zeros(n,dtype=bool);mask[order]=True
    return mask

def metrics(delta, score, frac, cost=0.0, cap=False):
    mask=select_mask(score,frac,cap,cost)
    return {'gain':float(np.mean(delta*mask)),
            'net_gain':float(np.mean((delta-cost)*mask)),
            'selected_frac':float(mask.mean()),
            'harm_rate_selected':float(np.mean(delta[mask]<0)) if mask.any() else 0.0}

def bootstrap_pair(delta, score_a, score_b, budgets=BUDGETS, nboot=10000, cost=0.0, cap=False):
    rng=np.random.default_rng(SEED)
    n=len(delta)
    def avg_curve(d,a,b):
        return np.mean([metrics(d,a,k,cost=cost,cap=cap)['net_gain'] -
                        metrics(d,b,k,cost=cost,cap=cap)['net_gain'] for k in budgets])
    estimate=avg_curve(delta,score_a,score_b)
    draws=np.empty(nboot)
    for i in range(nboot):
        ix=rng.integers(0,n,n)
        draws[i]=avg_curve(delta[ix],score_a[ix],score_b[ix])
    ci=np.quantile(draws,[.025,.975])
    # Permutation-style sign-flip test on paired per-query policy differences.
    # Scores & selection masks remain fixed under the randomization test.
    # Requires approximate sign symmetry for a valid two-sided p-value.
    diff=np.zeros(n,dtype=float)
    for b in budgets:
        ma=select_mask(score_a,b,cap,cost).astype(float)
        mb=select_mask(score_b,b,cap,cost).astype(float)
        diff+=(ma-mb)*(delta-cost)/len(budgets)
    observed=abs(np.mean(diff))
    extreme=0
    for _ in range(nboot):
        flips=rng.choice((-1.0,1.0),size=n)
        extreme+=(abs(np.mean(diff*flips))>=observed-1e-14)
    p=(extreme+1)/(nboot+1)
    return {'estimate':float(estimate),'ci_low':float(ci[0]),'ci_high':float(ci[1]),
            'signflip_p':float(p)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--actions',default='rerank,prf,qe')
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--cost',type=float,default=0.0,
                    help='Illustrative cost in nDCG units per intervention; report compute cost separately.')
    args=ap.parse_args()
    actions=[a.strip() for a in args.actions.split(',')]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.input)
    zcols=sorted(c for c in df if c.startswith('z_'))
    ecols=sorted(c for c in df if c.startswith('emb_'))
    check_data(df,actions,zcols,ecols)
    tr=df['split'].eq('train').to_numpy()
    va=df['split'].eq('val').to_numpy()
    te=df['split'].eq('test').to_numpy()
    xtr=df.loc[tr,zcols].to_numpy(dtype=float)
    xva=df.loc[va,zcols].to_numpy(dtype=float)
    xte=df.loc[te,zcols].to_numpy(dtype=float)
    # Training-only median imputation for GBDT; never derive imputers from test.
    imp=SimpleImputer(strategy='median').fit(xtr)
    xtr,xva,xte=imp.transform(xtr),imp.transform(xva),imp.transform(xte)
    etr=df.loc[tr,ecols].to_numpy(dtype=float)
    eva=df.loc[va,ecols].to_numpy(dtype=float)
    ete=df.loc[te,ecols].to_numpy(dtype=float)
    base=df.baseline_ndcg.to_numpy(float)
    fscore=fit_failure(xtr,base[tr],xte)
    if 'nqc_score' in df:
        # NQC convention: greater quality score implies stronger initial retrieval.
        nqc=-df.loc[te,'nqc_score'].to_numpy(float)
    else:
        nqc=np.full(sum(te),np.nan)
    rows=[];pairs=[];primaries=[];preds=[];hyper=[]
    for action in actions:
        delta=(df[f'{action}_ndcg']-df['baseline_ndcg']).to_numpy(float)
        yt=delta[tr]; yv=delta[va]; yd=delta[te]
        gb,config_gb=cat_gbdt(xtr,yt,xva,yv,xte)
        kn,config_kn=semantic_knn(etr,yt,eva,yv,ete)
        risk,config_risk=quantile_lcb(xtr,yt,xva,yv,xte)
        hyper.append({'action':action,'gbdt':config_gb,'knn':config_kn,'lcb':config_risk})
        # Perfect-failure routing (actual qrel failure indicator) is an ORACLE,
        # not an achievable pre-action baseline. Learned failure uses fscore.
        scores={'failure':fscore,'B1_gbdt':gb,'B2_semantic_knn':kn,
                'B3_quantile_LCB':risk,'oracle':yd.copy()}
        if np.isfinite(nqc).all():scores['nqc']=nqc
        # Random routing expected gain: no Monte Carlo variance.
        for b in BUDGETS:
            for method,s in scores.items():
                for mode in ('exact_budget','budget_cap'):
                    if method=='oracle' and mode=='budget_cap':
                        pass
                    m=metrics(yd,s,b,cost=args.cost,cap=(mode=='budget_cap'))
                    rows.append({'action':action,'budget':b,'mode':mode,
                                 'method':method,**m})
            rows.append({'action':action,'budget':b,'mode':'exact_budget','method':'random',
                         'gain':float(b*np.mean(yd)),
                         'net_gain':float(b*(np.mean(yd)-args.cost)),
                         'selected_frac':b,'harm_rate_selected':float(np.mean(yd<0))})
        for method in ('B1_gbdt','B2_semantic_knn','B3_quantile_LCB'):
            pairs.append({'action':action,'method':method,'vs':'failure', 'mode':'exact_budget',
                          **bootstrap_pair(yd,scores[method],fscore,nboot=args.bootstrap,cost=args.cost)})
        # Three PRE-SPECIFIED primary contrasts for reranking, all others exploratory.
        if action=='rerank':
            for method, comparator, capped in [
                ('B1_gbdt','failure',False),
                ('B2_semantic_knn','failure',False),
                ('B3_quantile_LCB','B1_gbdt',True)]:
                primaries.append({'action':action,'method':method,'vs':comparator,
                    'mode':'budget_cap' if capped else 'exact_budget',
                    **bootstrap_pair(yd,scores[method],scores[comparator],
                                     nboot=args.bootstrap,cost=args.cost,cap=capped)})
        preds.append(pd.DataFrame({'query_id':df.loc[te,'query_id'].to_numpy(),
            'action':action,'original_ndcg':base[te],'intervention_ndcg':df.loc[te,f'{action}_ndcg'].to_numpy(),
            'delta':yd,**scores}))
    pd.DataFrame(rows).to_csv(out/'budget_curves.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'paired_bootstrap.csv',index=False)
    if primaries:
        # Holm step-down correction over exactly three planned comparisons.
        ps=np.array([r['signflip_p'] for r in primaries])
        order=np.argsort(ps)
        adjusted=np.empty(len(ps))
        largest=0.0
        for rank,idx in enumerate(order):
            largest=max(largest,(len(ps)-rank)*ps[idx])
            adjusted[idx]=min(1.0,largest)
        for i,row in enumerate(primaries):row['holm_adjusted_signflip_p']=float(adjusted[i])
        pd.DataFrame(primaries).to_csv(out/'primary_comparisons.csv',index=False)
    pd.concat(preds,ignore_index=True).to_csv(out/'test_predictions.csv',index=False)
    (out/'hyperparameters.json').write_text(json.dumps(hyper,indent=2))
    (out/'run_metadata.json').write_text(json.dumps({'input':args.input,'actions':actions,
                 'seed':SEED,'budgets':BUDGETS,'cost':args.cost,
                 'bootstrap':args.bootstrap,'n_train':int(sum(tr)),
                 'n_val':int(sum(va)),'n_test':int(sum(te)),
                 'pre_features':zcols,'emb_features':len(ecols),
                 'warning':'Results cannot be called confirmatory if test labels were previously inspected.'},indent=2))
    print(f'Wrote frozen-policy output to {out}')

if __name__=='__main__': main()
