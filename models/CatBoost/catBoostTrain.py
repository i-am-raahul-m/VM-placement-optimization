import argparse, os, json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, log_loss
from catboost import CatBoostClassifier, Pool

def metrics_at_threshold(y,p,t):
    yhat=(p>=t).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,yhat,labels=[0,1]).ravel()
    return {"threshold":float(t),"PREC":float(precision_score(y,yhat,zero_division=0)),"REC":float(recall_score(y,yhat,zero_division=0)),"F1":float(f1_score(y,yhat,zero_division=0)),"ACC":float(accuracy_score(y,yhat)),"TN":int(tn),"FP":int(fp),"FN":int(fn),"TP":int(tp)}

def pick_threshold_max_precision_at_recall(y,p,recall_target):
    grid=np.linspace(0.01,0.99,99)
    cand=[m for t in grid if (m:=metrics_at_threshold(y,p,t))["REC"]>=recall_target]
    if cand:
        cand.sort(key=lambda m:(m["PREC"],m["REC"],m["F1"]),reverse=True)
        c=cand[0]; c["rule"]=f"max_precision_given_recall≥{recall_target}"; return c
    scored=[metrics_at_threshold(y,p,t) for t in grid]
    c=max(scored,key=lambda m:m["F1"]); c["rule"]="best_F1_fallback"; return c

def sigmoid(z): return 1/(1+np.exp(-z))
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))

def fit_temperature(p,y):
    z=logit(p)
    T_vals=np.linspace(0.5,5.0,91)
    best_T=1.0; best_ll=1e9
    for T in T_vals:
        q=sigmoid(z/T)
        ll=log_loss(y,q,labels=[0,1])
        if ll<best_ll: best_ll=ll; best_T=T
    return float(best_T)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_x",default="model_features_train.csv")
    ap.add_argument("--train_y",default="model_labels_train.csv")
    ap.add_argument("--test_x",default="data/model_features_test.csv")
    ap.add_argument("--test_y",default="model_labels_test.csv")
    ap.add_argument("--outdir",default="./artifacts")
    ap.add_argument("--iterations",type=int,default=8000)
    ap.add_argument("--depth",type=int,default=6)
    ap.add_argument("--lr",type=float,default=0.05)
    ap.add_argument("--l2",type=float,default=10.0)
    ap.add_argument("--recall_target",type=float,default=0.90)
    ap.add_argument("--early_stopping",type=int,default=150)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()

    Xtr_df=pd.read_csv(args.train_x)
    ytr=pd.read_csv(args.train_y)["sla_violation"].astype(int).to_numpy()
    Xte_df=pd.read_csv(args.test_x)
    yte=pd.read_csv(args.test_y)["sla_violation"].astype(int).to_numpy()
    feat_cols=Xtr_df.columns.tolist()

    sss=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=args.seed)
    tr_idx,va_idx=next(sss.split(Xtr_df,ytr))
    X_tr=Xtr_df.iloc[tr_idx].values; y_tr=ytr[tr_idx]
    X_va=Xtr_df.iloc[va_idx].values; y_va=ytr[va_idx]
    X_te=Xte_df[feat_cols].values

    pos=float(y_tr.sum()); neg=float(len(y_tr)-pos)
    w0=1.0; w1=(neg/max(1.0,pos)) if pos>0 else 1.0

    train_pool=Pool(X_tr,y_tr)
    val_pool=Pool(X_va,y_va)

    model=CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=args.iterations,
        learning_rate=args.lr,
        depth=args.depth,
        l2_leaf_reg=args.l2,
        border_count=254,
        random_seed=args.seed,
        verbose=False,
        class_weights=[w0,w1],
        use_best_model=True
    )
    model.fit(train_pool,eval_set=val_pool,early_stopping_rounds=args.early_stopping,verbose=False)

    p_val=model.predict_proba(X_va)[:,1]
    T=fit_temperature(p_val,y_va)
    z_val=logit(p_val); p_val_cal=sigmoid(z_val/T)

    chosen=pick_threshold_max_precision_at_recall(y_va,p_val_cal,args.recall_target)
    tau=chosen["threshold"]

    val_auc=roc_auc_score(y_va,p_val_cal)
    val_ap=average_precision_score(y_va,p_val_cal)
    val_metrics={**chosen,"AUC":float(val_auc),"PR_AUC":float(val_ap)}

    p_test=model.predict_proba(X_te)[:,1]
    p_test_cal=sigmoid(logit(p_test)/T)
    test_auc=roc_auc_score(yte,p_test_cal)
    test_ap=average_precision_score(yte,p_test_cal)
    test_metrics={**metrics_at_threshold(yte,p_test_cal,tau),"AUC":float(test_auc),"PR_AUC":float(test_ap)}

    os.makedirs(args.outdir,exist_ok=True)
    model_path=os.path.join(args.outdir,"catboost_model.cbm")
    meta_path=os.path.join(args.outdir,"catboost_meta.json")
    metrics_path=os.path.join(args.outdir,"metrics_catboost.json")
    model.save_model(model_path)
    with open(meta_path,"w") as f:
        json.dump({"feature_cols":feat_cols,"threshold":float(tau),"recall_target":float(args.recall_target),"temperature":float(T)},f,indent=2)
    with open(metrics_path,"w") as f:
        json.dump({"val_at_threshold":val_metrics,"test_at_threshold":test_metrics},f,indent=2)

    print(f"[VAL] AUC={val_auc:.4f} PR_AUC={val_ap:.4f} | rule={chosen['rule']} | τ={tau:.3f} | Prec={val_metrics['PREC']:.3f} Rec={val_metrics['REC']:.3f} F1={val_metrics['F1']:.3f}")
    print(f"[TEST] AUC={test_auc:.4f} PR_AUC={test_ap:.4f} @ τ={tau:.3f} | Prec={test_metrics['PREC']:.3f} Rec={test_metrics['REC']:.3f} F1={test_metrics['F1']:.3f} Acc={test_metrics['ACC']:.3f}")
if __name__=="__main__": main()