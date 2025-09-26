import argparse, os, json, numpy as np, pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix, average_precision_score

class TabDS(Dataset):
    def __init__(self, X, y): self.X=torch.tensor(X,dtype=torch.float32); self.y=torch.tensor(y,dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self,i): return self.X[i], self.y[i]

class NumericTabTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=8, layers=2, dim_ff=192, p=0.1, use_cls=True):
        super().__init__()
        self.use_cls=use_cls
        self.scalar_proj=nn.Linear(1,d_model)
        self.col_embed=nn.Parameter(torch.randn(num_features,d_model)*0.02)
        if use_cls:
            self.cls_token=nn.Parameter(torch.randn(1,1,d_model)*0.02)
            self.cls_pos  =nn.Parameter(torch.randn(1,1,d_model)*0.02)
        enc=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dim_ff,dropout=p,batch_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(enc,layers)
        self.norm=nn.LayerNorm(d_model)
        self.head=nn.Sequential(nn.Linear(d_model,d_model),nn.GELU(),nn.Dropout(p),nn.Linear(d_model,1))
    def forward(self,x):
        B,F=x.shape
        x=self.scalar_proj(x.view(B*F,1)).view(B,F,-1)
        x=x+self.col_embed.unsqueeze(0).expand(B,-1,-1)
        if self.use_cls:
            cls=self.cls_token.expand(B,-1,-1)+self.cls_pos
            x=torch.cat([cls,x],dim=1)
        z=self.encoder(x)
        pooled=z[:,0,:] if self.use_cls else z.mean(dim=1)
        pooled=self.norm(pooled)
        return self.head(pooled).squeeze(1)

def eval_dl(dl, model, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for xb,yb in dl:
            xb,yb=xb.to(device), yb.to(device)
            ps.append(torch.sigmoid(model(xb)).cpu().numpy())
            ys.append(yb.cpu().numpy())
    y=np.concatenate(ys); p=np.concatenate(ps); pred=(p>=0.5).astype(int)
    auc=roc_auc_score(y,p); acc=accuracy_score(y,pred)
    prec,rec,f1,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
    return {"AUC":float(auc),"ACC":float(acc),"PREC":float(prec),"REC":float(rec),"F1":float(f1)}

def eval_probs(dl, model, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for xb,yb in dl:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.sigmoid(logits).float().cpu().numpy()
            ys.append(yb.numpy())
            ps.append(prob)
    y = np.concatenate(ys).astype(int)
    p = np.concatenate(ps)
    return y, p

def metrics_at_threshold(y, p, thr):
    yhat = (p >= thr).astype(int)
    prec = precision_score(y, yhat, zero_division=0)
    rec  = recall_score(y, yhat, zero_division=0)
    f1   = f1_score(y, yhat, zero_division=0)
    acc  = accuracy_score(y, yhat)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
    return {"threshold": float(thr), "PREC": float(prec), "REC": float(rec), "F1": float(f1),
            "ACC": float(acc), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

def pick_threshold_max_precision_at_recall(y, p, recall_target=0.85):
    grid = np.linspace(0.01, 0.99, 99)
    candid = []
    for t in grid:
        m = metrics_at_threshold(y, p, t)
        if m["REC"] >= recall_target:
            candid.append(m)
    if candid:
        candid.sort(key=lambda m: (m["PREC"], m["REC"], m["F1"]), reverse=True)
        chosen = candid[0]
        chosen["rule"] = f"max_precision_given_recall≥{recall_target}"
        return chosen
    scored = [metrics_at_threshold(y, p, t) for t in grid]
    chosen = max(scored, key=lambda m: m["F1"])
    chosen["rule"] = "best_F1_fallback"
    return chosen

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_x",default="model_features_train.csv")
    ap.add_argument("--train_y",default="model_labels_train.csv")
    ap.add_argument("--test_x", default="model_features_test.csv")
    ap.add_argument("--test_y", default="model_labels_test.csv")
    ap.add_argument("--epochs",type=int,default=150)
    ap.add_argument("--batch_size",type=int,default=512)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--d_model",type=int,default=64)
    ap.add_argument("--nhead",type=int,default=8)
    ap.add_argument("--layers",type=int,default=2)
    ap.add_argument("--dim_ff",type=int,default=192)
    ap.add_argument("--dropout",type=float,default=0.1)
    ap.add_argument("--use_cls",action="store_true")
    ap.add_argument("--outdir",default="./artifacts")
    ap.add_argument("--recall_target", type=float, default=0.85)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    args=ap.parse_args()

    Xtr_df=pd.read_csv(args.train_x); Xte_df=pd.read_csv(args.test_x)
    ytr=pd.read_csv(args.train_y)["sla_violation"].values.astype(np.float32)
    yte=pd.read_csv(args.test_y)["sla_violation"].values.astype(np.float32)
    feat_cols=Xtr_df.columns.tolist()

    scaler=StandardScaler()
    Xtr=scaler.fit_transform(Xtr_df.values.astype(np.float32))
    Xte=scaler.transform(Xte_df.values.astype(np.float32))

    full=TabDS(Xtr,ytr)
    val_size=int(0.2*len(full)); tr_size=len(full)-val_size
    tr_ds,val_ds=random_split(full,[tr_size,val_size],generator=torch.Generator().manual_seed(42))
    tr_dl=DataLoader(tr_ds,batch_size=args.batch_size,shuffle=True)
    va_dl=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False)
    te_dl=DataLoader(TabDS(Xte,yte),batch_size=args.batch_size,shuffle=False)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=NumericTabTransformer(num_features=Xtr.shape[1],d_model=args.d_model,nhead=args.nhead,layers=args.layers,dim_ff=args.dim_ff,p=args.dropout,use_cls=args.use_cls).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)

    warmup_epochs=max(1, args.warmup_epochs)
    cosine_epochs=max(1, args.epochs - warmup_epochs)
    warmup=LambdaLR(opt, lr_lambda=lambda e: min(1.0, (e + 1) / warmup_epochs))
    cosine=CosineAnnealingLR(opt, T_max=cosine_epochs)
    scheduler=SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    pos=float(ytr.sum()); neg=float(len(ytr)-pos)
    pos_weight=torch.tensor([neg/max(1.0,pos)],dtype=torch.float32).to(device)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc=-1; best=None; patience=5; bad=0
    for ep in range(1,args.epochs+1):
        model.train(); tot=0.0
        for xb,yb in tr_dl:
            xb,yb=xb.to(device), yb.to(device)
            opt.zero_grad()
            loss=lossf(model(xb),yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            tot+=loss.item()*xb.size(0)
        vm=eval_dl(va_dl,model,device)
        print(f"Epoch {ep}/{args.epochs} loss={tot/len(tr_ds):.4f} | val AUC={vm['AUC']:.4f} ACC={vm['ACC']:.4f} F1={vm['F1']:.4f}")
        if vm["AUC"]>best_auc:
            best_auc=vm["AUC"]
            best={k:v.cpu() for k,v in model.state_dict().items()}
            bad=0
        else:
            bad+=1
            if bad>=patience:
                print("Early stop")
                break
        scheduler.step()

    if best is not None:
        model.load_state_dict(best)
    valm=eval_dl(va_dl,model,device)
    testm=eval_dl(te_dl,model,device)

    y_val, p_val = eval_probs(va_dl, model, device)
    val_auc  = roc_auc_score(y_val, p_val)
    val_ap   = average_precision_score(y_val, p_val)
    chosen   = pick_threshold_max_precision_at_recall(y_val, p_val, args.recall_target)
    tau      = chosen["threshold"]
    val_metrics_at_tau = {**chosen, "AUC": float(val_auc), "PR_AUC": float(val_ap)}

    y_test, p_test = eval_probs(te_dl, model, device)
    test_auc = roc_auc_score(y_test, p_test)
    test_ap  = average_precision_score(y_test, p_test)
    test_metrics_at_tau = {**metrics_at_threshold(y_test, p_test, tau),
                           "AUC": float(test_auc), "PR_AUC": float(test_ap)}

    print(f"\n[VAL] AUC={val_auc:.4f} PR_AUC={val_ap:.4f} | rule={chosen['rule']} | τ={tau:.3f} | Prec={chosen['PREC']:.3f} Rec={chosen['REC']:.3f} F1={chosen['F1']:.3f}")
    print(f"[TEST] AUC={test_auc:.4f} PR_AUC={test_ap:.4f} @ τ={tau:.3f} | Prec={test_metrics_at_tau['PREC']:.3f} Rec={test_metrics_at_tau['REC']:.3f} F1={test_metrics_at_tau['F1']:.3f} Acc={test_metrics_at_tau['ACC']:.3f}")

    os.makedirs(args.outdir,exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "scaler_mean_": scaler.mean_,
        "scaler_scale_": scaler.scale_,
        "feature_cols": feat_cols,
        "arch": {"d_model": args.d_model, "nhead": args.nhead, "layers": args.layers,
                 "dim_ff": args.dim_ff, "use_cls": args.use_cls},
        "threshold": float(tau),
        "recall_target": float(args.recall_target)
    }, os.path.join(args.outdir, "tabtransformer_numeric.pt"))

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({
            "val_at_threshold": val_metrics_at_tau,
            "test_at_threshold": test_metrics_at_tau
        }, f, indent=2)


if __name__=="__main__": main()
