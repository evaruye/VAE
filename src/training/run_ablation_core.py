# -*- coding: utf-8 -*-
"""VAE feature-representation ablation runner.

The original four training modules are imported unchanged; their split methods,
model builders and fold fitting methods are invoked. Optuna is NOT rerun.
Only input X is replaced for the two feature ablations. Baseline is the
original full-input cross-validation reference, not recomputed by this program.
"""
from __future__ import annotations
import sys, importlib.util, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=ROOT/'data'/'Data_S2'
SEED=23
TARGET_COLUMN={'EC':'Ethylene Content','SC':'Solid Content','VC':'Viscosity','HCHO':'Free HCHO Content'}
SOURCE_NAMES={'EC':'train_EC_CatBoost','SC':'train_SC_XGBoost','VC':'train_VC_CatBoost','HCHO':'train_HCHO_MLP'}
PROC=['init_temp','poly_pressure','poly_temp','time_log1p','initiator_feed_continuous']
POOL=['oxid_mean_E_LUMO','oxid_mean_f+','red_mean_TPSA','red_mean_E_HOMO','red_mean_V(vdw)',
      'red_mean_f-','ter_mean_E_gap','ter_mean_V(vdw)','emul_mean_EO_number','pc_mean_degree_of_hydrolysis','pc_mean_DP']
# Exact parameters from final original model objects and original model selection; EC/VC
# use Optuna requested iterations, as the original validation-fold early-stopping does.
PARAMS={
 'EC':{'iterations':590,'depth':4,'learning_rate':0.20805863969431487,'l2_leaf_reg':21.527069681462848,
       'random_strength':5.894286727965902,'rsm':0.8680118819084683,'border_count':64,
       'grow_policy':'Depthwise','leaf_estimation_iterations':1,'bootstrap_type':'Bernoulli',
       'min_data_in_leaf':49,'subsample':0.6679983833521714},
 'SC':{'n_estimators':134,'learning_rate':0.09842628256165152,'max_depth':3,'min_child_weight':1.559157706211127,
       'subsample':0.9021077916988999,'colsample_bytree':0.8251901743396407,'gamma':0.21951514981595247,
       'reg_alpha':1.2695020624223357e-06,'reg_lambda':13.159575050198168,'max_bin':512},
 'VC':{'iterations':1178,'depth':3,'learning_rate':0.053207410856167474,'l2_leaf_reg':301.86168836176023,
       'random_strength':68.61320655909958,'rsm':0.41627758111822694,'subsample':0.4885187579136772,
       'border_count':32,'grow_policy':'SymmetricTree','leaf_estimation_iterations':8,
       'bootstrap_type':'Bernoulli','min_data_in_leaf':120,'boosting_type':'Plain'},
 'HCHO':{'hidden_layer_sizes':(128,128,64),'activation':'tanh','alpha':0.0006584614339076447,
         'batch_size':32,'learning_rate_init':0.011542692268549048,'max_iter':2953,
         'tol':0.00017557296287695068,'n_iter_no_change':38,'early_stopping':False}
}
REFERENCE_FULL_CV={'EC':(0.9242,0.0640),'SC':(0.9150,0.0746),'VC':(0.7201,0.1429),'HCHO':(0.9442,0.0466)}

def import_original(target):
    name=SOURCE_NAMES[target]
    spec=importlib.util.spec_from_file_location('vae_original_'+name,HERE/f'{name}.py')
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

def load_original_and_splits(target,mod):
    if target=='EC':
        X,y=mod.build_xy_dropisna(mod.df_no_outlier_delete.copy())
        train,test=train_test_split(np.arange(len(y)),test_size=mod.TEST_SIZE,random_state=mod.RANDOM_SEED,shuffle=True)
        fold_source='KFold(shuffle=True, seed=23)'
    elif target=='SC':
        X,y=mod.build_xy_dropisna(mod.df_no_outlier_delete.copy())
        ybin=mod.make_continuous_stratify_labels(y,max_bins=mod.OUTER_STRATIFY_MAX_BINS,
                  min_bin_size=max(2,int(np.ceil(1/mod.TEST_SIZE))),tag='ABLATION_ORIGINAL_OUTER')
        train,test=train_test_split(np.arange(len(y)),test_size=mod.TEST_SIZE,
                  random_state=mod.RANDOM_SEED,shuffle=True,stratify=ybin)
        fold_source='continuous-quantile StratifiedKFold(shuffle=True, seed=23)'
    elif target=='VC':
        X,y=mod.build_xy_dropisna(mod.df_base.copy())
        train,test,_=mod.robust_train_test_split_indices(len(y),y,mod.TEST_SIZE,mod.RANDOM_SEED)
        fold_source='KFold(shuffle=True, seed=23)'
    else:
        df=pd.read_excel(DATA/'HCHO_Modeling_Data.xlsx').copy().reset_index(drop=True)
        df.insert(0,'row_index',np.arange(len(df)))
        X,y,ynz=mod.build_xy(df,drop_isna_cols=True)
        train,test=train_test_split(np.arange(len(y)),test_size=mod.TEST_SIZE,random_state=mod.RANDOM_SEED,
                                    shuffle=True,stratify=ynz)
        fold_source='zero/nonzero StratifiedKFold(shuffle=True, seed=23)'
    X=X.reset_index(drop=True);y=np.asarray(y,dtype=float)
    ytr=y[train]
    if target=='SC':
        inner_bin=mod.make_continuous_stratify_labels(ytr,max_bins=mod.INNER_STRATIFY_MAX_BINS,
                        min_bin_size=10,tag='ABLATION_ORIGINAL_INNER')
        if inner_bin is not None: folds=list(StratifiedKFold(10,shuffle=True,random_state=SEED).split(train,inner_bin))
        else: folds=list(KFold(10,shuffle=True,random_state=SEED).split(train))
    elif target=='HCHO':
        folds=list(StratifiedKFold(10,shuffle=True,random_state=SEED).split(train,(ytr>0).astype(int)))
    else:
        folds=list(KFold(10,shuffle=True,random_state=SEED).split(train))
    return X,y,np.asarray(train),np.asarray(test),folds,fold_source

def load_ablation_inputs(target,scenario,X_orig,y):
    workbook_name={'process_only':'Process_Only.xlsx','equal_pooling':'Equal_Weight_Pooling.xlsx'}[scenario]
    filename=DATA/workbook_name
    df=pd.read_excel(filename,sheet_name=target)
    if len(df)!=len(y): raise ValueError(f'{filename}: n mismatch {len(df)} vs {len(y)}')
    target_column=TARGET_COLUMN[target]
    y_input=pd.to_numeric(df[target_column],errors='raise').to_numpy(float)
    if not np.allclose(y_input,y,rtol=1e-12,atol=1e-12,equal_nan=False):
        raise ValueError(f'{filename}: y/row order/transform different from original source!')
    Xin=df.drop(columns=[target_column])
    if scenario=='process_only':
        expected=PROC+(['initiator_feed_semi_continuous'] if target=='HCHO' else [])
        if list(Xin.columns)!=expected:raise ValueError('Process-only feature list differs from original')
        # IMPORTANT: take the ORIGINAL XLSX float values; a source-table conversion can
        # perturb the last bit and even cause MLP optimization to follow a
        # different convergence path (especially for process-only HCHO).
        if not np.allclose(Xin.to_numpy(float),X_orig[expected].to_numpy(float),atol=1e-12,rtol=1e-12,equal_nan=True):
            raise ValueError('Process-only workbook differs numerically from original source')
        return X_orig[expected].copy()
    if list(Xin.columns)!=list(X_orig.columns):
        raise ValueError('Equal-pooling feature names/order differ from original')
    unchanged=[c for c in Xin.columns if c not in POOL]
    orig=X_orig[unchanged].to_numpy(float);other=Xin[unchanged].to_numpy(float)
    if not np.allclose(orig,other,atol=1e-10,rtol=1e-10,equal_nan=True):
        raise ValueError('Non-pooling features unexpectedly differ')
    # Keep all untouched columns at their exact original XLSX values;
    # replace ONLY the 11 pooled descriptor columns, as required by ablation.
    result=X_orig.copy()
    result.loc[:,POOL]=Xin[POOL].to_numpy(float)
    return result

def fit_original_fold(target,mod,Xfit,yfit,Xval,yval):
    p=PARAMS[target]
    if target=='EC':
        reg,info=mod.fit_catboost_cv_fold(p,Xfit,yfit,Xval,yval,tag='ABLATION_EC')
        pred=reg.predict(Xval); best=info['best_iterations_es']
    elif target=='VC':
        reg,info=mod.fit_catboost_with_eval(p,Xfit,yfit,Xval,yval,tag='ABLATION_VC',random_seed_override=SEED)
        pred=reg.predict(Xval); best=info['best_iterations_es']
    else:
        reg=mod.build_model(p)
        reg.fit(Xfit,yfit)
        pred=reg.predict(Xval)
        best=None
        if target=='HCHO':pred=mod.clip_nonneg(pred)
    return np.asarray(pred,dtype=float).reshape(-1),best

def main(target,scenario):
    if target not in TARGET_COLUMN or scenario not in ['process_only','equal_pooling']:
        raise ValueError('Usage: main(EC|SC|VC|HCHO, process_only|equal_pooling)')
    mod=import_original(target)
    X,y,train,test,folds,split_type=load_original_and_splits(target,mod)
    ablation_X=load_ablation_inputs(target,scenario,X,y)
    # The same row indices and validation folds are used regardless of input representation.
    Xtr=ablation_X.iloc[train].reset_index(drop=True)
    ytr=y[train]
    result_dir=ROOT/'results'/'aligned_ablation'/target/scenario
    result_dir.mkdir(parents=True,exist_ok=True)
    fold_rows=[];oof_rows=[];preds=np.full(len(train),np.nan)
    for fold,(i,j) in enumerate(folds,1):
        pv,best=fit_original_fold(target,mod,Xtr.iloc[i],ytr[i],Xtr.iloc[j],ytr[j])
        preds[j]=pv
        true=ytr[j]
        record={'target':target,'scenario':scenario,'fold':fold,'r2':float(r2_score(true,pv)),
              'rmse':float(mean_squared_error(true,pv)**0.5),'mae':float(mean_absolute_error(true,pv)),
              'n_train_fold':int(len(i)),'n_valid_fold':int(len(j)),'best_iteration':best}
        fold_rows.append(record)
        for pos,t in enumerate(j):
            oof_rows.append({'target':target,'scenario':scenario,'fold':fold,'original_row_0based':int(train[t]),
                             'y_true':float(true[pos]),'y_pred':float(pv[pos])})
        print(f'{target}/{scenario} fold {fold}/10: R²={record["r2"]:.5f} best_iter={best}',flush=True)
    if not np.isfinite(preds).all():raise RuntimeError('OOF predictions incomplete')
    scores=pd.DataFrame(fold_rows)
    summary={'target':target,'scenario':scenario,'n_total':len(y),'n_train':len(train),'n_test':len(test),
       'n_features':Xtr.shape[1], 'r2_mean':float(scores.r2.mean()),'r2_sd_sample':float(scores.r2.std(ddof=1)),
       'rmse_mean':float(scores.rmse.mean()),'rmse_sd_sample':float(scores.rmse.std(ddof=1)),
       'mae_mean':float(scores.mae.mean()),'mae_sd_sample':float(scores.mae.std(ddof=1)),
       'oof_r2':float(r2_score(ytr,preds)),
       'reference_full_r2_mean':REFERENCE_FULL_CV[target][0],
       'reference_full_r2_sd':REFERENCE_FULL_CV[target][1],
       'delta_vs_full_reference_mean_R2':float(REFERENCE_FULL_CV[target][0]-scores.r2.mean()),
       'cv_source':split_type,
       'fit_rule':'original target-specific fitting function; fixed original optimal params; original fold early stopping if applicable; no Optuna'}
    scores.to_csv(result_dir/'fold_scores.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(oof_rows).to_csv(result_dir/'oof_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([summary]).to_csv(result_dir/'summary_metrics.csv',index=False,encoding='utf-8-sig')
    with open(result_dir/'run_info.json','w',encoding='utf-8') as f:
        json.dump({'python':sys.version,'platform':platform.platform(),
                   'params':PARAMS[target],'train_indices':train.tolist(),'test_indices':test.tolist(),
                   'package_versions':{'pandas':pd.__version__,'numpy':np.__version__},
                   'description':'Full-input reference retained from fixed baseline metrics; two ablations use original CV logic'},f,indent=2,ensure_ascii=False)
    print(f'\nFINAL: {target} {scenario} R²={summary["r2_mean"]:.4f} ± {summary["r2_sd_sample"]:.4f}')
    print(f'Full-input reference baseline: {summary["reference_full_r2_mean"]:.4f} ± {summary["reference_full_r2_sd"]:.4f}')
    print(f'Output: {result_dir}')
    return summary
