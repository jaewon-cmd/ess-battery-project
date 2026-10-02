"""분할, 모델 파이프라인, 평가 지표, 성능표(Format) 만들기.

평가 원칙
1. 분할을 가장 먼저 한다. 스케일링·결측 대치는 Pipeline 안에서 '학습 데이터로만 fit' 한다.
2. Batch 2(Test)는 모델을 정한 뒤 마지막에 한 번만 평가한다. 모델 선택·튜닝에는 Batch 1(CV, Hold-out)만 쓴다.
3. 같은 충전 프로토콜의 셀이 학습/검증에 나뉘지 않도록 '프로토콜 단위'로 분할한다 (누수 방지).
4. 무작위성이 있는 모든 곳에 RANDOM_STATE 를 고정한다.
"""
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET = 'cycle_life'
PAPER_TARGET_MAPE = 9.1            # 원논문 Regression 성능 (MAPE %)
CORE_FEATURES = ['dq_logvar']      # EDA 에서 정한 핵심 피처


class PLS1D(PLSRegression):
    """PLSRegression 의 예측을 1차원으로 반환 (TransformedTargetRegressor 와 함께 쓰기 위한 래퍼)"""

    def predict(self, X, copy=True):
        return np.ravel(super().predict(X, copy=copy))


def mape(y_true, y_pred):
    """MAPE (%) — 수명(원래 단위)의 상대오차"""
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean(np.abs(y_true - np.asarray(y_pred)) / y_true) * 100)


def get_labeled(df, batch):
    """수명이 있는 셀만 (censored 제외) 해당 배치에서 가져온다."""
    return df[(df.batch == batch) & (~df.censored)].reset_index(drop=True)


def split_holdout(df_b1, test_size=0.2, random_state=RANDOM_STATE):
    """Batch 1 을 '프로토콜 단위'로 학습 / Hold-out 으로 나눈다 (같은 프로토콜은 한쪽에만)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, ho_idx = next(gss.split(df_b1, groups=df_b1['policy']))
    return df_b1.iloc[tr_idx].reset_index(drop=True), df_b1.iloc[ho_idx].reset_index(drop=True)


def group_cv(df_train, n_splits=5):
    """학습 데이터 안의 교차검증 폴드 (프로토콜 단위 GroupKFold). (train_idx, valid_idx) 를 낸다."""
    return list(GroupKFold(n_splits=n_splits).split(df_train, groups=df_train['policy']))


def build_pipeline(model=None):
    """결측 대치 -> 스케일링 -> 모델. 타깃은 log10 로 학습하고 예측은 원래 단위로 되돌린다.
    Pipeline 이므로 CV 의 각 폴드에서 대치·스케일링이 '그 폴드의 학습 부분'으로만 fit 된다."""
    model = model if model is not None else ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=50000)
    pipe = Pipeline([('impute', SimpleImputer(strategy='median')),
                     ('scale', StandardScaler()),
                     ('model', model)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log10, inverse_func=lambda v: np.power(10.0, v))


def cv_mape(df_train, features, model=None, n_splits=5):
    """학습 데이터 안의 GroupKFold CV 평균 MAPE (Train (Batch 1 CV) 칸)."""
    scores = []
    for tr, va in group_cv(df_train, n_splits):
        est = build_pipeline(model).fit(df_train.loc[tr, features], df_train.loc[tr, TARGET])
        scores.append(mape(df_train.loc[va, TARGET], est.predict(df_train.loc[va, features])))
    return float(np.mean(scores)), scores


def holdout_mape(df_train, df_holdout, features, model=None):
    """학습 데이터로 fit 한 뒤 Hold-out 셀의 MAPE (Valid (Batch 1 Hold-out) 칸)."""
    est = build_pipeline(model).fit(df_train[features], df_train[TARGET])
    return mape(df_holdout[TARGET], est.predict(df_holdout[features]))


def mean_baseline(df_train, df_eval):
    """아무 피처도 쓰지 않고 학습 셀 수명의 기하평균(log10 평균)만 예측했을 때의 MAPE. 모델이 이보다 나아야 의미가 있다."""
    pred = 10 ** np.log10(df_train[TARGET]).mean()
    return mape(df_eval[TARGET], np.full(len(df_eval), pred))


def shift_validation(df, features, model=None, q=0.25):
    """배치 이동을 흉내 낸 검증 (한 배치 안에서). 수명이 짧은 쪽 / 긴 쪽 q 분위 셀을 평가용으로 빼고,
    같은 충전 프로토콜의 셀도 학습에서 뺀 뒤 나머지로 학습해 평가한다. (짧은 쪽 MAPE, 긴 쪽 MAPE, 평균) 을 돌려준다."""
    lo, hi = df[TARGET].quantile([q, 1 - q])
    out = {}
    for name, mask in [('짧은 쪽', df[TARGET] <= lo), ('긴 쪽', df[TARGET] >= hi)]:
        te = df[mask]
        tr = df[~mask & ~df['policy'].isin(te['policy'])]
        est = build_pipeline(model).fit(tr[features], tr[TARGET])
        out[name] = mape(te[TARGET], est.predict(te[features]))
    out['평균'] = float(np.mean([out['짧은 쪽'], out['긴 쪽']]))
    return out


def fit_final(df_fit, features=None, model=None):
    """최종 모델을 df_fit(Batch 1 의 수명이 있는 셀)으로 학습한다. 하이퍼파라미터는 고정."""
    features = features or CORE_FEATURES
    return build_pipeline(model).fit(df_fit[features], df_fit[TARGET])


def evaluate_on(est, df_test, features=None):
    """학습된 모델로 df_test 를 예측해 (MAPE, 셀별 오차표) 를 돌려준다.
    err_pct = (예측 - 실제) / 실제 x 100  ->  (+) 는 과대예측, (-) 는 과소예측."""
    features = features or CORE_FEATURES
    out = df_test[['cid', 'batch', 'policy', 'newstructure', 'label_uncertain', TARGET] + features].copy()
    out['pred'] = est.predict(df_test[features])
    out['err_pct'] = (out['pred'] - out[TARGET]) / out[TARGET] * 100
    out['abs_err_pct'] = out['err_pct'].abs()
    return mape(out[TARGET], out['pred']), out.sort_values('abs_err_pct', ascending=False).reset_index(drop=True)


def performance_table(train_cv, valid_holdout, test_b2, test_b3=None, target=PAPER_TARGET_MAPE):
    """과제 Format 의 성능표 (Regression, MAPE 기준).
    Gap 은 'Format 의 (+) 설명이 맞도록' 오차가 커질수록 (+) 가 되게 정의한다 (뒤 칸 - 앞 칸, Target 은 Test - Target).
    - Gap (Train-Valid)  = Valid - Train   (+) : 과적합 의심
    - Gap (Valid-Test)   = Test - Valid    (+) : 배치간 일반화 저하 의심
    - Gap (Target-Test)  = Test - Target   (+) : 원논문보다 오차가 큼"""
    rows = [('Train (Batch 1 CV)', train_cv, ''),
            ('Valid (Batch 1 Hold-out)', valid_holdout, ''),
            ('Test (Batch 2)', test_b2, ''),
            ('Gap (Train-Valid)', valid_holdout - train_cv, '(+) : 과적합 의심'),
            ('Gap (Valid-Test)', test_b2 - valid_holdout, '(+) : 배치간 일반화 저하 의심'),
            ('Gap (Target-Test)', test_b2 - target, f'(+) : 원논문보다 오차가 큼 / Target : 원논문 {target}%')]
    if test_b3 is not None:
        rows += [('Test (Batch 3)', test_b3, '추가 검증 (선택)'),
                 ('Gap (Batch2-Batch3)', test_b3 - test_b2, '(+) : Batch 3 의 오차가 더 큼 (Test 성능 간 비교)'),
                 ('Gap (Target-Test) [Batch 3]', test_b3 - target, 'Batch 3 기준, 원논문 성능 비교')]
    return pd.DataFrame(rows, columns=['구분', 'MAPE (%)', '비고']).round({'MAPE (%)': 2})
