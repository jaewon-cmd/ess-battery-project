"""셀 단위 피처 표 만들기 (EDA 에서 확인한 피처 정의).

피처 (초기 100 사이클 기준)
- dq_logvar, dq_logmean, dq_logmin, dq_skew, dq_kurt : ΔQ(V) = Qdlin[100] - Qdlin[10] 의 통계 (log10)
- QD_slope : 사이클 2~100 방전 용량 기울기 (mAh/cycle),  IR / Tavg / Tmax / chargetime : 사이클 2~100 중앙값
- C1, switch, C2, avgC : 충전 프로토콜 `C1(Q1%)-C2` 에서 분해 (avgC 는 0->80% 구간 평균 C-rate)

주의: 이 모듈은 '셀 하나' 안에서만 계산하므로 다른 셀·다른 배치의 정보를 쓰지 않는다.
      (스케일링·결측 대치처럼 데이터 전체의 통계가 필요한 변환은 train.py 의 Pipeline 에서 학습 데이터로만 fit 한다.)
"""
import re

import numpy as np
import pandas as pd
from scipy import stats

EOL_Q = 0.88    # 공칭 1.1 Ah 의 80%
META_COLS = ['cid', 'batch', 'policy', 'newstructure', 'cycle_life', 'censored', 'end_QD', 'label_uncertain']


def clean_qd(qd, lo=0.7, hi=1.3, k=5):
    """방전 용량 스파이크 제거: 범위 밖 값은 NaN 처리 후 이동 중앙값"""
    qd = np.asarray(qd, dtype=float).copy()
    qd[(qd < lo) | (qd > hi)] = np.nan
    return pd.Series(qd).rolling(k, center=True, min_periods=1).median().to_numpy()


def delta_q(cell, hi=100, lo=10):
    """ΔQ(V) = Qdlin[hi] - Qdlin[lo]  (같은 셀 안에서의 차이 -> 배치 간 Qdlin 시작점 차이가 상쇄됨)"""
    return cell['Qdlin'][hi] - cell['Qdlin'][lo]


def dq_features(dq):
    dq = dq[np.isfinite(dq)]
    return {'dq_logvar': np.log10(np.var(dq)),
            'dq_logmean': np.log10(abs(np.mean(dq))),
            'dq_logmin': np.log10(abs(np.min(dq))),
            'dq_skew': stats.skew(dq), 'dq_kurt': stats.kurtosis(dq)}


# ΔQ(V) 를 전압 구간별로 나눈 분산 (구간 단위: V). Qdlin 의 전압 축은 3.6 V -> 2.0 V 를 1,000 포인트로 보간한 것.
V_AXIS = np.linspace(3.6, 2.0, 1000)
V_BINS = [(3.3, 3.1), (3.1, 2.9), (2.9, 2.6), (2.6, 2.0)]


def dq_bin_features(dq):
    """ΔQ(V) 곡선을 전압 구간별로 나눠 각 구간의 분산(log10)을 만든다. (단수명·장수명의 차이가 큰 구간이 따로 있는지 확인용)"""
    out = {}
    for hi, lo in V_BINS:
        seg = dq[(V_AXIS <= hi) & (V_AXIS > lo)]
        out[f'dqv_{round(hi * 10)}_{round(lo * 10)}'] = np.log10(np.nanvar(seg) + 1e-12)
    return out


def summary_features(cell, n=100):
    s = cell['summary']
    m = (s['cycle'] >= 2) & (s['cycle'] <= n)
    qd = clean_qd(s['QDischarge'])[m]
    x = s['cycle'][m]
    ok = np.isfinite(qd)
    ir = s['IR'][m]
    ir = ir[ir > 0]                                      # IR=0 은 측정 누락 -> 제외 (전부 0이면 NaN)
    return {'QD_slope': np.polyfit(x[ok], qd[ok], 1)[0] * 1000,
            'IR': np.median(ir) if ir.size else np.nan,
            'Tavg': np.nanmedian(s['Tavg'][m]), 'Tmax': np.nanmedian(s['Tmax'][m]),
            'chargetime': np.nanmedian(s['chargetime'][m])}


def protocol_features(policy):
    mm = re.match(r'^([\d.]+)C\((\d+)%\)-([\d.]+)C', policy)
    c1, sw, c2 = float(mm.group(1)), float(mm.group(2)), float(mm.group(3))
    t = sw / 100 / c1 + (80 - sw) / 100 / c2             # 0->80% 충전 시간(h)
    return {'C1': c1, 'switch': sw, 'C2': c2, 'avgC': 0.8 / t}


def build_feature_table(cohort):
    """cohort(list of cell dict) -> 셀당 한 행의 피처 표 (DataFrame)"""
    rows = []
    for c in cohort:
        r = {k: c[k] for k in META_COLS}
        dq = delta_q(c)
        if np.isfinite(dq).all():
            r.update(dq_features(dq))
            r.update(dq_bin_features(dq))
        r.update(summary_features(c))
        r.update(protocol_features(c['policy']))
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    from src.preprocess import build_cohort, load_cells
    df = build_feature_table(build_cohort(load_cells()))
    print(df.shape)
    print(df.groupby('batch').size().to_dict())
