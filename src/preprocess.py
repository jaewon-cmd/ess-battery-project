"""원본 .mat -> 분석용 캐시(pkl) 변환, 분석 대상 셀 정의.

사용:
    from src.preprocess import load_cells, build_cohort
    cells = load_cells()                # 3개 배치 전체 (처음 한 번만 오래 걸림, 이후 캐시 사용)
    cohort = build_cohort(cells)        # 표준 프로토콜 셀 (별도 실험 제외) + censored 표시
"""
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
CACHE_DIR = DATA_DIR / 'cache'

FILES = {
    'b1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',   # Batch 1 (학습)
    'b2': '2018-02-20_batchdata_updated_struct_errorcorrect.mat',   # Batch 2 (테스트)
    'b3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat',   # Batch 3 (추가 검증)
}
SUMMARY_KEYS = ['IR', 'QCharge', 'QDischarge', 'Tavg', 'Tmax', 'Tmin', 'chargetime', 'cycle']
LABEL_UNCERTAIN_QD = 0.95   # 이 값 이상에서 기록이 끝난 셀은 '라벨 불확실' 로 표시
N_EARLY = 101        # 사이클 인덱스 0~100 (ΔQ(V) = Qdlin[100] - Qdlin[10] 에 필요한 범위)


def _pad(arrs, n=1000):
    """길이가 n 이 아닌 사이클(비어 있는 사이클 0 등)은 NaN 행으로 채움"""
    out = np.full((len(arrs), n), np.nan)
    for j, a in enumerate(arrs):
        a = np.asarray(a, dtype=float).ravel()
        if a.size == n:
            out[j] = a
    return out


def load_batch(tag):
    """배치 하나를 불러온다. 캐시가 있으면 캐시를 쓴다."""
    path = CACHE_DIR / f'{tag}.pkl'
    if path.exists():
        return pickle.load(open(path, 'rb'))
    import mat73                                       # 원본을 읽을 때만 필요
    logging.disable(logging.CRITICAL)                  # mat73 의 'string type not supported' 경고 (결과에 영향 없음)
    warnings.filterwarnings('ignore')
    b = mat73.loadmat(str(DATA_DIR / FILES[tag]))['batch']
    cells = []
    for i in range(len(b['cycle_life'])):
        cyc = b['cycles'][i]
        cells.append(dict(
            cid=f'{tag}c{i}', batch=tag,
            policy=str(b['policy_readable'][i]),
            cycle_life=float(np.asarray(b['cycle_life'][i]).ravel()[0]),
            summary={k: np.asarray(b['summary'][i][k], dtype=float).ravel() for k in SUMMARY_KEYS},
            Qdlin=_pad(cyc['Qdlin'][:N_EARLY]),
        ))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(cells, open(path, 'wb'))
    return cells


def load_cells(batches=('b1', 'b2', 'b3')):
    cells = []
    for t in batches:
        cells += load_batch(t)
    return cells


def build_cohort(cells):
    """분석 대상: 표준 충전 프로토콜 셀 (VarCharge / SLOWCYCLE 같은 별도 실험 제외).
    cycle_life 가 비어 있는 셀은 EOL 미도달(right-censored) 이므로 수명을 채우지 않고 'censored' 로 표시한다.
    (모델 학습·평가에서는 censored 셀을 제외한다.)"""
    keep = []
    for c in cells:
        p = c['policy']
        if p.startswith('VarCharge') or 'SLOWCYCLE' in p:
            continue
        c = dict(c)
        c['newstructure'] = 'newstructure' in p
        c['policy'] = p.replace('-newstructure', '')
        c['censored'] = bool(np.isnan(c['cycle_life']))
        q = c['summary']['QDischarge']
        q = q[np.isfinite(q)]
        c['end_QD'] = float(np.median(q[-10:]))                          # 기록 마지막 구간의 방전 용량
        # 수명 라벨이 있는데 기록상 용량이 0.95 Ah 이상에서 끝나는 셀 -> 기록만으로는 EOL(0.88 Ah) 라벨을 검증할 수 없음
        c['label_uncertain'] = (not c['censored']) and c['end_QD'] > LABEL_UNCERTAIN_QD
        keep.append(c)
    return keep


if __name__ == '__main__':
    cohort = build_cohort(load_cells())
    print('분석 대상 셀 수:', len(cohort), '| censored:', sum(c['censored'] for c in cohort))
