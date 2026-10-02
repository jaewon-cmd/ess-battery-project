"""README 용 그림 생성 (results/features.csv 와 최종 모델만 사용). 실행: python -m src.make_figures"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.train import CORE_FEATURES, TARGET, evaluate_on, fit_final, get_labeled

plt.rcParams['font.family'] = ['AppleGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
OUT = 'results/figures/'
COL = {'b1': '#4c78a8', 'b2': '#e45756', 'b3': '#54a24b'}

df = pd.read_csv('results/features.csv')
b1, b2, b3 = (get_labeled(df, b) for b in ['b1', 'b2', 'b3'])

# 1) dq_logvar vs 수명 (배치별)
fig, ax = plt.subplots(figsize=(6, 4))
for b, d in [('b1', b1), ('b2', b2), ('b3', b3)]:
    ax.scatter(d['dq_logvar'], d[TARGET], s=22, c=COL[b], label=f'Batch {b[1]}', alpha=.8)
ax.set_xlabel('dq_logvar (ΔQ 분산, log10)'); ax.set_ylabel('Cycle Life'); ax.set_yscale('log')
ax.set_title('핵심 피처와 수명: 세 배치 모두 같은 방향'); ax.legend(); fig.tight_layout()
fig.savefig(OUT + 'dq_logvar_vs_life.png', dpi=150); plt.close(fig)

# 2) 예측 vs 실제 (Batch 2, 3)
est = fit_final(b1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
for ax, b, d in [(axes[0], 'b2', b2), (axes[1], 'b3', b3)]:
    m, r = evaluate_on(est, d)
    n = r['newstructure'].astype(bool) if b == 'b2' else pd.Series(False, index=r.index)
    ax.scatter(r.loc[~n, TARGET], r.loc[~n, 'pred'], s=24, c=COL['b2'] if b == 'b2' else COL['b3'], label='기존구조' if b == 'b2' else 'Batch 3 셀')
    if n.any():
        ax.scatter(r.loc[n, TARGET], r.loc[n, 'pred'], s=24, c='#f58518', marker='s', label='신규구조')
    ax.plot([300, 2000], [300, 2000], 'k--', lw=1)
    ax.set_title(f'Batch {b[1]}  MAPE {m:.1f}%'); ax.set_xlabel('실제 수명'); ax.legend()
axes[0].set_ylabel('예측 수명'); fig.tight_layout()
fig.savefig(OUT + 'pred_vs_actual.png', dpi=150); plt.close(fig)

# 3) 성능 막대
p = pd.read_csv('results/model_performance.csv').set_index('구분')['MAPE (%)']
labels = ['Train\n(B1 CV)', 'Valid\n(B1 Hold-out)', 'Test\n(Batch 2)', 'Test\n(Batch 3)']
vals = [p['Train (Batch 1 CV)'], p['Valid (Batch 1 Hold-out)'], p['Test (Batch 2)'], p['Test (Batch 3)']]
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels, vals, color=[COL['b1'], COL['b1'], COL['b2'], COL['b3']])
ax.axhline(9.1, color='k', ls='--', lw=1); ax.text(-0.45, 10.2, '원논문 9.1%', ha='left', fontsize=9)
for b_, v in zip(bars, vals):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 1.0, f'{v:.1f}', ha='center')
ax.set_ylabel('MAPE (%)'); ax.set_title('단계별 오차'); fig.tight_layout()
fig.savefig(OUT + 'performance.png', dpi=150); plt.close(fig)
