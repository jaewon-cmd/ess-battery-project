# data/

원본 데이터는 용량이 커서(약 7.9GB) 저장소에 올리지 않았습니다. 아래에서 받아 이 폴더(`data/`)에 넣어 주세요.

- 출처: [Kaggle — data-driven-prediction-of-battery-cycle](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)
- 원논문: Severson et al. (2019), *Nature Energy* 4, 383–391

| 파일 | 배치 | 크기 | 용도 |
|---|---|---|---|
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | Batch 1 | 2.8GB | 학습 |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | Batch 2 | 1.9GB | 테스트 |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | Batch 3 | 3.0GB | 추가 검증 (선택) |
| `2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat` | (사용 안 함) | 0.1GB | 다른 논문의 충전 최적화 실험 |

## 캐시 (`data/cache/`)
`.mat` 을 매번 읽으면 오래 걸리므로, 처음 한 번 필요한 부분(요약값 전체 + 초기 100 사이클의 `Qdlin`)만 뽑아 `data/cache/b1.pkl`, `b2.pkl`, `b3.pkl` 로 저장합니다.
`src/preprocess.py` 의 `load_cells()` 가 자동으로 만들고(배치당 약 20~30초), 이후에는 캐시를 사용합니다. 캐시는 저장소에 포함하지 않습니다.

## 분석 대상 정리
- 별도 실험 셀 8개(VarCharge, SLOWCYCLE; 모두 Batch 2)는 제외
- 수명 미도달 2셀(Batch 3)은 수명 라벨이 없어 학습·평가에서 제외
- 결과: Batch 1 46셀 / Batch 2 39셀 / Batch 3 44셀 (수명 있는 셀 기준)
