"""후보 모델 정의 (03_modeling 에서 사용).
모든 모델은 factory(함수)로 만들어, 호출할 때마다 새 객체를 돌려준다. 무작위성이 있는 모델은 RANDOM_STATE 고정."""
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
from xgboost import XGBRegressor

from src.train import PLS1D, RANDOM_STATE as RS


def get_models():
    return {
        'ElasticNet':       lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RS, max_iter=50000),
        'Ridge':            lambda: Ridge(alpha=1.0, random_state=RS),
        'Huber':            lambda: HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=1000),
        'PLS':              lambda: PLS1D(n_components=1),
        'RandomForest':     lambda: RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=3, random_state=RS),
        'GradientBoosting': lambda: GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=RS),
        'XGBoost':          lambda: XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=RS, n_jobs=1),
        'LightGBM':         lambda: LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, num_leaves=4, min_child_samples=3,
                                                  subsample=0.8, subsample_freq=1, random_state=RS, n_jobs=1, verbose=-1),
        'GaussianProcess':  lambda: GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1), normalize_y=True, random_state=RS),
    }


def average_ensemble(names, models=None):
    """선택한 모델들의 예측을 단순 평균하는 앙상블 factory (log10 스케일의 예측을 평균 → build_pipeline 의 타깃 변환과 함께 쓴다)."""
    models = models or get_models()
    return lambda: VotingRegressor([(n, models[n]()) for n in names])
