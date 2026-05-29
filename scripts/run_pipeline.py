import sys
import os
import yaml
import pandas as pd
import numpy as np

# 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.loading import LottoDataEngine
from src.models.baseline import LottoBaseline
from src.models.deep_model import LottoModelEngine
from src.analysis.evaluation import LottoValidationEngine

def main():
    # 1. Config 로드
    with open('config/analysis_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*50)
    print("      Fortune Accelerator Pipeline Starting...")
    print("="*50)

    # 2. 데이터 수집 및 전처리 (Session 4/5)
    engine = LottoDataEngine(config['paths']['raw_data'])
    engine.sync_data()
    features = engine.get_features()
    
    # 3. 데이터 분할
    WINDOW_SIZE = config['data']['window_size']
    train_limit = len(features) - 20
    X, y = [], []
    for i in range(WINDOW_SIZE, len(features)):
        X.append(features[i-WINDOW_SIZE:i])
        y.append(features[i, :45])
    X, y = np.array(X), np.array(y)
    
    X_train, X_test = X[:-20], X[-20:]
    y_train, y_test = y[:-20], y[-20:]
    
    # 4. Baseline 모델링 (Session 6)
    # MLP를 위해 3D 데이터를 2D로 평탄화
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    baseline = LottoBaseline()
    baseline.train(X_train_flat, y_train)
    base_metrics = baseline.evaluate(X_test_flat, y_test)
    print(f"[Results] Baseline F1: {base_metrics['baseline_f1']:.4f}")
    
    # 5. Deep Learning 모델링 (Session 7)
    print("\n[Deep Learning] Training Bi-LSTM...")
    deep_model = LottoModelEngine(window_size=WINDOW_SIZE, feature_dim=X_train.shape[2])
    deep_model.train(X_train, y_train, epochs=config['deep_learning']['epochs'])
    
    # 6. 검증 및 결과 저장
    validator = LottoValidationEngine()
    hits = validator.backtest(deep_model, features, WINDOW_SIZE, 20)
    avg_hits = np.mean(hits)
    print(f"[Results] Deep Learning Avg Hits: {avg_hits:.2f}")

    print("\n" + "="*50)
    print("      Pipeline Completed Successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
