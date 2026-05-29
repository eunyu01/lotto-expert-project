import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_engine import LottoDataEngine
from model_engine import LottoModelEngine
from validation_engine import LottoValidationEngine
import sys

# ---------------------------------------------------------
# 配置区
# ---------------------------------------------------------
WINDOW_SIZE = 12
FILE_PATH = 'lotto_history.csv'
BACKTEST_ROUNDS = 20
STRESS_TEST_MODE = False  # 如果设为 True，将使用纯随机数据进行压力测试

def generate_smart_combinations(probs, count=5):
    combinations = []
    attempts = 0
    
    # 归一化概率
    p = probs / np.sum(probs)
    
    while len(combinations) < count and attempts < 1000:
        attempts += 1
        # 基于概率分布进行加权采样，模拟冷热码混合
        indices = np.random.choice(45, size=6, replace=False, p=p)
        nums = sorted(indices + 1)
        
        # --- 过滤逻辑 (避免特定规律) ---
        consecutive = 0
        for i in range(len(nums)-1):
            if nums[i+1] - nums[i] == 1:
                consecutive += 1
        if consecutive > 2: continue
        
        odds = len([x for x in nums if x % 2 != 0])
        if odds == 0 or odds == 6: continue
        
        total_sum = sum(nums)
        if total_sum < 90 or total_sum > 190: continue
        
        tails = [x % 10 for x in nums]
        if any(tails.count(t) > 3 for t in tails): continue

        if nums not in combinations:
            combinations.append(nums)
            
    return combinations

def run_pipeline(features, title_suffix=""):
    # 构造训练集
    train_limit = len(features) - BACKTEST_ROUNDS
    X_train, y_train = [], []
    for i in range(WINDOW_SIZE, train_limit):
        X_train.append(features[i-WINDOW_SIZE:i])
        y_train.append(features[i, :45])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # 模型训练
    model = LottoModelEngine(window_size=WINDOW_SIZE, feature_dim=X_train.shape[2])
    model.train(X_train, y_train, epochs=60)
    
    # 回测验证
    validator = LottoValidationEngine()
    hits_history = validator.backtest(model, features, WINDOW_SIZE, BACKTEST_ROUNDS)
    
    # 最终预测
    print(f"\n[最终阶段] 正在基于最新趋势生成多样化组合{title_suffix}...")
    X_all, y_all = [], []
    for i in range(WINDOW_SIZE, len(features)):
        X_all.append(features[i-WINDOW_SIZE:i])
        y_all.append(features[i, :45])
    model.train(np.array(X_all), np.array(y_all), epochs=10)
    
    last_seq = features[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, -1)
    probs = model.predict(last_seq)[0]
    
    # 生成组合
    smart_combos = generate_smart_combinations(probs, count=5)
    
    # 绘图 (注意: 如果是压力测试，文件名可以区分一下)
    report_name = f"lotto_report{title_suffix.lower().replace(' ', '_')}.png"
    validator.plot_insights(hits_history, probs) # 注意：原来的 plot_insights 内部可能固定了文件名，这里暂时维持原样
    
    return smart_combos, hits_history

def main():
    print("="*50)
    print("      发财加速器 (Fortune Accelerator) - AI/Data Lab")
    print("="*50)

    engine = LottoDataEngine(FILE_PATH)
    
    if STRESS_TEST_MODE:
        print("\n[警告] 当前处于压力测试模式：使用纯随机数据验证模型！")
        synthetic_data = engine.generate_synthetic_data(num_rounds=1000)
        features = engine.get_features(custom_data=synthetic_data)
        title = " (Stress Test)"
    else:
        # 正常同步数据
        engine.sync_data()
        features = engine.get_features()
        title = ""

    smart_combos, hits_history = run_pipeline(features, title)
    
    print("\n" + "!"*50)
    print(f"推荐组合 (基于 Bi-LSTM 概率分布 + 专家策略过滤){title}:")
    for i, combo in enumerate(smart_combos):
        print(f"  组合 {i+1}: {combo}")
    print("!"*50 + "\n")
    
    avg_hits = np.mean(hits_history)
    print(f"平均命中数: {avg_hits:.2f} (通常随机水平为 0.8~1.0)")
    
    if STRESS_TEST_MODE:
        if avg_hits > 1.2:
            print(">>> 结论: 模型在随机数据中表现过强，可能存在严重的过拟合！请增加 Dropout。")
        else:
            print(">>> 结论: 模型在随机数据中表现符合预期，具备基础的稳健性。")

if __name__ == "__main__":
    main()
