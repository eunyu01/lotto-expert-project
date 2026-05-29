import numpy as np
import pandas as pd
from data_engine import LottoDataEngine
from model_engine import LottoModelEngine
from main import generate_smart_combinations
import time

# ---------------------------------------------------------
# 配置区
# ---------------------------------------------------------
WINDOW_SIZE = 12
FILE_PATH = 'lotto_history.csv'
MAX_ATTEMPTS_PER_ROUND = 50000  # 每期最大尝试次数
SIMULATION_ROUNDS = 10          # 增加模拟次数到 10 次，获取更多统计样本

def check_prize(target_nums, predicted_combo):
    """
    检查奖等 (1-5等奖)
    """
    target_main = set(target_nums[:6])
    target_bonus = target_nums[6]
    pred = set(predicted_combo)
    
    hits = len(target_main.intersection(pred))
    
    if hits == 6: return 1
    if hits == 5 and target_bonus in pred: return 2
    if hits == 5: return 3
    if hits == 4: return 4
    if hits == 3: return 5
    return 0

def run_blind_sim(target_drw_index, features, raw_data):
    """
    针对特定一期进行盲测
    """
    target_actual = raw_data[target_drw_index]
    print(f"\n>>> 开始模拟第 {target_drw_index + 1} 期...")
    print(f"    目标开奖号: {target_actual[:6]} [特别号: {target_actual[6]}]")
    
    # 1. 屏蔽目标期及之后的数据
    features_blind = features[:target_drw_index]
    
    # 2. 构造训练集
    X_train, y_train = [], []
    for i in range(WINDOW_SIZE, len(features_blind)):
        X_train.append(features_blind[i-WINDOW_SIZE:i])
        y_train.append(features_blind[i, :45])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # 3. 训练模型
    model = LottoModelEngine(window_size=WINDOW_SIZE, feature_dim=X_train.shape[2])
    model.train(X_train, y_train, epochs=50)
    
    # 4. 获取预测概率
    last_seq = features_blind[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, -1)
    probs = model.predict(last_seq)[0]
    
    # 5. 暴力破解尝试
    attempts = 0
    best_prize = 99
    hit_counts = {1:0, 2:0, 3:0, 4:0, 5:0}
    start_time = time.time()
    
    print(f"    正在分析概率空间... (目标: 至少中得 3 等奖)")
    
    while attempts < MAX_ATTEMPTS_PER_ROUND:
        attempts += 1
        combo = generate_smart_combinations(probs, count=1)[0]
        
        prize = check_prize(target_actual, combo)
        if prize > 0:
            hit_counts[prize] += 1
            if prize < best_prize:
                best_prize = prize
            
            # 如果中了大奖 (1-3等)，直接停止
            if prize <= 3:
                elapsed = time.time() - start_time
                print(f"    ✨ 惊喜！在第 {attempts} 次尝试时击中 {prize} 等奖！")
                print(f"    命中组合: {combo}")
                return attempts, prize, hit_counts
        
        if attempts % 10000 == 0:
            print(f"    - 已尝试 {attempts} 次... (目前最好成绩: {best_prize if best_prize < 99 else '无'}等奖)")
            
    return attempts, (best_prize if best_prize < 99 else 0), hit_counts

def run_consensus_sim(target_drw_index, features, raw_data):
    """
    共识测试：独立训练 5 次模型，看高频出现的“共识号”是否更准。
    """
    target_actual = raw_data[target_drw_index]
    target_main_set = set(target_actual[:6])
    print(f"\n>>> [共识盲测] 5次独立训练对比 - 第 {target_drw_index + 1} 期...")
    
    all_selected_nums = []
    
    # 独立训练 5 次
    for run_i in range(5):
        print(f"    正在进行第 {run_i + 1} 次独立模型训练...")
        features_blind = features[:target_drw_index]
        X_train, y_train = [], []
        for i in range(WINDOW_SIZE, len(features_blind)):
            X_train.append(features_blind[i-WINDOW_SIZE:i])
            y_train.append(features_blind[i, :45])
        
        X_train, y_train = np.array(X_train), np.array(y_train)
        model = LottoModelEngine(window_size=WINDOW_SIZE, feature_dim=X_train.shape[2])
        model.train(X_train, y_train, epochs=40) # 稍微减少 epoch 加快速度
        
        last_seq = features_blind[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, -1)
        probs = model.predict(last_seq)[0]
        
        # 每次训练生成 5 组号码
        combos = generate_smart_combinations(probs, count=5)
        for c in combos:
            all_selected_nums.extend(c)
            
    # 统计数字频率
    from collections import Counter
    counts = Counter(all_selected_nums)
    most_common = counts.most_common(10)
    
    print(f"    [频率报告] 出现次数最多的前 10 个号:")
    consensus_set = []
    for num, freq in most_common:
        is_hit = "✅" if num in target_main_set else "❌"
        print(f"      号码 {num:02}: 出现 {freq} 次 {is_hit}")
        consensus_set.append(num)
        
    top_6_consensus = set(consensus_set[:6])
    final_hits = len(top_6_consensus.intersection(target_main_set))
    print(f"    >>> 最终结论: Top 6 共识号命中了 {final_hits} 个。")
    return final_hits

def main():
    print("="*50)
    print("      发财加速器: Blind Simulation (共识度测试)")
    print("      目标: 验证多次训练产生的“共识号”是否有更高命中率")
    print("="*50)

    engine = LottoDataEngine(FILE_PATH)
    engine.sync_data()
    
    raw_df = pd.read_csv(FILE_PATH, header=None)
    raw_data = raw_df.values
    features = engine.get_features()
    
    latest_index = len(raw_data) - 1
    # 随机选 3 期进行共识测试
    target_indices = np.random.choice(range(1000, latest_index), 3, replace=False)
    
    results = []
    for idx in target_indices:
        hit_count = run_consensus_sim(idx, features, raw_data)
        results.append((idx + 1, hit_count))
    
    print("\n" + "="*50)
    print("共识策略实验报告:")
    for drw, hits in results:
        print(f"  第 {drw} 期: Top 6 共识号命中 {hits} 个")
    print("="*50)

if __name__ == "__main__":
    main()
