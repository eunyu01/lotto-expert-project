import requests
import pandas as pd
import numpy as np
import os
import time

class LottoDataEngine:
    def __init__(self, raw_path='data/raw/lotto_history.csv', processed_path='data/processed/lotto_history_synced.csv'):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.base_url = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo="

    def sync_data(self):
        print(f"[1/4] 正在同步数据至最新期数...")
        
        # 优先从已处理的路径读取，如果不存在则从原始路径读取
        source_path = self.processed_path if os.path.exists(self.processed_path) else self.raw_path
        
        df = pd.DataFrame()
        start_drw = 1
        
        if os.path.exists(source_path):
            try:
                df = pd.read_csv(source_path, header=None)
                start_drw = len(df) + 1
            except Exception as e:
                print(f"   - 读取源文件失败: {e}")
        
        new_data = []
        consecutive_fails = 0
        while consecutive_fails < 3:
            try:
                resp = requests.get(f"{self.base_url}{start_drw}", timeout=5)
                data = resp.json()
                if data.get("returnValue") == "fail":
                    consecutive_fails += 1
                else:
                    nums = [data[f"drwtNo{i}"] for i in range(1, 7)]
                    nums.append(data["bnusNo"])
                    new_data.append(nums)
                    if start_drw % 50 == 0:
                        print(f"   - 已获取第 {start_drw} 期")
                    start_drw += 1
                    consecutive_fails = 0
                time.sleep(0.05) # 礼貌抓取
            except:
                break
        
        if new_data:
            df_new = pd.DataFrame(new_data)
            df_final = pd.concat([df, df_new], ignore_index=True)
            # 确保目录存在
            os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
            df_final.to_csv(self.processed_path, header=False, index=False)
            print(f"   - 新增 {len(new_data)} 期数据，同步完成。保存至: {self.processed_path}")
        else:
            print("   - 数据已是最新。")
            if not os.path.exists(self.processed_path) and os.path.exists(self.raw_path):
                # 如果没有新数据，但 processed 文件还没生成，则复制一份
                os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
                pd.read_csv(self.raw_path, header=None).to_csv(self.processed_path, header=False, index=False)

    def generate_synthetic_data(self, num_rounds=1000):
        """
        生成纯随机的模拟数据，用于压力测试和过拟合检测。
        """
        print(f"   - 正在生成 {num_rounds} 期纯随机模拟数据...")
        synthetic_data = []
        for _ in range(num_rounds):
            # 模拟 45 选 6
            nums = np.random.choice(range(1, 46), 6, replace=False)
            nums.sort()
            # 模拟特别号 (bonus)
            remaining = [n for n in range(1, 46) if n not in nums]
            bonus = np.random.choice(remaining)
            synthetic_data.append(list(nums) + [bonus])
        return np.array(synthetic_data)

    def get_features(self, custom_data=None):
        """
        提取特征。增加形态学特征：连号数、奇偶比、大小比。
        """
        if custom_data is not None:
            data = custom_data
        else:
            # 优先使用同步后的数据
            load_path = self.processed_path if os.path.exists(self.processed_path) else self.raw_path
            df = pd.read_csv(load_path, header=None)
            data = df.values
        
        num_rows = len(data)
        # 1. Multi-hot (45 dim)
        multi_hot = np.zeros((num_rows, 45))
        # 2. Morphological features (3 dim)
        morphology = np.zeros((num_rows, 3))
        
        for i, row in enumerate(data):
            main_nums = sorted(row[:6])
            # Multi-hot
            indices = np.array(main_nums) - 1
            multi_hot[i, indices.astype(int)] = 1
            
            # Morphology: Consecutive counts
            consecutive = 0
            for j in range(len(main_nums)-1):
                if main_nums[j+1] - main_nums[j] == 1:
                    consecutive += 1
            morphology[i, 0] = consecutive / 5.0 # 归一化
            
            # Morphology: Odd ratio
            odds = len([x for x in main_nums if x % 2 != 0])
            morphology[i, 1] = odds / 6.0
            
            # Morphology: High ratio (23-45 为大号)
            highs = len([x for x in main_nums if x >= 23])
            morphology[i, 2] = highs / 6.0
            
        # 3. Omission values (45 dim)
        omissions = np.zeros((num_rows, 45))
        current_omission = np.zeros(45)
        for i in range(num_rows):
            omissions[i] = current_omission.copy()
            current_omission += 1
            appeared = np.where(multi_hot[i] == 1)[0]
            current_omission[appeared] = 0
            
        # 合并特征: [45(分布), 45(遗漏), 3(形态)] = 93 维
        features = np.hstack([multi_hot, omissions, morphology])
        return features

if __name__ == "__main__":
    engine = LottoDataEngine()
    engine.sync_data()
    print("特征提取测试:", engine.get_features()[-1])
