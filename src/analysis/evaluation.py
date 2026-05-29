import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class LottoValidationEngine:
    def __init__(self):
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # 支持中文显示
        plt.rcParams['axes.unicode_minus'] = False

    def backtest(self, model_engine, features, window_size, test_rounds=20):
        print(f"[3/4] 启动回测系统 (测试最近 {test_rounds} 期表现)...")
        results = []
        
        for i in range(len(features) - test_rounds, len(features)):
            # 准备回测输入 (前 window_size 期)
            input_seq = features[i-window_size : i]
            # 真实的中奖号索引 (值为1的索引)
            actual_indices = np.where(features[i] == 1)[0]
            
            # 预测概率
            probs = model_engine.predict(input_seq.reshape(1, window_size, -1))[0]
            # 选取概率最高的 6 个号码索引
            pred_indices = np.argsort(probs)[-6:]
            
            # 计算命中数
            hits = len(set(pred_indices) & set(actual_indices))
            results.append(hits)
            
        avg_hits = np.mean(results)
        print(f"   - 回测平均命中数: {avg_hits:.2f} (随机基准约为 0.8)")
        return results

    def plot_insights(self, hits_history, prob_dist):
        plt.figure(figsize=(12, 5))
        
        # 1. 命中趋势图
        plt.subplot(1, 2, 1)
        plt.plot(hits_history, marker='o', color='b', label='模型命中数')
        plt.axhline(y=0.8, color='r', linestyle='--', label='随机基准')
        plt.title("回测命中趋势分析")
        plt.xlabel("测试期数")
        plt.ylabel("命中个数")
        plt.legend()
        
        # 2. 预测概率分布热力图 (模拟)
        plt.subplot(1, 2, 2)
        grid = prob_dist.reshape(5, 9)
        sns.heatmap(grid, annot=True, cmap="YlGnBu", cbar=False)
        plt.title("1-45 号码预测热力分布")
        
        plt.tight_layout()
        plt.savefig("lotto_report.png")
        print("[4/4] 已生成可视化研究报告: lotto_report.png")
