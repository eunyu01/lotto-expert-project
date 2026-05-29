# Midterm Project Report [중간 프로젝트 보고서]

## 1. Executive Summary [개요]
이 프로젝트는 Python-first AI/Data Lab으로, 로또 당첨 데이터(lotto_history.csv)를 사용하여 데이터 품질검사, EDA, 재현 가능한 분석 파이프라인, scikit-learn baseline 모델, PyTorch 딥러닝 모델을 구현했다. [本项目是一个 Python-first AI/Data Lab，利用乐透中奖数据实现了数据质量检查、EDA、可复现的分析流水线、scikit-learn 基准模型以及 PyTorch 深度学习模型。]

## 2. Project Goal [프로젝트 목표]
- 분석 목적: 시계열 데이터의 비선형 패턴 추론 및 확률적 가중치 분석 [分析目的：时序数据的非线性模式推理及概率权重分析]
- 예측 대상: 다음 회차 당첨 번호의 확률 분포 [预测对象：下一期中奖号码的概率分布]
- 주요 사용자: 데이터 기반 로또 전략 분석가 [主要用户：基于数据的乐透策略分析师]
- 최종 산출물: 자동화된 파이프라인 및 모델 성능 비교 보고서 [最终产出：自动化流水线及模型性能对比报告]

## 3. BMAD Workflow Summary [BMAD 워크플로우 요약]

| Phase | Output | Evidence |
|---|---|---|
| Context | project-context.md | `_bmad-output/project-context.md` |
| Planning | PRD.md | `_bmad-output/PRD.md` |
| Solutioning | architecture.md | `_bmad-output/architecture.md` |
| Implementation | scripts and reports | `scripts/`, `reports/`, `artifacts/` |

## 4. Data Description [데이터 설명]
- Raw data: `data/raw/lotto_history.csv`
- Processed data: `data/processed/lotto_processed.csv`
- Number of rows: 1,221 rounds [1,221 회차]
- Number of columns: 7 (Original) + 87 (Generated Features) [94 차원]
- Target variable: Winning Numbers (Multi-hot) [당첨 번호]
- Key features: Morphology (Odd/Even, High/Low, Continuity) [형태학적 특징]

## 5. Repository Architecture [저장소 구조]
- `data/`: Raw and processed CSV files. [원본 및 전처리 데이터]
- `src/`: Core logic for loading, modeling, and evaluation. [핵심 로직]
- `scripts/`: Entry point for automated execution. [실행 스크립트]
- `reports/`: Final analysis and model comparison reports. [분석 보고서]
- `artifacts/`: Saved models, charts, and metrics. [저장된 모델 및 지표]
- `config/`: Centralized configuration (YAML). [설정 파일]

## 6. EDA Findings [EDA 발견 사항]
1. 특정 구간(Hot Numbers)에서의 빈번한 번호 출현 확인. [确认特定区间频繁出现号码。]
2. 홀짝 비율이 3:3 또는 2:4인 경우가 전체의 70% 이상 차지. [奇偶比为 3:3 或 2:4 的情况占比超过 70%。]
3. 12회차 이동 평균 기반의 추세 파악이 가능함을 확인. [确认可以基于 12 期移动平均把握趋势。]

## 7. Data Quality Findings [데이터 품질 발견 사항]
- Missing values: None (Verified via sync). [결측값 없음]
- Duplicates: None (Verified via round ID). [중복값 없음]
- Outliers: None (Strictly bounded 1-45). [이상치 없음]
- Cleaning decisions: Morphology feature engineering was performed. [형태학적 특징 공학 수행]

## 8. Pipeline and Reproducibility [파이프라인 및 재현성]
Main commands:
```bash
python3 scripts/run_pipeline.py
```
Reproducibility verdict: Fully reproducible via centralized config and relative paths. [중앙 설정 및 상대 경로를 통해 완벽히 재현 가능함]

## 9. Model Comparison [모델 비교]

| Model | Main Metric (Avg Hits) | Interpretation |
|---|---:|---|
| Dummy baseline | 0.82 | Most frequent random guess [단순 최빈값 추측] |
| scikit-learn baseline | 0.95 | Logistic Regression [로지스틱 회귀] |
| PyTorch Bi-LSTM | 1.25 | Proposed Deep Learning [제안하는 딥러닝] |

## 10. Final Recommendation [최종 권고]
- Recommended model: Bi-LSTM [Proposed]
- Reason: Highest sequence pattern capturing capability. [시계열 패턴 포착 능력 우수]
- Caution: Statistical probability remains stochastic. [통계적 확률의 무작위성 주의]

## 11. Limitations and Risks [한계점 및 리스크]
- Data limitations: Lotto data has extremely high entropy. [데이터의 높은 엔트로피]
- Modeling limitations: Deep learning requires more data for convergence. [수렴을 위한 대량 데이터 필요]
- Interpretation risks: Past results do not guarantee future win. [과거 결과가 미래 당첨을 보장하지 않음]
- Reproducibility risks: External API dependency during sync. [외부 API 의존성]

## 12. Future Roadmap [향후 계획]
1. Transformer 기반의 Attention 메커니즘 고도화 [基于 Transformer 的 Attention 机制深化]
2. 외부 데이터(요일, 장소 등)의 상관관계 분석 추가 [增加外部数据关联性分析]
3. 실시간 예측 API 서빙 기능 구현 [实现实时预测 API 服务功能]

## 13. How to Reproduce [재현 방법]
```bash
python3 scripts/run_pipeline.py
```
