# AI 에이전트를 위한 프로젝트 맥락 [面向 AI 代理的项目 맥락]

1 프로젝트 유형 [项目类型]
이 프로젝트는 Python first AI Data Lab입니다. [本项目是 Python-first AI/Data Lab。]
웹 애플리케이션이나 상업용 서비스가 아닌 데이터 기반의 추론과 알고리즘 성능 검증을 위한 실험 환경입니다. [这不是网页应用或商业服务，而是为了基于数据的推理和算法性能验证的实验环境。]

2 핵심 도구 [核心工具]
AI 코딩 도구는 Gemini CLI 또는 Claude Code를 사용합니다. [AI 编程工具使用 Gemini CLI 或 Claude Code。]
워크플로우는 BMAD 방식을 따릅니다. [工作流遵循 BMAD 方式。]
언어 및 라이브러리는 Python 3.12, Pandas, Scikit learn, PyTorch, Matplotlib을 사용합니다. [语言和库使用 Python 3.12, Pandas, Scikit-learn, PyTorch, Matplotlib。]

3 데이터 및 재현성 규칙 [数据及再现性规则]
데이터 정책으로 data raw 폴더의 원본 데이터는 절대 수정하지 않습니다. [数据政策规定绝对不修改 data/raw 文件夹中的原始数据。]
모든 정제 데이터는 data processed 폴더에 저장합니다. [所有清洗后的数据都保存在 data/processed 文件夹中。]
노트북 정책으로 notebooks 폴더는 탐색용으로만 사용하며 실제 로직은 src 폴더에 구현합니다. [笔记本政策规定 notebooks 文件夹仅用于探索，实际逻辑在 src 文件夹中实现。]
재현성을 위해 모든 분석과 모델 학습은 scripts 폴더의 진입점을 통해 한 줄의 명령어로 재실행 가능해야 합니다. [为了保证再现性，所有的分析和模型训练必须通过 scripts 文件夹的入口点，使用一行命令即可重新执行。]
