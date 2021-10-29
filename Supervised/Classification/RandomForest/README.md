# 랜덤 포레스트(Random Forest)
### 정의
### : 독립적으로 다수의 결정 트리를 생성하고 그 결과를 결합한(예측 레이블 빈도/확률) 최종 모델을 생성함으로써 성능을 높이고 과대적합을 방지해 안정적인 모델을 생성하는 앙상블 방법
> 일반적으로 설명변수 및 표본을 무작위로 추출(복원추출, bagging 방식)해 모델 생성  
> 생성된 다수 모델의 분류한 결과를 결합해 다수결 투표로 레이블을 결정, 모델의 일반화(안정성)이 향상됨  
> 과대 적합 위험이 있는 하나의 결정 트리 보다 안정적이고 예측 성능이 높음

## 파라미터
### * 조정 대상 파라미터 (의사결정나무 참조)
- 분리 기준
- 트리 개수
- 최대 깊이
- 분리 노드 최소 자료 수
- 잎사귀 노드 최소 자료 수
- 최대 잎사귀 노드 수  

### * OOB 평가
### * 설명변수 중요도

### 랜덤 포레스트 모델 : RandomForestClassifier
sklearn.ensemble.**RandomForestClassifier**(n_estimators = "warn", criterion = "gini", max_depth = None, min_samples_split = 2, min_samples_leaf = 1, max_features = "auto", max_leaf_noeds = None, bootstrap = True, oob_score = False, random_state = None)

> n_estimators : 생성할 트리 개수 지정. default(warn)  

> criterion : 분리 기준 지정. default(gini)  
>    > gini : 지니 지수  
>    > entropy : 엔트로피 지수  

> max_depth : 최대 깊이 지정. default(None)  
> min_samples_split : 분리 노드의 최소 자료 수 지정. 지정값보다 자료 수가 작으면 분리 미실행. default(2)  
> min_samples_leaf : 잎사귀 노드 최소 자료 수 지정. 지정값보다 자료 수가 작으면 분리 미실행. default(1)  

> max_features : 모델 생성시 사용하는 설명변수의 수 지정(계산된 수만큼 설명변수를 임의 선택). default(auto)  
>    > auto : sqrt(전체 변수의 수) 값 만큼 변수 임의 선택. sqrt와 동일  
>    > sqrt : sqrt(전체 변수의 수) 값 만큼 변수 임의 선택  
>    > log2 : log2(전체 변수의 수) 값 만큼 변수 임의 선택  
>    > None : 전체 변수 사용  

> max_leaf_nodes : 최대 분리 노드(leaf) 수 지정. default(None)  
> bootstrap : bootstrap 사용 지정. default(None)  
> oob_score : 일반적인 정확도 추정을 위한 oob 데이터 사용 지정. default(False)  
> random_state : 초기 자료 선택 기준. 값에 따라 선택되는 데이터가 달라짐. default(None)
