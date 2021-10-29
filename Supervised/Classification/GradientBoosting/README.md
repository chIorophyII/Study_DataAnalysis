# 그래디언트 부스팅(Gradient Boosting)
### 정의
### : 다수의 결정 트리를 통합해 강력한 모델을 만드는 앙상법 기법으로 이전 학습의 결과를 다음 학습에 전달해 이전의 오차(잔요오차)를 점진적으로 개선하는 방법
> - 복잡하지 않은 모델에서 시작해 이전 모델에 대한 학습을 통해 모델 성능을 개선(boosting 방식)
> - 이전 모델의 (잔여)오차를 지속적으로 개선하는 모델로 다양한 분야에 적용되는 인기 있는 기계학습 방법
> - 매개변수 조정(특히 학습률)에 민감해 트리가 추가 될수록 과대적합이 될 수 있음

## 파라미터
### * 조정 대상 파라미터(의사결정나무 참조)
- 분리 기준
- 트리 개수
- 최대 깊이
- 분리 노드 최소 자료 수
- 잎사귀 노드 최소 자료 수
- 최대 잎사귀 노드 수

### * 학습률
### * 설명변수 중요도

### 그래디언트 부스팅 모델 : GradientBoostingClassifier
sklearn.ensemble.**GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 100, criterion = "friedman_mse", min_samples_split = 2, min_samples_leaf = 1, random_state = None, max_depth = 3, max_features = None,...)

> learning_rate =: 학습률 지정. 이전 모델을 학습하는 정도. default(0.1) [0~1.0]  
> n_estimators : 생성할 트리 개수 지정. default(100)  

> criterion : 분리 기준 지정. default(friedman_mse)  
>    > mse : 평균 제곱 오차(mean squared error)  
>    > friedman_mse : friedman_mse에 의해 보완된 mse  
>    > mae : 평균 절대 오차(mean absolute error)  

> max_depth : 최대 깊이 지정. default(None)  
> min_samples_split : 분리 노드의 최소 자료 수 지정. 지정값보다 자료 수가 작으면 분리 미실행. default(2)  
> min_samples_leaf : 잎사귀 노드 최소 자료 수 지정. 모든 리프(끝 노드)의 수가 잎사귀 노드 최소 자료 수보다 클 때까지 분리    

> random_state : 초기 자료 선택 기준. 값에 따라 선택되는 데이터가 달라짐. default(None)  

> max_features : 모델 생성시 사용하는 설명변수의 수 지정(계산된 수만큼 설명변수를 임의 선택). default(None)  
>    > auto : sqrt(전체 변수의 수) 값 만큼 변수 임의 선택. sqrt와 동일  
>    > sqrt : sqrt(전체 변수의 수) 값 만큼 변수 임의 선택  
>    > log2 : log2(전체 변수의 수) 값 만큼 변수 임의 선택  
>    > None : 전체 변수 사용  
