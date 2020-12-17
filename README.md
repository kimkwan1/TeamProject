# TeamProject
## 주제 : 딥러닝을 이용한 2차원 이징 모형 데이터 학습

딥러닝 기법인 CNN(Convolutional Neural Network)을 이용하여 2차원 이징 모형을 학습시켜 앞에서 생성한 이징 모형 데이터의 온도를 예측하는 알고리즘을 만들었다. CNN에서 학습한 가중치 행렬이 어떠한 방식으로 학습을 하였는지 알아보고 order parameter의 역할에 대하여 생각해본다. 

#### 예시
<img src="https://github.com/kimkwan1/TeamProject/blob/main/Feature%20visual.png" />

# Ising model

통계역학에서 이징 모형은 자석의 간단한 격자모형이다. 강자성체를 위치가 고정되어 있는 자기 쌍극자의 격자로 나타낸다. 각 쌍극자는 +1 또는 -1 두개의 상태를 가질 수 있고, 하나의 격자에서 주변의 쌍극자와 상호작용한다. 1/2 스핀들의 모임으로 구성된 계에서 스핀들 사이의 상호작용에 의한 해밀토니안은 다음과 같이 기술된다.

<img src="https://latex.codecogs.com/gif.latex?H%20%3D%20-%5Csum%5Climits_%7B%3Ci%2Cj%3E%7DJ_%7Bij%7D%5Ccdot%5Csigma_i%5Ccdot%5Csigma_j%20-%20h%5Ccdot%5Csigma_i" />

* 시그마 기호 아래의 <i, j> 는 i 와 j 가 인접한 격자인 경우만 생각한다는 뜻이고,J는 두 스핀 사이의 상호작용, sigma는 i번째 격자점의 스핀 상태, h는 외부자기장을 나타내는 매개변수이다. 

위와 같은 시스템에서 스핀이 같은 방향으로 정렬될수록 전체 에너지가 낮아지는 것을 알 수 있다. 따라서 2차원 이징 모형에서는 온도 변화에 따라 강자성-상자성 사이의 상전이 특징을 확인할 수 있다. T=0 K인 경우, 모든 스핀들은 +1또는 -1로 정렬되며 강자성을 띄게 된다. 그러나 온도 증가로 임계온도가 되면 시스템은 자성을 잃게 된다.

우리는 J가 1이며 외부 자기장이 없는 2차원 이징 모형을 사용하였고, 시뮬레이션을 하기 위해 상호작용은 각 격자점에서 동서남북 4개의 스핀들과 이루어 진다고 가정하였다. 

# Monte - Carlo Method

몬테카를로 방법은 난수를 이용하여 함수의 값을 확률적으로 계산하는 알고리즘이다. 이징 모형의 데이터를 만들기 위하여 시뮬레이션을 할 때, 특정 온도에서 시스템이 갖는 상태를 만들어야 한다. 격자점의 수가 많아질수록 계산이 복잡해지기 때문에 확률적으로 스핀이 뒤집어지는 것을 이용하여 몬테카를로 방법을 사용한다.

## Metropolis algorithm

메트로폴리스 알고리즘은 몬테카를로 방법 중 하나로 대략적인 순서는 다음과 같다.

1. 특정 온도에서 임의로 한 격자점을 선택한다. 
2. 그 격자점의 스핀을 뒤집어 시스템의 에너지를 낮출 수 있으면 실행하고 그렇지 않다면 특정 확률을 따라 스핀을 뒤집는다. 
3. 이러한 과정을 열적 평형에 도달할 때까지 반복한다. 
4. 온도를 약간 증가시킨후 1-3의 과정을 실행한다.

# Ising model dataset

```python
import matplotlib.pyplot as plt
import numpy as np

def initialize(N):
  NN= np.random.choice([1,-1],(N,N))
  return NN
```

```python
def boundary_condition(NN,N,T,i,j):
  beta=1/T 
  ## i = row, j = column
  if j>N :
    j-=N
    i+=1
  if i>N :
    i-=N

  n_sum = 0
  ## 동서남북 1칸 스핀값 더하기, Periodic boundary
  pointer=i+1
  if pointer == N :
    pointer-=N
  n_sum=NN[pointer,j]
  pointer=i-1
  if pointer < 0:
    pointer+=N
  n_sum += NN[pointer,j]
  pointer=j+1
  if pointer == N :
    pointer-=N
  n_sum += NN[i,pointer]
  pointer=j-1
  if pointer <0:
    pointer+=N
  n_sum += NN[i,pointer]
  
 ## 스핀 뒤집기
  E= 2 * n_sum * NN[i,j]
  if E <= 0:     # 스핀을 뒤집고 나서 측정한 에너지가 낮아진 경우
    NN[i,j] *= -1
  elif np.random.rand() < np.exp(-beta*E) :  # 에너지가 낮아지지 않은 경우
    NN[i,j] *= -1
  
  return NN,i,j+1
```

```python
N=100          ## (100,100)
NN=initialize(N)
T_list=[]
Magnet=[]
T=0
## 온도를 0.1씩 증가시키며 저장
for Temper in range(50):
  T_list.append(T)
  ind_i,ind_j=0,0
  ## N*N 회 반복해야 1 update * 5000회 시행하여 평형 상태
  for i in range(5000*N*N):
    NN,ind_i,ind_j=boundary_condition(NN,N,T,ind_i,ind_j)
  Magnet.append(sum(sum(NN))/(N*N)) # 자기화 정도 측정
  T+=0.1
```

<img src="https://github.com/kimkwan1/TeamProject/blob/main/Mag-sus.png" />

그림은 위의 코드로부터 얻은 전체 시스템의 자기화와 자화율의 결과값이다. 온도가 증가하여  2.3 정도에서 상전이가 일어나는 것을 확인할 수 있다.

<div>
  <img width="200" src="https://github.com/kimkwan1/TeamProject/blob/main/T%3D1.5.gif" />
  <img width="200" src="https://github.com/kimkwan1/TeamProject/blob/main/T%3D2.3.gif" />
  <img width="200" src="https://github.com/kimkwan1/TeamProject/blob/main/T%3D3.5.gif" />
</div>



2차원 이징 모형의 데이터는 위의 그림과 같다. 온도는 1.5 ~ 3.5 사이에 0.1 간격을 두었고 각각 평형 상태의 데이터 5000개씩 총 105,000 개의 데이터셋을 얻었다. 전체 데이터 셋의 70%인 73500개는 Training_set, 나머지 31500개를 Test_set 으로 나누었다. label은 T의 값으로 설정.


# 합성곱 신경망(Convolutional Neural Network, CNN)

2차원 이징 모형 이미지 데이터는 스핀이 주변 격자와 상호작용하기 때문에 CNN 방식을 선택하였다. CNN의 핵심 개념은 이미지 데이터에 필터라고 불리는 가중치 행렬을 연산하여 필터가 데이터의 특징을 학습하는 방식이다. 

우리가 사용한 모델은 합성곱층 3개 와 출력 신경망 2개를 사용하였다.

```python
model = tf.keras.Sequential()
model.add(layers.Conv2D(25, (50, 50), strides=2, padding='same',
                        input_shape=(100, 100, 1), activation='relu', name='conv2_layer1'))
# model.add(layers.BatchNormalization(name='batch_norm1'))
model.add(layers.Conv2D(50, (10, 10), strides=2, padding='same',
                        activation='relu', name='conv2_layer2'))
# model.add(layers.BatchNormalization(name='batch_norm2'))
model.add(layers.Conv2D(100, (5, 5), strides=5, padding='same',
                        activation='relu', name='conv2_layer3'))
# model.add(layers.BatchNormalization(name='batch_norm3'))
model.add(layers.Flatten(name='flat_layer'))
model.add(layers.Dense(1024, activation='relu', name='linear_layer1'))
model.add(layers.Dropout(0.2, name='dropout'))
model.add(layers.Dense(1, activation='relu', name='linear_layer2'))
```
다음 그림은 위의 CNN model을 도식화하여 나타낸 것이다. 

<img src="https://github.com/kimkwan1/TeamProject/blob/main/model.png" />

미니배치로 256개의 데이터를 288회 업데이트, 총 20번 학습하였다.

**Batch_size = 256, Iteration = 288,  Epoch = 20**

다음 그림은 1~20에폭에 따라 Mean Square Error를 나타낸다.

<img src="https://github.com/kimkwan1/TeamProject/blob/main/epoch-MSE.png" />

<img src="https://github.com/kimkwan1/TeamProject/blob/main/pred.png" />

학습이 끝난 후, Test_set을 이용하여 평가하였다. 결과는 31500개의 Test_set에 대하여 MSE값이 0.081로 학습이 잘 되었다.
그렇다면 우리의 바람대로 CNN의 가중치행렬이 학습을 잘 하였고 필터들을 분석한다면 데이터에서 압축한 정보, 즉 Order Parameter와 관련지어 볼 수 있을 것이다.

**필터의 평균값**

<img src="https://github.com/kimkwan1/TeamProject/blob/main/weights1.png" />

<img src="https://github.com/kimkwan1/TeamProject/blob/main/weights2.png" />

실패하였다. 

**온도별 학습 후, 필터 평균값** 

<img src="https://github.com/kimkwan1/TeamProject/blob/main/T_weights.png" />

위의 그림은 각각의 온도마다 데이터를 따로 학습 후, 필터의 평균값을 구했다. 다른 이미지(MNIST, 사람) 데이터는 특정 부분마다 개연성 있는 공통적인 특징(ex:코 아래 입)이 있고 가중치 행렬이 학습을 하였다면, 우리가 사용한 데이터는 전체적인 클러스터의 분포를 학습했다고 볼 수 있다. 
또한 위 그림의 필터값을 보면 2.3~2.6사이에 다른 온도보다 전체적으로 활성화 된 모습을 띈다. 다른 온도에서는 데이터 사이의 급격한 변화가 적고 한 데이터 속에서 부분 부분이 비슷하기 때문에 필터가 작은 정보로도 충분히 학습한 것으로 보여진다. 그러나 상전이 현상이 일어나는 온도 근처에서는 각 데이터의 클러스터 위치 변화가 크고 학습에 많은 정보를 필요로 하여 복잡한 계산 때문에 활성화 되었다고 유추할 수 있다.
