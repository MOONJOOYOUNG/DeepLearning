# t-SNE(Stochastic Neighbor Embedding)
* 차원축소(dimesionality reduction)
* 시각화(visualization)

# Stochastic Neighbor Embedding
* 고차원 데이터 x를 이웃 간의 거리를 보존하며 저차원의 y로 학습.
* 거리 정보(Euclidean distances)를 확률적으로 나타냄.

# crowding problem
* 비선형 차원축소 기법 적용 시 다수의 관측치들이 겹쳐 보이는 문제.
* 기존의 SNE 경우 정규분포를 이용하여 유사도를 계산했지만, 이를 방지하기 위해 T 분포를 사용.

# MNIST Data Visualization

![텍스트](https://t1.daumcdn.net/cfile/tistory/9964B04B5B8D0D6212)

Paper: (http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

blog: (http://mjdeeplearning.tistory.com/36/)
