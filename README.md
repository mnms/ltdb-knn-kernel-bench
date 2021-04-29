# README

Lightning DB 에서 K-Nearest Neighbor Search 를 처리하기 위해 사용되고 있는 함수들입니다.

`int single()` 혹은 `int batch()` 함수를 통해 함수들의 성능을 측정할 수 있습니다.

`test_size` 는 전체 데이터셋의 크기를, `test_dim` 은 테스트에 사용되는 feature 의 차원을 `test_k` 는 구하고자 하는 k 값을, `test_iter` 는 속도 측정의 정확성을 높이기 위한 반복 회수를 지정할 수 있습니다.

`run.sh` 로 필요한 [tsimd](https://github.com/jeffamstutz/tsimd) 라이브러리를 받아서 컴파일 후 실행할 수 있습니다.
