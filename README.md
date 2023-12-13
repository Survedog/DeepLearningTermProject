# DeepLearningTermProject
2023-2학기 숭실대학교 딥러닝 과목의 프로젝트 저장소입니다.

프로젝트 클래스 구조, 각 모델 세부 구현 방식은 ‘밑바닥부터 시작하는 딥러닝2’의 github 예제코드를 참고하였음.
- https://github.com/WegraLee/deep-learning-from-scratch-2

프로젝트 환경
-	Windows 10
-	IDE: Pycharm 2023.2.2
-	언어: Python 3.10

* konlpy를 사용하기 위해선 Jpype이 설치되어야 함.

사용 데이터: 에세이 글 평가 데이터
- https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=545
- https://www.youtube.com/watch?v=IET9IkfQIKE&ab_channel=NIAAIHub
- json 데이터를 다음의 경로에 넣어야 함.
   - '01.데이터\1.Training\라벨링데이터'에 위치한 압축 파일들을 아래 경로에서 압축 해제.
      - DeepLearningTermProject\data\test_data
   - 01.데이터\2.Validation\라벨링데이터에 위치한 압축 파일들을 아래 경로에 압축 해제.
      - DeepLearningTermProject\data\train_data      
