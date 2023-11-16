# Face tracking

* 환경 설정

  1. python 3.9 이상 버전 설치
  2. torch 설치
     * 아래 사이트에서 맞는 버전을 설치
     * https://pytorch.org/get-started/locally/

    3. 나머지 설치 

       ~~~
       pip install opencv-python
       pip install hydra-core --upgrade
       ~~~

-----------------------

* 프로그램 실행

  ~~~python
  cd facetracker
  python main.py
  ~~~

----------

* Parameter 조정
  * Config 폴더 내부에 yaml 파일에서 조정가능
    * 조정이 가능한 것들은 주석으로 설명 추가해둠