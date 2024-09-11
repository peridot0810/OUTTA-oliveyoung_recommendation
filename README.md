# OUTTA-oliveyoung_recommendation🌟 

## 프로젝트 설명

올리브영의 특정 상품을 사용자에게 맞춤 추천하는 추천시스템입니다.
이 프로젝트는 사용자 입력기반 필터링을 통해 사용자의 피부타입과 사용자의 요구사항에 맞는 제품을 필터링합니다. 
또한 sementic search를 활용하여 사용자의 입력을 문장으로 받은 후 리뷰, 제목과 문장 유사도를 탐색하여 요구에 맞는 리뷰가 있는 상품을 추천합니다. 
이 밖에도 별점, 한달사용, 재구매 여부 등에 다양한 가중치를 부여하여 사용자에게 최적의 상품을 추천하는 것을 목표로 합니다.

### 주요 기능
- **기능 1**: 사용자의 피부타입에 맞는 제품을 추천합니다.
- **기능 2**: 사용자가 원하는 요구사항이 보편적이지 않더라도 sementic search를 통해 이와 관련된 상품을 찾아냅니다.
- **기능 3**: 편리한 인터페이스로 사용자에게 정보를 제공합니다.
- **기능 4**: 제품 바로가기 링크를 제공합니다. (24.09.11 기능 추가)

### 기술 스택
- **프로그래밍 언어**: Python, css, html
- **프레임워크**: Flask

## 데모 (Demo)


https://github.com/user-attachments/assets/940238e1-c433-428d-b1a0-42c081d9450e


## 설치 (Installation)
**OUTTA-oliveyoung_recommendation**를 설치하려면 다음 단계를 따르세요.
1. 저장소를 클론합니다:
   ```bash
   git clone https://github.com/peridot0810/OUTTA-oliveyoung_recommendation.git
   cd OUTTA-oliveyoung_recommendation

2. 가상환경을 설정합니다.
   ```bash
   python -m venv venv
   .\venv\Scripts\activate

3. 필요한 패키지를 설치합니다.
   ```bash
   pip install -r requirements.txt
**주의**
   - Python 3.7 이상, 3.11 미만의 버전에서만 설치
   - macOS 사용 시 필요한 패키지 별도 다운로드가 필요할 수 있습니다
   
4. Flask 서버 실행
   ```bash
   python app.py

## 파일구조
```bash
OUTTA-OLIVEYOUNG_RECOMMENDATION/
│
├── data/
│   ├── product_preprocessed.xlsx                      
│   └── review_preprocessed_summerized_embedding.xlsx  
│
├── OUTTA-oliveyoung_recommendation.git                
│
├── static/
│   └── style.css                                      
│
├── templates/
│   └── index.html                                     
│
├── venv/                                            
│   ├── etc/                                           
│   ├── Include/                                     
│   ├── Lib/                                           
│   ├── Scripts/                                       
│   └── share/                                         
│   └── pyvenv.cfg                                     
│
├── .gitattributes                                     
├── .gitignore                                         
│
├── app.py                                             
├── README.md                                          
└── requirements.txt                                   
```


## 팀 멤버
- 김준혁
- 김태욱
- 정재웅
- 신성현

  
