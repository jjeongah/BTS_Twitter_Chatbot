<p align="center"><img src="https://user-images.githubusercontent.com/65378914/217187454-b8159fff-7152-4125-9a18-2c0ccf236aeb.png" width="80%" height="80%"/></p>

<br/>

## 1️⃣ Introduction
Let's fangirl(덕질) together! **ArmyBot** is a chatbot service that allows you to have various conversations like a typical BTS fan friend on Twitter. Just tweet [@armybot_13](https://twitter.com/armybot_13) with your questions or any chit chats, and ArmyBot will reply to you.

<br/>

## 2️⃣ Team Members

김별희|이원재|이정아|임성근|정준녕|
:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/42535803?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/61496071?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/65378914?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/14817039?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/51015187?v=4' height=80 width=80px></img>
[Github](https://github.com/kimbyeolhee)|[Github](https://github.com/wjlee-ling)|[Github](https://github.com/jjeongah)|[Github](https://github.com/lim4349)|[Github](https://github.com/ezez-refer)


### Contribution

- `김별희` Twitter data collection and preprocessing pipeline, answer retriever implementation
- `이원재` Service planning and project management, pre-training data collection and preprocessing, answer-related processing
- `이정아` Spam filtering data collection and model implementation, Twitter integration and service implementation, keyword visualization
- `임성근` TheQoo data collection and preprocessing pipeline, informative data collection
- `정준녕` Generation model pipeline implementation, generation model prototype and demonstration page implementation, chatbot service data construction

<br/>

## 3️⃣ Demo Video

![service example](./utils/example.gif)

<br/>

## 4️⃣ Service Architecture

<p align="center"><img src="https://user-images.githubusercontent.com/42535803/217479698-d16965e8-4ac0-4b65-9cfa-e7d2011ef02a.png" width="90%" height="90%"/></p>

1. User tags the bot account and writes a tweet.
2. Spam tweet filtering:
    - If a tweet is determined to be malicious, a predefined response is returned.
3. Intent keyword matching and BM25-based Elastic Search:
    - If the retrieved reply has a BM25 score above a threshold and matches the intent keywords, the reply is post-processed and returned to the user.
    - If the above two conditions are not satisfied, the input is passed to the Generation model, and the generated result is filtered for offensive expressions before being returned to the user.
4. Input/output analysis and other information are stored in MongoDB for analysis.

<br/>

<details>
    <summary><b><font size="10">Project Tree</font></b></summary>
<div markdown="1">

```
.
|-- agent.py # 트위터 챗봇 서비스 최종 코드
|-- chatbot # 챗봇 generator, retriever 모델 모듈
|   |-- generator
|   |-- pipeline
|   |-- readme.md
|   `-- retriever
|-- corpus # 코퍼스 구축 모듈
|   |-- README.md
|   |-- build_corpus.py
|   |-- crawlers
|   `-- twitter_classification
|-- database
|   `-- mongodb.py
|-- install_requirements.sh
|-- notebook
|   |-- AIhub_data_to_csv.ipynb # 데이터를 csv로 변환
|   |-- spell_compare.ipynb # 맞춤법 교정 라이브러리 성능 비교
|   `-- upload_dataset_to_huggingface.ipynb # 데이터셋을 HF에 ㅇ
|-- poetry.lock
|-- prototype # 프로토타입 app 모듈
|   |-- Makefile
|   |-- app
|   |-- config
|   |-- poetry.lock
|   |-- pyproject.toml
|   |-- readme.md
|   `-- requirements.txt
|-- pyproject.toml
|-- readme.md
|-- requirements.txt
|-- spam_filter # 스팸 필터 모듈
|   |-- config
|   |-- data_loader
|   |-- readme.md
|   |-- spam_filter.py
|   |-- spam_inference.py
|   `-- spam_train.py
|-- twitter # 트위터 연결 모듈
|   |-- automatic_reply.py
|   |-- config
|   |-- data_pipeline.py
|   |-- last_seen_id.txt
|   |-- main.py
|   |-- readme.md
|   |-- tweet_pipeline.py
|   `-- utils
`-- utils
    |-- EDA.py # kiwi로 단어 빈도수 확인
    |-- base_config.yaml
    |-- classes.py # dataclass 모듈별 입출력 포맷
    |-- push_model_to_hub.py # huggingface에 모델 업로드
    `-- wordcloud.py # 워드클라우드로 단어 시각화
```
    
</div>
</details>

<br/>

## 5️⃣ DataSets
<p align="center"><img src="https://user-images.githubusercontent.com/42535803/217480915-626de87e-b45f-4945-8454-1918ff2f8362.png" width="80%" height="80%"/></p>

- [AI Hub Entertainment News](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=625): 3,144 articles, 10.67MB
- Naver News BTS-related articles: 1,337 articles, 4.85MB
- [Daily Conversation and Comfort Chatbot Data](https://github.com/songys/Chatbot_data): 962,681 pairs of questions and answers, 108.43MB
- Naver Knowledgein BTS category: 7,785 questions and answers, 8.70MB
- TheQoo BTS category posts/comments: 13,709 posts, 3.53MB
- Twitter BTS fan tweets/replies: 8,106 tweets and replies, 1.45MB
- [Korean-hate-speech](https://github.com/kocohub/korean-hate-speech): 7,896 sentences
- [KOLD](https://github.com/boychaboy/KOLD): 40,429 sentences
- [Korean_unsmile_data](https://github.com/smilegate-ai/korean_unsmile_dataset): 7,896 sentences
- [Curse-detection-data](https://github.com/2runo/Curse-detection-data): 6,154 sentences

## 6️⃣ Modeling
- Generation model
    - paust/pko-t5-base 기반 pretrainig + finetuning
        - [nlpotato/pko-t5-base_ver1.1](https://huggingface.co/nlpotato/pko-t5-base_ver1.1)
    - BTS 관련 토큰 추가
        - Vocab size : 50383
        - "BTS", "bts", "RM", "rm", "진", "김석진", "석진", "김남준", "남준", "슈가", "민윤기", "윤기", "제이홉", "정호석", "지민", "박지민", "뷔", "김태형", "태형", "V", "정국", "전정국", "아미", "빅히트", "하이브", "아미", "보라해"
    - Finetuning
        1. 일상 대화 및 위로 문답 챗봇 데이터
        2. BTS 관련 네이버 지식인 데이터
        3. 더쿠 BTS 카테고리 글/댓글 + 트위터 BTS 팬 트윗/답글 데이터
    - Model size : 1.1GB
    - Number of trainable parameters : 275,617,536
- Retreiver model
    - Elastic Search with BM25
- Spam filtering model
    - klue/bert-base 기반 finetuning 
        -  [nlpotato/spam-filtering-bert-base-10e](https://huggingface.co/nlpotato/spam-filtering-bert-base-10e)

<br/>

## 7️⃣ How to Run
### Clone Repo & Install dependency

```python
$ git clone https://github.com/boostcampaitech4lv23nlp2/final-project-level3-nlp-13.git
$ cd final-project-level3-nlp-13
$ poetry install

```

### Set up Elastic Search

```python
$ bash install_elastic_search.sh
```

### Run

```python
$ python agent.py
```

<br/>

## 8️⃣  Future Works
- Preparation for the discontinuation of the free Twitter API service starting from February 9, 2023
- Improvement of keyword-based intent classification & entity recognition:
    - Try FastText embeddings
    - Transition to machine learning-based intent classifier & entity detector
- Enhancement of the generation model:
    - Addition of training data and preprocessing, further experimentation and optimization
- Improvement of service quality:
    - Addition of intent and answer templates to the database
- Enhancement of malicious tweet filtering:
    - Addition of training data focusing on sarcastic sentences
- Pretraining with Salient Span Masking:
    - Pretraining with masking applied to key BTS-related keywords
- Addition of post-reply chatbot features and event functionalities
- Transition from single-turn to multi-turn approach in the chatbot, considering conversation context for responses.

<br/>

## 9️⃣ Development Environment

- Collaboration tools: Notion, Slack, Huggingface, Wandb
- Settings
    - GPU: V100
    - Languages: Python==3.8.5
    - dependency: PyTorch == 1.13.1
