from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 기존 코드에서 불러오는 부분
product_df = pd.read_excel('./data/product_preprocessed.xlsx')
review_embedded=pd.read_excel('./data/review_preprocessed_summerized_embedded.xlsx')
review_embedded['review_content_embedding'] = review_embedded['review_content_embedding'].apply(lambda x: eval(x) if isinstance(x, str) else x)
review_embedded['product_name_embedding' ]=review_embedded['product_name_embedding'].apply(lambda x: eval(x) if isinstance(x, str) else x)
review_embedded['리뷰어 정보_encoded'] = review_embedded['리뷰어 정보_encoded'].apply(lambda x: [int(item.strip().strip("'")) for item in x.strip('[]').split(',')])

# 모델 및 토크나이저 로드
model = AutoModel.from_pretrained("kakaobank/kf-deberta-base")
tokenizer = AutoTokenizer.from_pretrained("kakaobank/kf-deberta-base")

# 함수들 (기존 코드를 함수로 그대로 가져옵니다)
def calculate_score(product, user_input, priorities):
    weights = {"skin_type": 0, "skin_problem": 0, "reaction": 0}

    # 가중치 설정
    if len(priorities) == 1:
        weights[priorities[0]] = 0.6
        remaining = [key for key in weights.keys() if key not in priorities]
        weights[remaining[0]] = 0.2
        weights[remaining[1]] = 0.2
    elif len(priorities) == 2:
        weights[priorities[0]] = 0.4
        weights[priorities[1]] = 0.4
        remaining = [key for key in weights.keys() if key not in priorities]
        weights[remaining[0]] = 0.2
    elif len(priorities) == 3:
        weights = {"skin_type": 1/3, "skin_problem": 1/3, "reaction": 1/3}

    # 각 카테고리별 점수 계산
    skin_type_col = '피부 타입_{0}'.format(user_input["skin_type"])
    skin_problem_col = '피부 고민_{0}'.format(user_input["skin_problem"])
    reaction_col = '자극도_자극없음'

    skin_type_score = product[skin_type_col] * weights["skin_type"]
    skin_problem_score = product[skin_problem_col] * weights["skin_problem"]
    if user_input["reaction"] == "y":
        reaction_score = product[reaction_col] * weights["reaction"]*1.4
    else:
        reaction_score=product[reaction_col]*weights["reaction"]


    # 점수 편차 조정
    scores = [skin_type_score, skin_problem_score, reaction_score]
    average_score = sum(scores) / len(scores)
    adjusted_scores = [score + (average_score - score) * 0.5 for score in scores]

    final_score = sum(adjusted_scores)

    return final_score

#recommend_products 함수
def recommend_products1(products, user_input, priorities):
    scored_products = []

    for _, product in products.iterrows():
        score = calculate_score(product, user_input, priorities)
        scored_products.append({"name": product["제품명"], "score": score, "price":product['최종 가격']})

    # 점수에 따라 제품을 내림차순으로 정렬
    scored_products.sort(key=lambda x: x["score"], reverse=True)

    return scored_products

def embed_text(text):

 inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
 with torch.no_grad():
  model_output = model(**inputs)

# [CLS] 토큰의 출력(첫 번째 토큰)을 임베딩으로 사용
 embedding = model_output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
 return embedding



# 유사도 계산 함수
def calculate_cosine_similarity(query_embedding, embeddings):
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    return similarities

def get_top_products_by_similarity(query, reviews_df, filtered_products_df, concerns_list_user):
    # 쿼리 임베딩 생성
    query_embedding = embed_text(query)

    # 제품명과 리뷰 내용 임베딩 유사도 계산
    product_name_embeddings = np.array(reviews_df['product_name_embedding'].tolist())
    review_content_embeddings = np.array(reviews_df['review_content_embedding'].tolist())

    # 유사도 계산
    reviews_df['name_cosine_similarity'] = calculate_cosine_similarity(query_embedding, product_name_embeddings)
    reviews_df['review_cosine_similarity'] = calculate_cosine_similarity(query_embedding, review_content_embeddings)

    # 리뷰어와 사용자의 유사도 계산
    reviews_df['reviewer_user_cosine_similarity'] = reviews_df['리뷰어 정보_encoded'].apply(lambda x : cosine_similarity([x], [concerns_list_user])[0][0])

    # 유사도 평균 계산
    reviews_df['similarity_average'] = reviews_df[['name_cosine_similarity', 'review_cosine_similarity']].mean(axis=1)

    # '리뷰내용'을 기준으로 두 데이터프레임 병합
    merged_df = pd.merge(reviews_df, filtered_products_df, on='리뷰내용', how='inner', suffixes=('_review', '_product'))

    # 가중치를 적용한 최종 점수 계산 => 가중치를 곱하지 않고 유사도 평균이 그대로 final_score이 됨
    merged_df['final_score'] = merged_df['similarity_average']

    # 열 이름 정리: '제품명_review' 또는 '제품명_product' 중 하나를 선택
    product_column = '제품명_product'  # '제품명_product'로 설정

    if product_column not in merged_df.columns:
        raise KeyError(f"'{product_column}' column not found in merged_df. Available columns are: {merged_df.columns.tolist()}")

    # 한달사용기/재구매의사 에 해당하는 리뷰들만 한번씩 더 추가하기 => 해당 리뷰들의 영향력을 더 크게 하기 위함
    reliable_reviews = merged_df[(merged_df['한달이상사용']==1)|(merged_df['재구매여부']==1)]
    merged_df = pd.concat([merged_df, reliable_reviews])  # 영향력을 더 크게 하려면 여러번 추가해주면 됨

    # 제품별 final_score_result 계산 (제품별 리뷰의 final_score 평균)
    product_scores_df = merged_df.groupby(product_column)['final_score'].mean().reset_index()
    product_scores_df.rename(columns={'final_score': 'final_score_result'}, inplace=True)

    # 추가된 한달사용기/재구매의사 리뷰들 제거 (중복되어있으므로)
    merged_df = merged_df.drop_duplicates(subset=['리뷰내용'])


    # final_score_result 기준으로 상위 5개 제품 선택
    top_products_df = product_scores_df.sort_values(by='final_score_result', ascending=False).head(5)
    
    return top_products_df, merged_df

def recommend_products2(user_choice, product_df, top_products_df, merged_df):
    # 상위 5개 제품과 원본 데이터를 병합하여 상세 정보를 포함
    merged_top_df = pd.merge(merged_df, top_products_df, left_on='제품명_product', right_on='제품명_product', how='inner')

    # '최종 가격'과 '리뷰 점수'를 병합된 데이터프레임에 추가
    merged_top_df = pd.merge(merged_top_df, product_df[['제품명', '최종 가격', '리뷰 점수', '리뷰 수']], left_on='제품명_product', right_on='제품명', how='left')

     # 예상 별점 계산
    product_list = list(merged_top_df['제품명'].unique())
    merged_top_df['별점*유사도'] = merged_top_df['별점'] * merged_top_df['reviewer_user_cosine_similarity']
    merged_top_df['유사도합'] = 0
    merged_top_df['예상 별점'] = 0

    for product in product_list:
      target_df = merged_top_df[merged_top_df['제품명']==product]
      sim_sum = target_df['reviewer_user_cosine_similarity'].sum()
      merged_top_df.loc[merged_top_df['제품명']==product, '유사도합'] = sim_sum
      if sim_sum != 0:  # 유사도 합이 0이 아닌경우에만! 0인 경우 예상별점은 0점으로 처리
        merged_top_df.loc[merged_top_df['제품명']==product, '예상 별점'] = target_df['별점*유사도'].sum() / sim_sum

    # 제품명 기준으로 중복 제거 (첫 번째 리뷰만 선택)
    merged_top_df = merged_top_df.drop_duplicates(subset=['제품명_product'])

    # 추천 기준에 따른 정렬
    if user_choice == '인기순':
        merged_top_df['인기도'] = merged_top_df['리뷰 수'] * merged_top_df['리뷰 점수']
        sorted_df = merged_top_df.sort_values(by='인기도', ascending=False)
    elif user_choice == '관련도':
        sorted_df = merged_top_df.sort_values(by='final_score_result', ascending=False)
    elif user_choice == '가격 순':
        # 1차: 가격 오름차순, 2차: final_score_result 내림차순
        sorted_df = merged_top_df.sort_values(by=['최종 가격', 'final_score_result'], ascending=[True, False])
    elif user_choice == '평점 순':
        # 1차: 리뷰 점수 내림차순, 2차: final_score_result 내림차순
        sorted_df = merged_top_df.sort_values(by=['리뷰 점수', 'final_score_result'], ascending=[False, False])
    elif user_choice == '맞춤 추천':
        # 사용자와 각 리뷰어의 유사도와 각 리뷰어가 남긴 별점을 고려, 사용자가 남길 평점 예측
        # 1차: 예상 별점 내림차순, 2차: final_score_result 내림차순
        sorted_df = merged_top_df.sort_values(by=['예상 별점', 'final_score_result'], ascending=[False, False])
    else:
        raise ValueError("잘못된 선택입니다. '인기순', '관련도', '가격 순', '평점 순' 중에서 선택해 주세요.")

    # 최종 추천 결과에서 상위 5개 제품만 선택
    final_recommendations = sorted_df.head(5)

    # 최종 추천 결과 리턴
    return final_recommendations[['제품명_product', 'final_score_result', '최종 가격', '리뷰 점수']]


# 기본 라우트 - 사용자 입력을 받는 페이지
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        concerns_list = ['각질', '다크서클', '모공', '블랙헤드', '아토피', '잡티', '탄력', '트러블', '피지과다', '홍조']
        concerns_list_user = [0 for _ in range(len(concerns_list))]
        concerns=request.form.getlist('skin_problem_plus')
        try:
            for i in range(len(concerns_list)):
              if concerns_list[i] in concerns:
                concerns_list_user[i] = 1
        except Exception as e:
            print({e})
        
        # 사용자 입력 수집
        user_input = {
            "skin_type": request.form['skin_type'],
            "skin_problem": request.form['skin_problem'],
            "reaction": request.form['reaction'].lower(),
            "max_price": float(request.form['max_price']),
            "skin_problem_plus":concerns_list_user
        }
        priorities = request.form.getlist('priorities')
        query = request.form['query']
        user_choice = request.form['sort_order']

        try:
            # 리뷰 수가 50개 이상인 제품들만 필터링
            filtered_product_df = product_df[product_df['리뷰 수'] >= 50]
            # 사용자가 입력한 가격 상한선 이하의 제품들만 필터링
            filtered_product_df = filtered_product_df[filtered_product_df['최종 가격'] <= user_input["max_price"]]
            # 제품 추천
            recommendations = recommend_products1(filtered_product_df, user_input, priorities)
            # 상위 20개의 추천 제품만 선택
            if len(recommendations) >= 20:
                top_recommendations = recommendations[:20]
            else:
                top_recommendations = recommendations
            # 추천 제품을 DataFrame으로 저장
            recommended_df = pd.DataFrame(top_recommendations)

            # review_df를 추천된 제품명 순서에 따라 필터링 및 정렬
            filtered_products_df = review_embedded[review_embedded['제품명'].isin(recommended_df['name'])]
            # 추천된 제품의 순서에 맞게 정렬
            filtered_products_df = filtered_products_df.set_index('제품명').loc[recommended_df['name']].reset_index()
            #위의 코드에서 필터링된 df을 semantic search 용도로 사용
            reviews_df = filtered_products_df.loc[:, ['제품명', '리뷰내용','product_name_embedding','review_content_embedding', '리뷰어 정보_encoded']]
            
            
            
            try:
                # 시멘틱 서치 기반 추천
                top_products_df, merged_df = get_top_products_by_similarity(query, reviews_df, filtered_products_df, concerns_list_user)
                print("Top Products DF:", top_products_df)
                print("Merged DF:", merged_df)

                # 사용자가 선택한 기준으로 최종 추천
                result_df = recommend_products2(user_choice, product_df, top_products_df, merged_df)
                # 인덱스를 재설정하고, 기존 인덱스는 제거
                result_df = result_df.reset_index(drop=True)

                # 인덱스를 1부터 시작하도록 조정
                result_df.index = result_df.index + 1
            
                print("Result Df",result_df)
                

                # 결과를 index.html에 렌더링 (결과 포함)
                # return render_template('index.html', tables=[result_df.to_html(classes='data')], titles=result_df.columns.values)
                # 일부 열만 선택하여 전달
                return render_template('index.html', result_df=result_df, titles=result_df.columns.values)
                # return render_template('index.html', tables=[1,2,3,4])
             
            
            except Exception as e:
                print(f"Error in recommend_products or processing result_df: {e}")
                error_message = str(e)
                return render_template('index.html', error_message=error_message)
            
        except Exception as e:
            print(f"Error: {e}")
            return render_template('index.html', error_message=str(e))
        # 결과를 index.html에 렌더링 (결과 포함)

    # GET 요청 시 index.html 렌더링
    return render_template('index.html',result_df=None)

if __name__ == "__main__":
    app.run(debug=True)
