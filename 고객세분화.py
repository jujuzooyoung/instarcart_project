import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


def data():
    aisles = pd.read_csv('drive-download-20250325T061508Z-001/aisles.csv')
    departments = pd.read_csv('drive-download-20250325T061508Z-001/departments.csv')
    orders = pd.read_csv('drive-download-20250325T061508Z-001/orders.csv')
    products = pd.read_csv('drive-download-20250325T061508Z-001/products.csv')
    order_products_prior = pd.read_csv('drive-download-20250325T061508Z-001/order_products__prior.csv')
    aisle_price = pd.read_csv('drive-download-20250325T061508Z-001/aisle_price.xlsx - Sheet1 (1).csv')
    
    aisle_price = aisle_price.drop('number', axis = 1)
    
    print('데이터 불러오기 완료')

    orders = orders[orders['eval_set'] != 'train']
    orders = orders[orders['eval_set'] != 'test']
    
    print('orders 수정 완료')
    
    merged_products = order_products_prior.merge(products, on="product_id", how="left")
    merged_products = merged_products.merge(aisles, on = 'aisle_id', how = 'left')
    merged_products = merged_products.merge(departments, on = 'department_id', how = 'left')
    merged_products = merged_products.merge(orders, on = 'order_id', how = 'left')
    merged_products = merged_products.merge(aisle_price, on = 'aisle', how = 'left')
    merged_products = merged_products.drop('eval_set', axis = 1)
    
    merged_products = merged_products.rename(columns={'price':'aisle_price'})
    
    print("데이터 merge 완료")
    
    
    
       # 1. 평균 주문 간격 계산
    Recency = merged_products.groupby('user_id')['days_since_prior_order'].mean().reset_index()
    Recency.rename(columns={'days_since_prior_order': 'avg_days_between_orders'}, inplace=True)
    
    # 2. 총 주문 횟수 계산
    Frequency = merged_products.groupby('user_id')['order_number'].max().reset_index()
    Frequency.rename(columns={'order_number': 'total_orders'}, inplace=True)
    
    # 3. 병합 (Recency로 사용)
    Recency = pd.merge(Recency, Frequency, on='user_id')
    
    # 4. custom_recency 계산
    Recency['custom_recency'] = (1 / (Recency['avg_days_between_orders'] + 1)) * Recency['total_orders']
    # 5. 정렬 및 필요한 컬럼만 남기기
    Recency = Recency.sort_values(by='custom_recency', ascending=False)
    Recency = Recency[['user_id', 'custom_recency']]
    
    Monetary = merged_products.groupby('user_id')[['aisle_price']].sum().sort_values(by = 'aisle_price', ascending = False).reset_index() #주문한 상품 수

    
    print('RFM 값 정렬 완료')
    
    
    
    # 1. Recency 점수 계산 (최근성 점수)
    Recency['R_score'] = pd.qcut(Recency['custom_recency'], 5, labels=False, duplicates='drop') + 1
    
    # 2. Frequency 점수 계산 (주문 횟수 점수)
    Frequency['F_score'] = pd.qcut(Frequency['total_orders'], 5, labels=False, duplicates='drop') + 1
    
    # 3. Monetary 점수 계산 (주문한 상품 수 점수)  
    Monetary['M_score'] = pd.qcut(Monetary['aisle_price'], 5, labels=False, duplicates='drop') + 1
    
    print('RFM 각각 계산 완료')
    
    
    
    rfm_df = pd.merge(Recency[['user_id', 'R_score']], Frequency[['user_id', 'F_score']], on='user_id', how='left')
    rfm_df = pd.merge(rfm_df, Monetary[['user_id', 'M_score']], on='user_id', how='left')
    
    # 5. 가중치를 반영한 RFM 점수 계산
    rfm_df['RFM_score'] = rfm_df['R_score'] * 0.2 + rfm_df['F_score'] * 0.5 + rfm_df['M_score'] * 0.3  # 가중치 부여
    
    # 6. 가중치 기반 RFM 점수를 1~10 사이로 조정 (점수화)
    rfm_df['RFM_score'] = pd.qcut(rfm_df['RFM_score'],5, labels=False, duplicates='drop') + 1  # 점수화
    
    print('전체 RFM_score 계산 완료')
    
    
    
    # 7. 충성도 분류 (1~5 이탈율 높은 고객, 6~8 중간 고객, 9~10 충성도 높은 고객)

    for_merge = rfm_df[['user_id', 'Loyalty']]
    
    
    
    merged_products = merged_products.merge(for_merge, on = 'user_id', how = 'left')
    merged_products = merged_products[merged_products['aisle'] != 'missing']
    merged_products = merged_products[merged_products['aisle'] != 'other']

    print('최종 데이터 완성')

    return merged_products