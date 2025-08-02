import streamlit as st
import time
import pandas as pd
import random
import numpy as np
import plotly.express as px
import datetime
import math
import json
import os
import joblib
import itertools
import psutil

# 初始化状态
time_gap = 3

Regression_model_name = "XGBoost-Regressor"
Reg_path = "xgb_reg_model_2025-08-03_00-49-04_r2_0.97.pkl"
xgb_reg_model = joblib.load(Reg_path)

# 加载MOC模型
MOC_model_name = "MOC-TabTransformer"
MOC_path = "tabtransformer_moc_model_2025-08-02_23-34-05_mAP0.8165.pkl"
MOC_model = joblib.load(MOC_path)

# 载入scaler
scaler_dir = "scalers"
uv_scaler = joblib.load(
    os.path.join(
        scaler_dir, [f for f in os.listdir(scaler_dir) if f.startswith("uv") and f.endswith(".pkl")][0]))
search_scaler = joblib.load(
    os.path.join(
        scaler_dir, [f for f in os.listdir(scaler_dir) if f.startswith("search") and f.endswith(".pkl")][0]))
ocr_scaler = joblib.load(
    os.path.join(
        scaler_dir, [f for f in os.listdir(scaler_dir) if f.startswith("ocr") and f.endswith(".pkl")][0]))

# 载入label mapping
label_mapping_path = "label_mapping.json"
with open(label_mapping_path, "r", encoding="utf-8") as f:
    label_mapping = json.load(f)


# 将ad_stra中的np.int64数字用label_mapping进行解码
def decode_ad_stra(results, mapping = label_mapping):
    material_labels = {v: k for k, v in mapping['material'].items()}
    placement_labels = {v: k for k, v in mapping['placement'].items()}
    payment_labels = {v: k for k, v in mapping['payment'].items()}
    sellingpoint_labels = {v: k for k, v in mapping['sellingpoint'].items()}

    decoded_results = []
    for combo in results:
        decoded = [
            material_labels.get(int(combo[0]), combo[0]),
            placement_labels.get(int(combo[1]), combo[1]),
            payment_labels.get(int(combo[2]), combo[2]),
            sellingpoint_labels.get(int(combo[3]), combo[3])
        ]
        decoded_results.append(decoded)
    return decoded_results


def ad_strategy_recommender_with_topk(input_uv, input_search, k_num=3):
    input_data = np.array([[input_uv, input_search]])  # shape=(1, 特征数)

    # 预测输出（每个维度的 label 概率）
    preds = MOC_model.predict(input_data)

    # 为每个输出维度提取 Top-K 候选标签索引及其概率
    topk_indices_per_dim = []
    topk_probs_per_dim = []

    for dim in preds:
        probs = dim[0]
        topk_idx = probs.argsort()[-k_num:][::-1]
        topk_probs = probs[topk_idx]
        topk_indices_per_dim.append(topk_idx)
        topk_probs_per_dim.append(topk_probs)

    # 笛卡尔积生成所有可能的组合及对应概率乘积
    strategy_candidates = list(itertools.product(*topk_indices_per_dim))
    score_candidates = []

    for combo in strategy_candidates:
        prob_product = 1.0
        for i in range(4):
            idx = list(topk_indices_per_dim[i]).index(combo[i])
            prob = topk_probs_per_dim[i][idx]
            prob_product *= prob
        score_candidates.append((combo, prob_product))

    # 选出置信度 Top-K 的组合
    top_k_sorted = sorted(score_candidates, key=lambda x: x[1], reverse=True)[:k_num]
    results = [item[0] for item in top_k_sorted]

    # 解码并输出
    decoded_results = decode_ad_stra(results)
    for i, decoded in enumerate(decoded_results):
        print(f"Top-{i+1} strategy candidate: {decoded}")

    return results, decoded_results


def ocr_predictor(raw_input):
    # 数值特征归一化
    uv_scaled = uv_scaler.transform(pd.DataFrame({'uv': [raw_input['uv']]}))[0][0]
    search_scaled = search_scaler.transform(pd.DataFrame({'search': [raw_input['search']]}))[0][0]

    # 类别特征编码
    material_encoded = label_mapping['material'][raw_input['material']]
    placement_encoded = label_mapping['placement'][raw_input['placement']]
    payment_encoded = label_mapping['payment'][raw_input['payment']]
    sellingpoint_encoded = label_mapping['sellingpoint'][raw_input['sellingpoint']]

    # 构造模型输入
    input_data = pd.DataFrame([{
        'uv': uv_scaled,
        'search': search_scaled,
        'material': material_encoded,
        'placement': placement_encoded,
        'payment': payment_encoded,
        'sellingpoint': sellingpoint_encoded
    }])

    # 预测
    prediction_scaled = xgb_reg_model.predict(input_data)
    # 反归一化
    prediction = ocr_scaler.inverse_transform([[prediction_scaled[0]]])[0][0]

    return prediction


# 构造一组输入
input_uv = 150
input_search = 450
k_num = 5
multi_k = 4

start_time = datetime.datetime.now()

_, decoded_ad_stra = ad_strategy_recommender_with_topk(input_uv, (input_search/100), k_num=k_num*multi_k)

top_k_strategy_ocr = {}

for i in range(0, multi_k * k_num):
    data_input = {
        'uv': input_uv,
        'search': input_search,
        'material': decoded_ad_stra[i][0],
        'placement': decoded_ad_stra[i][1],
        'payment': decoded_ad_stra[i][2],
        'sellingpoint': decoded_ad_stra[i][3]
    }
    ocr_value = ocr_predictor(data_input)
    top_k_strategy_ocr[i] = {
        'input_data': data_input,
        'predicted_ocr': float(ocr_value)
    }

end_time = datetime.datetime.now()
process = psutil.Process(os.getpid())
mem_info = process.memory_info()
print(f"\nMemory usage: {mem_info.rss / 1024 ** 2:.2f} MB")

end_time = datetime.datetime.now()
duration = (end_time - start_time).total_seconds()
print(f"\nTotal time taken: {duration:.4f} seconds")

# 按照 predicted_ocr 从大到小排序并打印，重置 idx 使 OCR 最高的为 Top-1
sorted_results = sorted(top_k_strategy_ocr.items(), key=lambda x: x[1]['predicted_ocr'], reverse=True)

print(f"\nTop-{k_num} AD Strategy Candidates with Predicted OCR:")
n = 1
for rank, (orig_idx, result) in enumerate(sorted_results, start=1):
    if n<=k_num:
        n += 1
        print(f"\nTop-{rank}. AD Strategy Candidate with Predicted OCR")
        print(f"Input Data: ", result['input_data'])
        print("OCR:\t {:.4f}".format(result['predicted_ocr']))
    else:
        break


# # 按钮触发状态更新
# if st.button("Next Step"):
#     st.session_state.next_clicked = True


st.markdown("<h1 style='text-align: center;'>🎯Localized Fine-Tuned E-Commerce AD ROI Prediction System DEMO</h1>",
            unsafe_allow_html=True)

# # Refresh button
# if st.button("🔄 Refresh Chart"):
#     st.rerun()

# Load the latest CSV
df = pd.read_csv("SYS_DEMO/simulateData.csv")
df = df.rename(columns={'data': 'date'})
df['date'] = pd.to_datetime(df['date'], format='%Y.%m.%d')

# Line selection
options = st.multiselect(
    "Select the series to display:",
    options=[col for col in df.columns if col != 'date'],
    default=['OCR']
)

# Display chart if selection exists
if options:
    df_melted = df.melt(id_vars='date', value_vars=options, var_name='Series', value_name='Value')

    fig = px.line(df_melted, x='date', y='Value', color='Series', markers=True)
    fig.update_layout(
        title={
            'text': "Performance Trends Over Time",
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.9
        },
        title_font=dict(size=24)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠️ Please select at least one data series to display.")


# Step 1: 店铺信息输入
st.markdown("<h2 style='text-align: center;'>1️⃣ Input Store Profile</h2>", unsafe_allow_html=True)
with st.form(key="form1"):

    date_input = st.date_input("Date of Shop Profile",
                              help="Used to record the time of the predicted data corresponding to the current input")

    input_info = {"avg. Daily UV": None, "avg. Daily Search Volume": None}

    input_info['avg. Daily UV'] = st.number_input(label="avg. Daily UV Volume", value=153)

    input_info['avg. Daily Search Volume'] = st.number_input(label="avg. Daily Search Volume", value=453)

    input_info['Top-K'] = st.number_input(label="Input K", value=5)

    info_submitted = st.form_submit_button(label='Submit it!')

    if info_submitted:
        st.success('Shop Info Submitted, the Tailor-Made ADs Combinations Generating...')
        st.balloons()

# 添加自定义CSS来调整表格宽度
st.markdown("""
<style>
    .stDataFrame {
        width: 100%;
    }
    .stDataFrame > div {
        width: 100%;
    }
    .stDataFrame table {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Step 2: 基于用户输入的广告策略推荐 - 显示完整的decoded_ad_stra
if info_submitted:
    st.divider()

    st.markdown("<h2 style='text-align: center;'>2️⃣ AD. Strategy Recommender</h2>",
                unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center;'>Power by {MOC_model_name} Model</h4>",
                unsafe_allow_html=True)
    st.markdown(f"<h5 style='text-align: center;'>Demonstration of 4K AD Strategies</h5>",
                unsafe_allow_html=True)

    with st.spinner("⏳ Generating tailor-made ad strategies..."):
        # 获取用户输入
        input_uv = input_info['avg. Daily UV']
        input_search = input_info['avg. Daily Search Volume']
        k_num = input_info['Top-K']  # 推荐的策略数量
        multi_k = 4  # 生成多个策略以便后续筛选

        # 调用MOC模型生成策略
        _, decoded_ad_stra = ad_strategy_recommender_with_topk(input_uv, (input_search/100), k_num=k_num*multi_k)

    st.success("✅ Done! Recommended strategies are ready.")

    # 创建DataFrame展示推荐的策略 (显示完整的decoded_ad_stra)
    strategy_data = []
    for i in range(len(decoded_ad_stra)):
        strategy_data.append([
            decoded_ad_stra[i][0],  # 素材
            decoded_ad_stra[i][1],  # 投放位置
            decoded_ad_stra[i][2],  # 支付方式
            decoded_ad_stra[i][3],  # 卖点
        ])

    strategy_df = pd.DataFrame(strategy_data,
                             columns=["CreativeFormat", "AdType", "PaymentMethod", "SellingPoint"])
    strategy_df.index = range(1, len(strategy_df) + 1)

    st.dataframe(strategy_df)
    st.balloons()

    # Step 3: 对推荐的策略进行OCR预测 - 显示sorted_results的前K个结果
    st.divider()
    st.markdown("<h2 style='text-align: center;'>3️⃣ ROI Predictor</h2>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center;'>Powered by {Regression_model_name} Model</h4>",
                unsafe_allow_html=True)
    st.markdown(f"<h5 style='text-align: center;'>4K-to-Top-K Strategy</h5>",
                unsafe_allow_html=True)

    with st.spinner("⏳ Generating High ROI ADs. strategies..."):
        # 存储策略和对应的OCR预测值
        top_k_strategy_ocr = {}

        # 对每个推荐的策略使用XGBoost模型预测OCR
        for i in range(len(decoded_ad_stra)):
            data_input = {
                'uv': input_uv,
                'search': input_search,
                'material': decoded_ad_stra[i][0],
                'placement': decoded_ad_stra[i][1],
                'payment': decoded_ad_stra[i][2],
                'sellingpoint': decoded_ad_stra[i][3]
            }
            ocr_value = ocr_predictor(data_input)
            top_k_strategy_ocr[i] = {
                'input_data': data_input,
                'predicted_ocr': float(ocr_value)
            }

        # 按OCR预测值从大到小排序
        sorted_results = sorted(top_k_strategy_ocr.items(), key=lambda x: x[1]['predicted_ocr'], reverse=True)

    st.success("✅ Done! ROI Prediction Complete.")

    # 创建DataFrame展示前K个带有OCR预测的策略
    ocr_strategy_data = []
    for rank, (orig_idx, result) in enumerate(sorted_results[:k_num], start=1):
        ocr_strategy_data.append([
            result['input_data']['material'],
            result['input_data']['placement'],
            result['input_data']['payment'],
            result['input_data']['sellingpoint'],
            result['predicted_ocr']
        ])

    ocr_strategy_df = pd.DataFrame(ocr_strategy_data,
                               columns=["CreativeFormat", "AdType", "PaymentMethod", "SellingPoint", "Predicted OCR"])
    ocr_strategy_df.index = range(1, len(ocr_strategy_df) + 1)

    st.dataframe(ocr_strategy_df)
    st.balloons()

    # # Step 4: 展示Top-K结果
    # st.divider()
    # st.markdown("<h2 style='text-align: center;'>4️⃣ Top-K Result Display</h2>", unsafe_allow_html=True)
    # st.markdown("<h3 style='text-align: center;'>Powered by K-Means Clustering</h3>", unsafe_allow_html=True)
    #
    # # 这部分可以保留原有的随机生成逻辑，或者进一步使用模型进行聚类分析
    # # 这里简单地展示前K个OCR最高的策略
    # st.dataframe(ocr_strategy_df)
    # st.balloons()

# Command:
# streamlit run ROI_SYS_DEMO.py
# Ctrl + C to Stop