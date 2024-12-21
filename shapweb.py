import streamlit as st
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_predict(self):
    # 将输入数据转化为 DataFrame
    subject_data = {
        "认知障碍": [self.认知障碍],
        "体育锻炼运动量": [self.体育锻炼运动量],
        "慢性疼痛": [self.慢性疼痛],
        "营养状态": [self.营养状态],
        "HbA1c": [self.HbA1c],
        "查尔斯共病指数": [self.查尔斯共病指数],
        "步速下降": [self.步速下降],
        "糖尿病肾病": [self.糖尿病肾病],
    }

    df_subject = pd.DataFrame(subject_data)

    # 对分类变量进行编码
    df_subject = pd.get_dummies(df_subject, drop_first=True)  # drop_first=True 避免虚拟变量陷阱

    # 模型预测
    prediction = lgbm.predict_proba(df_subject)[:, 1]
    adjusted_prediction = np.round(prediction * 100, 2)
    st.write(f"""
        <div class='all'>
            <p style='text-align: center; font-size: 20px;'>
                <b>模型预测老年糖尿病患者衰弱风险为 {adjusted_prediction[0]} %</b>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # SHAP 可视化
    explainer = shap.Explainer(lgbm)
    shap_values = explainer.shap_values(df_subject)

    # 绘制 SHAP 力图
    shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], df_subject.iloc[0, :], matplotlib=True)
    st.pyplot(plt.gcf())


    # 设置页面配置
    st.set_page_config(page_title='老年糖尿病患者衰弱风险预测')

    # 页面标题
    st.markdown(f"""
                <div class='all'>
                    <h1 style='text-align: center;'>老年糖尿病患者衰弱风险预测</h1>
                </div>
                """, unsafe_allow_html=True)

    # 输入特征
    认知障碍 = st.selectbox("认知障碍 (是 = 1, 否 = 0)", [1, 0], index=1)
    体育锻炼运动量 = st.selectbox("体育锻炼运动量", ["低运动量", "中运动量", "高运动量"], index=0)
    慢性疼痛 = st.selectbox("慢性疼痛 (有 = 1, 无 = 0)", [1, 0], index=1)
    营养状态 = st.selectbox("营养状态", ["营养良好", "营养不良风险", "营养不良"], index=0)
    HbA1c = st.number_input("HbA1c (mmol/L)", value=7.0, min_value=4.0, max_value=20.0)
    查尔斯共病指数 = st.number_input("查尔斯指数", value=2, min_value=0, max_value=30)
    步速下降 = st.selectbox("步速下降 (是 = 1, 否 = 0)", [1, 0], index=1)
    糖尿病肾病 = st.selectbox("糖尿病肾病 (有 = 1, 无 = 0)", [1, 0], index=1)

    # 提交按钮
    if st.button(label="提交"):
        user = Subject(认知障碍, 体育锻炼运动量, 慢性疼痛, 营养状态, HbA1c, 查尔斯共病指数, 步速下降, 糖尿病肾病)
        user.make_predict()



main()
