import streamlit as st
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # 设置页面配置
    st.set_page_config(page_title='老年糖尿病患者衰弱风险预测')

    # 加载模型
    lgbm = joblib.load('xgb_model.pkl')  # 确保路径正确

    # 特征名称映射字典
    feature_name_mapping = {
        "查尔斯共病指数": "CCI指数",
        "认知障碍": "认知能力",
        "体育锻炼运动量": "体育锻炼运动量",
        "慢性疼痛": "慢性疼痛",
        "营养状态": "营养状态",
        "HbA1c": "HbA1c",
        "步速下降": "步速下降",
    }

    # 定义输入特征
    class Subject:
        def __init__(self, 认知障碍, 体育锻炼运动量, 慢性疼痛, 营养状态, HbA1c, 查尔斯共病指数, 步速下降):
            self.认知障碍 = 认知障碍
            self.体育锻炼运动量 = 体育锻炼运动量
            self.慢性疼痛 = 慢性疼痛
            self.营养状态 = 营养状态
            self.HbA1c = HbA1c
            self.查尔斯共病指数 = 查尔斯共病指数
            self.步速下降 = 步速下降

        def make_predict(self):
            # 将输入数据转化为 DataFrame
            subject_data = {
                "查尔斯共病指数": [self.查尔斯共病指数],
                "认知障碍": [self.认知障碍],
                "体育锻炼运动量": [0 if self.体育锻炼运动量 == "低运动量" else 1 if self.体育锻炼运动量 == "中运动量" else 2],
                "慢性疼痛": [self.慢性疼痛],
                "营养状态": [0 if self.营养状态 == "营养良好" else 1 if self.营养状态 == "营养不良风险" else 2],
                "HbA1c": [self.HbA1c],
                "步速下降": [self.步速下降],
            }

            df_subject = pd.DataFrame(subject_data)

            # 映射特征名称
            df_subject.rename(columns=feature_name_mapping, inplace=True)

            # 模型预测
            try:
                prediction = lgbm.predict_proba(df_subject)[:, 1]
                adjusted_prediction = np.round(prediction * 100, 2)
                st.write(f"模型预测老年糖尿病患者衰弱风险为 {adjusted_prediction[0]} %")

                # SHAP 可视化
                explainer = shap.TreeExplainer(lgbm)
                shap_values = explainer.shap_values(df_subject)

                # 绘制 SHAP 力图
                shap.force_plot(explainer.expected_value, shap_values[0], df_subject.iloc[0, :], matplotlib=True)
                st.pyplot(plt.gcf())
                plt.clf()  # 清空画布
            except Exception as e:
                st.error(f"预测时发生错误：{str(e)}")

    # 页面标题
    st.markdown("""
        <div style='text-align: center;'>
            <h1>老年糖尿病患者衰弱风险预测</h1>
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

    # 提交按钮
    if st.button(label="提交"):
        user = Subject(认知障碍, 体育锻炼运动量, 慢性疼痛, 营养状态, HbA1c, 查尔斯共病指数, 步速下降)
        user.make_predict()


main()
