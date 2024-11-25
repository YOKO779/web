import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
# 设置中文字体，避免中文乱码
rcParams['font.family'] = 'SimHei'  # 黑体 (SimHei) for Windows or macOS 可以使用 'Songti'
rcParams['axes.unicode_minus'] = False  # 防止负号显示问题

def main():
    # 加载模型
    model = joblib.load('xgb_model.pkl')

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
            # 设置字体（使用 Noto Sans SC）
            import matplotlib.pyplot as plt
            from matplotlib import rcParams

            rcParams['font.sans-serif'] = ['Noto Sans SC']  # 使用 Google Fonts 的 Noto Sans SC
            rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

            # 数据映射
            subject_data = {
                "认知障碍": [self.认知障碍],
                "体育锻炼运动量": [self.体育锻炼运动量],
                "慢性疼痛": [self.慢性疼痛],
                "营养状态": [self.营养状态],
                "HbA1c": [self.HbA1c],
                "查尔斯共病指数": [self.查尔斯共病指数],
                "步速下降": [self.步速下降]
            }
            df_subject = pd.DataFrame(subject_data)

            # 模型预测
            prediction = model.predict_proba(df_subject)[:, 1]
            adjusted_prediction = np.round(prediction * 100, 2)
            st.write(f"""
                <div style="text-align: center; font-size: 20px;">
                    <b>模型预测衰弱风险为: {adjusted_prediction[0]}%</b>
                </div>
            """, unsafe_allow_html=True)

            # SHAP 可视化部分
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_subject)

            # 检查 SHAP 的输出
            if len(shap_values) == 0:
                st.error("SHAP 计算失败，请检查输入数据或模型。")
                return

            # 获取基值（expected_value）
            if isinstance(explainer.expected_value, list):
                base_value = explainer.expected_value[0]
            else:
                base_value = explainer.expected_value

            # 绘制 SHAP force_plot 图
            shap.force_plot(
                base_value,
                shap_values[0],         # 使用第一个样本的 SHAP 值
                df_subject.iloc[0, :],  # 输入第一个样本数据
                matplotlib=True
            )
            st.pyplot(plt.gcf())  # 将绘图渲染到 Streamlit

            # 保存 SHAP 图为 PDF 文件
            plt.savefig("force_plot.pdf", bbox_inches="tight")
            plt.close()  # 关闭图形，防止后续绘图冲突

    # Streamlit 页面配置
    st.set_page_config(page_title="老年糖尿病患者衰弱风险预测", layout="centered")

    # 输入字段
    认知障碍 = st.selectbox("认知障碍 (1: 是, 0: 否)", [1, 0], index=1)
    体育锻炼运动量 = st.selectbox("体育锻炼运动量 (1: 低, 2: 中, 3: 高)", [1, 2, 3], index=0)
    慢性疼痛 = st.selectbox("慢性疼痛 (1: 有, 0: 无)", [1, 0], index=1)
    营养状态 = st.selectbox("营养状态 (0: 营养良好, 1: 营养不良风险, 2: 营养不良)", [0, 1, 2], index=0)
    HbA1c = st.number_input("HbA1c 值 (mmol/L)", value=7.0, min_value=4.0, max_value=20.0, step=0.1)
    查尔斯共病指数 = st.number_input("查尔斯共病指数", value=4, min_value=0, max_value=20, step=1)
    步速下降 = st.selectbox("步速下降 (1: 是, 0: 否)", [1, 0], index=1)

    # 提交按钮
    if st.button("提交"):
        user = Subject(认知障碍, 体育锻炼运动量, 慢性疼痛, 营养状态, HbA1c, 查尔斯共病指数, 步速下降)
        user.make_predict()


if __name__ == "__main__":
    main()
