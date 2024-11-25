import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from matplotlib import font_manager, rcParams

import matplotlib as mpl
import matplotlib.pyplot as plt

# 临时注册新的全局字体
mpl.font_manager.fontManager.addfont('字体/SimHei.ttf')  # 指定字体路径
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体为 SimHei
plt.rcParams["axes.unicode_minus"] = False   # 确保负号正常显示



# 设置页面配置（必须是第一条 Streamlit 命令）
st.set_page_config(page_title="老年糖尿病患者衰弱风险预测", layout="centered")


def main():
    # 加载 XGBoost 模型
    model = joblib.load('xgb_model.pkl')

  from matplotlib import font_manager, rcParams

font_path = "C:/Windows/Fonts/simsunb.ttf"
font = font_manager.FontProperties(fname=font_path)
print("加载字体名称:", font.get_name())

shap.force_plot(
    base_value,
    shap_values[0],
    df_subject.iloc[0, :],
    matplotlib=True
)

# 显示绘图
st.pyplot(plt.gcf())

# 保存图像
plt.savefig("force_plot.png", bbox_inches="tight", dpi=300)
plt.close()



    # 定义用户输入的类
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

            # SHAP 可视化
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_subject)

            # 获取基值
            if isinstance(explainer.expected_value, list):
                base_value = explainer.expected_value[0]
            else:
                base_value = explainer.expected_value

            # 绘制 SHAP force_plot
            shap.force_plot(
                base_value,
                shap_values[0],
                df_subject.iloc[0, :],
                matplotlib=True
            )
            st.pyplot(plt.gcf())  # 显示图形

            # 保存图像为 PNG 文件
            plt.savefig("force_plot.png", bbox_inches="tight", dpi=300)
            plt.close()  # 关闭图形，防止内存占用

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
