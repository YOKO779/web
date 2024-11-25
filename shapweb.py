import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import matplotlib as mpl
from matplotlib import font_manager

# 加载字体 SimHei
font_path = "simhei.ttf"  # 字体文件路径
try:
    simhei_font = font_manager.FontProperties(fname=font_path)
    mpl.font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False  # 确保负号正常显示
    print("成功加载字体: SimHei")
except FileNotFoundError:
    simhei_font = None
    print("未找到 SimHei 字体文件，请检查路径或上传字体文件。")


# 设置页面配置（必须是第一条 Streamlit 命令）
st.set_page_config(page_title="老年糖尿病患者衰弱风险预测", layout="centered")

# 设置页面标题
st.title("老年糖尿病患者衰弱风险预测")

def main():
    # 加载模型
    model = joblib.load('xgb_model.pkl')

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
    plt.figure(figsize=(10, 2))  # 调整图片大小
    shap.force_plot(
        base_value,
        shap_values[0],
        df_subject.iloc[0, :],
        matplotlib=True
    )
    if simhei_font:
        plt.rcParams["font.family"] = simhei_font.get_name()  # 应用 SimHei 字体
    st.pyplot(plt.gcf())  # 显示图形

    # 保存图像为 PNG 文件
    plt.savefig("force_plot.png", bbox_inches="tight", dpi=300)
    plt.close()  # 关闭图形，防止内存占用
