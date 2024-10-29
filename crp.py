import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import joblib
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
st.set_page_config(page_title="腐蚀速率预测与模型建立", layout="wide", initial_sidebar_state="expanded")

# 页面标题
st.title("腐蚀速率预测与模型建立")
st.markdown("---")  # 添加分隔线

# 切换功能
page = st.sidebar.radio("选择功能", ["预测", "建模/更新模型"])

# 预测功能
if page == "预测":
    st.header("预测界面")
    st.markdown("请上传您训练好的模型，并输入相应的参数进行预测。")

    # 上传模型
    model_file = st.sidebar.file_uploader("上传模型文件 (pkl)", type=["pkl"])

    if model_file:
        model = joblib.load(model_file)

        # 选择批量导入或手动输入
        input_method = st.sidebar.selectbox("选择输入方式", ["手动输入", "批量导入Excel"])

        if input_method == "手动输入":
            st.subheader("手动输入参数")
            input_data = {}
            parameters = [
                '环境电阻率（Ω·m）', '环境Ph', '工作面面积（cm2）', '时长（d）',
                '通电电位最大值（VCSE）', '通电电位最小值（VCSE）', '通电电位平均值（VCSE）',
                '断电电位最大值（VCSE）', '断电电位最小值（VCSE）', '断电电位平均值（VCSE）',
                '断电电位正于阴极保护准则比例', '断电电位正于阴极保护准则+50mV比例',
                '断电电位正于阴极保护准则+100mV比例', '断电电位正于阴极保护准则+850mV比例',
                '交流电压最大值（V）', '交流电压最小值（V）', '交流电压平均值（V）',
                '交流电流密度最大值（A/m2）', '交流电流密度最小值（A/m2）', '交流电流密度平均值（A/m2）',
                '直流电流密度平均值（A/m2）'
            ]
            for param in parameters:
                input_data[param] = st.sidebar.number_input(param, format="%.4f")

            # 预测按钮
            if st.sidebar.button("预测"):
                input_df = pd.DataFrame([input_data])  # 确保是二维数据框
                prediction = model.predict(input_df)
                st.subheader("预测结果")

                # 创建包含输入参数和预测值的 DataFrame
                result_df = pd.DataFrame(input_data, index=[0])  # 转换为 DataFrame
                result_df["预测值"] = prediction[0]  # 将预测值添加到 DataFrame 中

                st.dataframe(result_df.style.set_table_attributes("style='width:100%;'"))  # 表格铺满屏幕

        elif input_method == "批量导入Excel":
            st.subheader("批量导入Excel文件")
            excel_file = st.sidebar.file_uploader("上传Excel文件", type=["xlsx"])
            if excel_file:
                input_data = pd.read_excel(excel_file)
                predictions = model.predict(input_data)
                st.subheader("预测结果")

                # 创建结果 DataFrame，并确保有表头
                result_df = input_data.copy()
                result_df["预测值"] = predictions

                st.dataframe(result_df.style.set_table_attributes("style='width:100%;'"))  # 表格铺满屏幕

# 建模/更新模型功能
else:
    st.header("建模/更新模型界面")
    st.markdown("请上传您的数据集以训练模型。")

    # 上传数据集
    data_file = st.sidebar.file_uploader("上传数据集 (Excel)", type=["xlsx"])

    if data_file:
        data = pd.read_excel(data_file)

        # 数据预处理
        feature_columns = data.columns[:-1]  # 假设最后一列是标签
        label_column = data.columns[-1]  # 假设最后一列是标签
        data.fillna(data.mean(), inplace=True)

        # 划分数据集
        X = data[feature_columns]
        y = data[label_column]

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 选择回归模型（仅保留XGBoost）
        model_option = st.sidebar.selectbox("选择回归模型", ["XGBoost"])

        # 根据选择的模型初始化模型
        if st.sidebar.button("建模/更新模型"):
            if model_option == "XGBoost":
                model = xgb.XGBRegressor(objective='reg:squarederror')
                model.fit(X_train, y_train)

                # 保存模型
                joblib.dump(model, r'D:\model.pkl')
                st.success("模型已保存到D:\\model.pkl")

                # 可视化模型效果
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)

                # 特征重要性
                importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.zeros(len(feature_columns))
                importance_df = pd.DataFrame({'特征': feature_columns, '重要性': importance})
                importance_df = importance_df.sort_values(by='重要性', ascending=False)

                # 创建上下排列的图
                st.subheader("模型效果可视化")

                # 第一个图：真实值与预测值的对比

                fig1 = px.scatter(x=y_train, y=y_train_pred, title=f'训练集真实值与预测值的对比 (R²: {train_r2:.2f})',
                                  color_discrete_sequence=["blue"])
                fig1.add_scatter(x=[y_train.min(), y_train.max()], y=[y_train.min(), y_train.max()], mode='lines',
                                 name='理想预测线', line=dict(color='red', dash='dash'))
                fig1.update_layout(xaxis_title='真实值', yaxis_title='预测值')  # 设置 X 和 Y 轴标题

                # 第二个图：测试集真实值与预测值的对比
                fig2 = px.scatter(x=y_test, y=y_test_pred, title=f'测试集真实值与预测值的对比 (R²: {test_r2:.2f})',
                                  color_discrete_sequence=["green"])
                fig2.add_scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines',
                                 name='理想预测线', line=dict(color='red', dash='dash'))
                fig2.update_layout(xaxis_title='真实值', yaxis_title='预测值')  # 设置 X 和 Y 轴标题

                # 显示两个图上下排列
                st.plotly_chart(fig1)
                st.plotly_chart(fig2)

                # 第二个图：特征重要性
                st.subheader("特征重要性")
                fig3 = px.bar(importance_df, x='特征', y='重要性', title='特征重要性排序')
                st.plotly_chart(fig3)

# 结束
st.markdown("---")  # 添加分隔线
