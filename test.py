import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
import pickle 
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt 
from copy import deepcopy
class CoxModelPredict():
    def __init__(self,cox_model_path,shap_model_path):
        self.feature_col_list=["治疗方式2","治疗方式3","cN分期","腹水分级","腹膜转移评分二分","肝脏转移"]
        self.load_model(cox_model_path,shap_model_path)
    
    def load_model(self,cox_model_path,shap_model_path):
        self.model=pickle.load(open(cox_model_path,"rb"))
        self.shap_explainer=pickle.load(open(shap_model_path,"rb"))

    def predict_one_case(self,threshold=0.74,case_feature=None,time_point_list=[]):
        xdata=deepcopy(case_feature)
        xdata["治疗方式1"]=(xdata["治疗方式"]==1).map(int)
        xdata["治疗方式2"]=(xdata["治疗方式"]==2).map(int)
        del xdata["治疗方式"]
        risk=self.model.predict_partial_hazard(xdata).iloc[0]
        group="低危组" if risk<=threshold else "高危组"
        rel={"risk":risk,"group":group}
        for time_point in time_point_list:
            survival_prob=self.model.predict_survival_function(xdata,times=time_point).iloc[0,0]
            rel[time_point]=survival_prob
        return rel


    def plt_survival_curve_one_case(self,case_feature=None):
        pass
    def shap_one_case(self,case_feature=None):
        shap_values=self.shap_explainer(case_feature)
        shap.plots.waterfall(shap_values[0,:], max_display=20,show=False) 
        plt.rcParams['axes.unicode_minus'] = False 
        st.pyplot()
    
    def analyze_survival_all(self,df,event_col,time_col,time_point_list):
        pass
    

    def predict_group(self,):
        pass
    def shape_group(self):
        pass
    def plot_group_at_timepoint(self):
        pass
        
# cox_model_path="cox_model.pkl"
# shap_model_path="shap_model.pkl"
# c=CoxModelPredict(cox_model_path,shap_model_path)
# c.predict_one_case(case_feature_dic={"治疗方式2":1,"治疗方式3":1,"cN分期":1,"腹水分级":1,"腹膜转移评分二分":1,"肝脏转移":1},time_point_list=[10,20,40])


model_path="./cox_model.pkl"
shap_path="./shap_model.pkl" 



# def analyze_one_case(c,case_feature,time_point,ci_legend):
def frontend():
    c=CoxModelPredict(model_path,shap_path)
    # 页面配置
    st.set_page_config(
        page_title="数据分析平台",
        page_icon="📊",
        layout="wide"
    )

    # 应用标题
    st.title("📈 数据分析平台")
    st.markdown("---")

    # 初始化session state
    if 'case_results' not in st.session_state:
        st.session_state.case_results = None
    if 'group_results' not in st.session_state:
        st.session_state.group_results = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    # 侧边栏导航
    st.sidebar.title("导航")
    analysis_type = st.sidebar.radio(
        "选择分析类型",
        ["Case 分析", "Group 分析"]
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        ### 使用说明
        1. **Case 分析**: 输入个体指标进行分析
        2. **Group 分析**: 上传Excel文件进行批量分析
        """
    )

    # 主内容区域
    if analysis_type == "Case 分析":
        st.header("🔍 Case 分析")
        
        # 创建两列布局
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 输入表单
            with st.form("case_input_form"):
                st.subheader("输入指标")
                
                # 基础信息
                st.markdown("#### case信息")
                case_id = st.text_input("Case ID", value="CASE_001")
                analysis_date = st.date_input("分析日期", value=datetime.now().date())
                
                # 指标输入 - 分为几个部分
                st.markdown("#### case指标")
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    f1 = st.number_input("治疗方式", min_value=1, max_value=3, value=1,step=1,help="1:联合治疗")
                    f2= st.number_input("腹水分级", min_value=1, max_value=3, value=2,step=1)
                    f3 = st.number_input("肝脏转移", min_value=0, max_value=1, value=1,step=1)
                    
                with col1_2:
                    f4 = st.number_input("腹膜转移评分", min_value=0,max_value=10, step=1,value=5)
                    f5 = st.number_input("cN分期", min_value=0, value=80)

                #  st.markdown("#### 配置参数")
                st.markdown("#### 时间设置")
                col_time1, col_time2, = st.columns([1,1])
                
                with col_time1:
                    time_unit = st.selectbox(
                        "时间单位",
                        options=["年", "月", "天"],
                        index=0,
                        help="选择生存概率的时间单位"
                    )
                
                with col_time2:
                    time_points_input = st.text_input(
                        f"时间点({time_unit})",
                        value="1,2,3,4,5",
                        help=f"输入时间点，用逗号分隔"
                    )

                st.markdown("#### 置信区间设置")
                confidence_level = st.slider(
                                "置信水平",
                                min_value=0.90,
                                max_value=0.99,
                                value=0.95,
                                step=0.01
                            )
            
                # 提交按钮
                submitted = st.form_submit_button("开始分析", type="primary")
        
        with col2:
            st.subheader("📋 当前输入概览")
            if submitted or st.session_state.case_results is not None:
      

                input_summary={"腹膜转移评分":f4,"cN分期":f5,"肝脏转移":f3,"腹水分级":f2,"治疗方式":f1}
                feature_map={"腹膜转移评分二分":0 if f4<2 else 1,"cN分期":f5,"肝脏转移":f3,"腹水分级":f2,"治疗方式":f1}
                summary_df = pd.DataFrame(list(input_summary.items()), 
                                        columns=["参数","值"],index=input_summary.keys())
                case_feature=pd.Series(feature_map).to_frame().T
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # 结果展示区域
        if submitted:
     
            
            # 这里放置后端计算逻辑
            # 示例：模拟计算过程
            with st.spinner("正在进行分析计算..."):
                time_point_list=[1,2,3,4,5]
                case_results=c.predict_one_case(case_feature=case_feature,time_point_list=time_point_list)

                
                # 保存结果到session state
                st.session_state.case_results = case_results
            
            if st.session_state.case_results is not None:
                if st.button("🔄 分析新Case", type="secondary", use_container_width=True):
                    st.session_state.case_results = None
                    st.rerun()
            st.markdown("---")



            st.subheader("📊 特征shap分析")
            c.shap_one_case(case_feature)

            # 显示结果

            st.subheader("📊 分析结果")
            if st.session_state.case_results:
                results = st.session_state.case_results
                
                # 结果卡片
                cols = st.columns(2)
                with cols[0]:
                    st.metric("预测分数", f"{results['risk']:.3f}")
                with cols[1]:
                    st.metric("风险等级", results['group'])
                for i in range(len(time_point_list)):
                    with cols[i%2]:
                        st.metric(f"{time_point_list[i]}年生存预测概率", f"{results[time_point_list[i]]:.2%}")
                
                # 详细结果
                # with st.expander("查看详细结果", expanded=True):
                #     st.write(f"**主要发现**: {results['主要发现']}")
                #     st.write("**建议措施**:")
                #     for i, action in enumerate(results['建议措施'], 1):
                #         st.write(f"{i}. {action}")
                
                # 下载按钮
                results_df = pd.DataFrame([results])
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="下载分析结果 (CSV)",
                    data=csv,
                    file_name=f"case_analysis_{case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    else:  # Group 分析
        st.header("👥 Group 分析")
        
        # 文件上传区域
        st.subheader("📁 数据上传")
        # if st.session_state.get('uploaded_file') is not None:
        if st.button("🗑️ 清除当前文件并上传新数据", type="secondary", use_container_width=True):
            st.session_state.uploaded_file = None
            st.rerun()
        st.markdown("---")

        st.session_state.uploaded_file = st.file_uploader(
            "上传Excel文件",
            type=['xlsx', 'xls'],
            help="请上传包含分析数据的Excel文件，确保包含必要的列"
        )

        if st.session_state.uploaded_file is not None:
            
            # 读取并预览数据
            try:

                df = pd.read_excel(st.session_state.uploaded_file)
                st.success(f"✅ 文件上传成功！共读取到 {len(df)} 行数据，{len(df.columns)} 个字段")

                
                # 显示数据预览
                with st.expander("查看数据预览", expanded=True):
                    st.dataframe(df.head(max(df.shape[0],100)), use_container_width=True)
                    
                    # 显示数据统计信息
                    st.subheader("📈 数据统计摘要")
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    with col_stats1:
                        st.metric("总样本数", len(df))
                    with col_stats2:
                        st.metric("变量数量", len(df.columns))
                    with col_stats3:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        st.metric("数值变量", len(numeric_cols))

                st.markdown("---")
                st.subheader("数据核验")
                with st.form("数据核验"):
                    col1,col2=st.columns(2)
                    with col1:
                        event_col=st.text_input("请输入事件列名称",value="是否死亡")
                    with col2:
                        time_col=st.text_input("请输入时间列名称",value="OS")
                    
                    start_check=st.form_submit_button("开始数据核验", type="primary")

                if start_check:
                    st.session_state.check_state="good"
                    all_need_col_dict={event_col:[float,[0,1]],time_col:[float,None],"腹膜转移评分":[float,set(list(range(11)))],"腹水分级":[float,set([0,1,2,3])],"肝脏转移":[float,[1,0]],"治疗方式":[float,set([1,2,3])],"cN分期":[float,set([0,1,2,3])]}
                    st.session_state.event_col=event_col
                    st.session_state.time_col=time_col
                    
                    bad_col=[]
                    for col,col_rule in all_need_col_dict.items():     
                        if col not in df.columns:
                            bad_col.append(col)
                    if len(bad_col)!=0:
                        st.session_state.check_state="bad"
                        st.error(f"缺失必要列{bad_col}")
                    else:
                        st.success(f"✅ 核验1:列名核验成功！！！")

                    bad_col=[]
                    for col,col_rule in all_need_col_dict.items():     
                        if col in df.columns:
                            try:
                                df[col]=df[col].map(col_rule[0])
                            except:
                                bad_col.append(col)
                    if len(bad_col)!=0:
                        st.session_state.check_state="bad"
                        st.error(f"下面列含有非数字取值{bad_col}")
                    else:
                        st.success(f"✅ 核验2:数字核验成功！！！")

                    bad_col=[]
                    for col,col_rule in all_need_col_dict.items():     
                        if col in df.columns and col_rule[1]:
                            for i in df[col].unique():
                                if i not in col_rule[1]:
                                    bad_col.append(col)
                                    break
                    if len(bad_col)!=0:
                        st.session_state.check_state="bad"
                        st.error(f"下面列含有不符取值{bad_col}")
                    else:
                        st.success(f"✅ 核验3:取值核验成功！！！")


                    if st.session_state.check_state=="good":
                        st.success(f"✅ 数据核验成功！可以进行下一步分析了！！！")
                        
                st.markdown("---")
                st.subheader("⚙️ 整体预后分析")
                with st.form("整体预后分析"):
                    col1,col2=st.columns(2)
                    with col1:
                        # 选择ID列
                        event_column = st.selectbox(
                            "选择事件列",
                            options=df.columns.tolist(),
                            index=df.columns.tolist().index(st.session_state.event_col)
                        )
                    with col2:
                        # 选择分组变量
                        time_column = st.selectbox(
                            "选择时间列",
                            options=df.columns.tolist(),
                            index=df.columns.tolist().index(st.session_state.time_col)
                        )
                    submitter=st.form_submit_button("开始整体生存分析",type="primary")
                if submitter:
                    if st.session_state.check_state!="good":
                        st.info("请先进行数据核验并成功")
                    else:
                        pass

                # 分析控制面板
                st.markdown("---")
                st.subheader("⚙️ 分析设置")
                
                with st.form("group_analysis_settings"):
                    col_set1, col_set2 = st.columns(2)
                    
                    with col_set1:
                        # 选择ID列
                        event_column = st.selectbox(
                            "选择事件列",
                            options=df.columns.tolist(),
                            index=df.columns.tolist().index(st.session_state.event_col)
                        )
                        
                        # 选择分组变量
                        time_column = st.selectbox(
                            "选择时间列",
                            options=df.columns.tolist(),
                            index=df.columns.tolist().index(st.session_state.time_col)
                        )
                        
                        timepoint_list= st.multiselect(
                            "要分析的时间点(年)",
                            options=[1,2,3,4,5],
                            default=[1,2,3,4,5]
                        )
                    with col_set2:
                        # 选择分析变量
                        analysis_columns = st.multiselect(
                            "选择分析变量",
                            options=df.columns.tolist(),
                            default=["腹膜转移评分","腹水分级","肝脏转移","治疗方式","cN分期"]
                        )
                        
                        # 设置分析参数
                        confidence_level = st.slider(
                            "置信水平",
                            min_value=0.90,
                            max_value=0.99,
                            value=0.95,
                            step=0.01
                        )
                    # 开始分析按钮
                    start_analysis = st.form_submit_button("开始批量分析", type="primary")
                
                if start_analysis:

                    if st.session_state.check_state!="good":
                        st.info("请先进行数据核验并成功")
                    else:

                        st.markdown("---")
                        st.subheader("📊 分析结果")
                        
                        with st.spinner("正在进行批量分析计算..."):
                            # TODO: 在这里调用你的模型进行批量计算
                            # 模拟数据处理过程
                            progress_bar = st.progress(0)
                            
                            for i in range(100):
                                # 模拟计算进度
                                # 这里替换为你的实际计算逻辑
                                time.sleep(0.01)
                                progress_bar.progress(i + 1)
                            
                            # 示例结果
                            placeholder_group_results = {
                                "total_cases": len(df),
                                "analysis_completed": len(df),
                                "summary_stats": {
                                    "mean_values": df[analysis_columns].mean().to_dict() if analysis_columns else {},
                                    "std_values": df[analysis_columns].std().to_dict() if analysis_columns else {}
                                },
                                "results_dataframe": pd.DataFrame({
                                    'ID': df[id_column] if id_column in df.columns else range(len(df)),
                                    'score': np.random.randn(len(df)),
                                    'category': ['A', 'B', 'C'] * (len(df)//3 + 1)[:len(df)]
                                })
                            }
                            
                            st.session_state.group_results = placeholder_group_results
                    
                    # 显示分析结果
                    if st.session_state.group_results:
                        results = st.session_state.group_results
                        
                        # 结果概览
                        st.success(f"✅ 分析完成！共处理 {results['total_cases']} 个案例")
                        
                        # 统计摘要表格
                        st.subheader("📈 统计摘要")
                        if results['summary_stats']['mean_values']:
                            stats_df = pd.DataFrame({
                                '变量': list(results['summary_stats']['mean_values'].keys()),
                                '均值': list(results['summary_stats']['mean_values'].values()),
                                '标准差': list(results['summary_stats']['std_values'].values())
                            })
                            st.dataframe(stats_df, use_container_width=True)
                        
                        # 结果显示表格
                        st.subheader("📋 详细结果")
                        st.dataframe(results['results_dataframe'], use_container_width=True)
                        
                        # 可视化（示例）
                        st.subheader("📊 可视化")
                        tab1, tab2, tab3 = st.tabs(["分布图", "箱线图", "散点图"])
                        
                        with tab1:
                            if analysis_columns:
                                selected_col = st.selectbox("选择变量查看分布", analysis_columns)
                                if selected_col in df.columns:
                                    st.bar_chart(df[selected_col].value_counts().sort_index())
                        
                        # 下载结果
                        st.subheader("💾 导出结果")
                        
                        col_dl1, col_dl2 = st.columns(2)
                        
                        with col_dl1:
                            # 下载CSV
                            csv = results['results_dataframe'].to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="下载详细结果 (CSV)",
                                data=csv,
                                file_name=f"group_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col_dl2:
                            # 下载统计摘要
                            if results['summary_stats']['mean_values']:
                                stats_csv = stats_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="下载统计摘要 (CSV)",
                                    data=stats_csv,
                                    file_name=f"group_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
            
            except Exception as e:
                st.error(f"❌ 读取文件时出错: {str(e)}")
                st.info("请确保上传的是有效的Excel文件，并包含正确的数据格式。")
        
        else:
            # 上传文件前的说明
            st.info("👆 请上传Excel文件开始分析")
            
            # 模板下载
            st.markdown("---")
            st.subheader("📄 数据模板")
            
            col_temp1, col_temp2 = st.columns(2)
            
            with col_temp1:
                st.markdown("""
                ### 模板要求
                1. 第一行为列名
                2. 包含ID列用于标识
                3. 数值型数据放在对应列
                4. 支持.xlsx和.xls格式
                """)
            
            with col_temp2:
                # 提供模板文件下载
                template_data = pd.DataFrame({
                    'ID': ['001', '002', '003'],
                    'age': [45, 50, 35],
                    'weight': [65, 70, 60],
                    'height': [170, 175, 165],
                    'score_1': [5.2, 6.1, 4.8],
                    'score_2': [3.4, 4.2, 3.0]
                })
                
                template_csv = template_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="下载模板文件 (CSV)",
                    data=template_csv,
                    file_name="group_analysis_template.csv",
                    mime="text/csv",
                    help="下载模板文件了解数据格式要求"
                )

    # 页脚
    st.markdown("---")
    st.caption("© 2024 数据分析平台 | 版本 1.0")



def main():
 # 修改你的侧边栏代码
    with st.sidebar:
        st.title("🧭 导航")
        
        # 添加主页选项
        menu_options = ["🏠 平台主页", "🔍 Case 分析", "👥 Group 分析"]
        
        # 使用 selectbox 可以有 placeholder
        selected_page = st.selectbox(
            "选择页面",
            options=menu_options,
            index=0,  # 默认选择主页
            label_visibility="collapsed"
        )
        
        # 映射到分析类型
        if selected_page == "🏠 平台主页":
            # 显示主页内容
            st.session_state.show_home = True
            analysis_type = None
        elif selected_page == "🔍 Case 分析":
            analysis_type = "Case 分析"
            st.session_state.show_home = False
        else:  # "👥 Group 分析"
            analysis_type = "Group 分析"
            st.session_state.show_home = False
        
        st.markdown("---")
        
        # ... 其他侧边栏内容 ...

    # 主内容区域
    if st.session_state.get('show_home', True):
        # 显示主页
        st.title("🏠 欢迎使用预后分析平台")
        st.info("请从侧边栏选择分析类型开始使用")
        st.stop()
main()
if __name__=="__main":
    main()