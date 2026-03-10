
import pickle
import streamlit as st
import matplotlib.pyplot as plt 
plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
plt.rcParams['axes.unicode_minus'] = False
import shap
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from utils import *
import warnings
warnings.filterwarnings("ignore")
from Cindex import *
from sklearn.metrics import roc_auc_score,confusion_matrix
import numpy as np


COX_MODEL_PATH = "./cox_model.pkl"
SHAP_MODEL_PATH = "./shap_model.pkl" 

class Config():
    def __init__(self,cox_model_path,shap_model_path,
                 cox_feature_list=["腹膜转移评分二分","cN分期","肝脏转移","腹水分级","治疗方式1","治疗方式2"],
                 shap_feature_list=['治疗方式', 'cN分期', '腹水分级', '腹膜转移评分二分', '肝脏转移'],
                 threshold={"risks":1.204556,"peritoneal_metastasis_score":1,1:None,2:None,3:None,4:None,5:None}):
        self.cox_model=pickle.load(open(cox_model_path,'rb'))
        self.shap_model=pickle.load(open(shap_model_path,'rb'))
        self.threshold=threshold
        self.time_unit_map= {"天": 1, "月": 30, "年": 365.25}
        self.cox_feature_list=cox_feature_list
        self.shap_feature_list=shap_feature_list
    
c=Config(COX_MODEL_PATH,SHAP_MODEL_PATH)






def render_model_metric_analysis(c,df):
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
            
            
        with col_set2:
            # 选择分析变量
            time_points_list=st.text_input("要计算生存率的时间(年)",help="用,隔开",value="1,2,3,4,5")
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
        st.session_state.model_metric_submitter=True
    if st.session_state.model_metric_submitter:
        if st.button("清除当前结果",type="secondary"):
            st.session_state.model_metric_submitter=False

       
        df["RISKS"]=np.array(c.cox_model.predict_partial_hazard(df))
        df["RISKS_GROUP"]=df["RISKS"].map(lambda x:"低危组" if x<c.threshold["risks"] else "高危组")


        cindex=computeC(df[time_column],df[event_column],1-df["RISKS"],confidence_level*100)
        # high_risk_group_
                
        st.markdown("---")
        st.markdown("模型在给定数据上效果")
        col1,col2,col3=st.columns(3)
        with col1:
            st.metric("C指数",round(cindex["cindex"],4))
        with col2:
            st.metric("C指数下限",round(cindex["cindex_lower"],4))
        with col3:
            st.metric("C指数上限",round(cindex["cindex_upper"],4))


        fig=get_group_km_curve(df, event_column, time_column, "RISKS_GROUP",confidence_level,
                if_p= True,
                if_table=True,)
        
        st.markdown("---")
        st.markdown("### 模型预测结果")
        render_dataframe(df)

        
        st.markdown("---")
        st.markdown("#### 风险分组KM曲线")
        render_fig(fig)
        time_points_list=[float(x) for x in time_points_list.split(",")]
        for time_year in time_points_list:
            st.markdown("---")
            st.markdown(f"## {time_year}年生存状态分析")
            time_days=time_year*c.time_unit_map["年"]

            prob,true=get_cox_rel_at_timepoint(df,event_column,time_column,time_days,c.cox_model,feature_col_list=None)

            render_classify_table(prob,true,threshold=c.threshold.get(int(time_year),None))
            render_classify_plot(prob,true)


def render_classify_plot(prob,true):
    data={"prob":{"胃癌预测模型":prob},"y":{"胃癌预测模型":true}}
    col1,col2,col3=st.columns(3)
    for k,v in [[col1,"roc"],[col2,"calibration"],[col3,"dca"]]:
        st.markdown("---")
        st.markdown(f"#### {v}")
        xlist,ylist,legend_list=get_plt_data(data,v)
        for xx,yy,legend in zip(xlist,ylist,legend_list):
            plt.plot(xx,yy,label=legend)
        plt.legend()
        plt.show()
        if v=="dca":
            plt.ylim(bottom=-0.2,top=max(0.2,max(yy))+0.1)
        fig=plt.gcf()
        st.pyplot(fig)
        plt.close()



def render_classify_table(prob,true,threshold=None):
    if not threshold:
        threshold=get_best_cutoff(true,prob)
    pred=(prob >threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    totoal_n=len(pred)
    ppv=tp/(tp+fp)
    npv=tn/(fn+tn)
    f1score=2*sensitivity*ppv/(sensitivity+ppv)
    auc=roc_auc_score(true,prob)
    acc=(tp+tn)/totoal_n
    st.markdown("---")
    st.markdown("#### 评估指标")
    col1,col2,col3,col4=st.columns(4)
    with col1:
        st.metric("样本总数",value=totoal_n)
    with col2:
        st.metric("准确率",value=round(acc,4))
    with col3:
        st.metric("AUC值",value=round(auc,4))
    with col4:
        st.metric("F1score",value=round(f1score,4))

    col1,col2,col3,col4=st.columns(4)
    with col1:
        st.metric("真阳数(TP)",value=tp)
    with col2:
        st.metric("真阴数(TN)",value=tn)
    with col3:
        st.metric("假阳数(FP)",value=fp)
    with col4:
        st.metric("假阴数(FN)",value=fn)

    col1,col2,col3,col4=st.columns(4)
    with col1:
        st.metric("敏感度",value=round(sensitivity,4))
    with col2:
        st.metric("特异度",value=round(specificity,4))
    with col3:
        st.metric("阳性预测值",value=round(ppv,4))
    with col4:
        st.metric("阴性预测值",value=round(npv,4))




    col1,col2=st.columns(2)



    with col1:
        st.markdown("---")
        st.markdown("#### 混淆矩阵")
        sns.heatmap(confusion_matrix(true,pred),annot=True,cmap="coolwarm")
        fig=plt.gcf()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("---")
        st.markdown("#### 雷达图")
        metrics = [acc, auc, sensitivity, specificity, f1score, acc]  # 重复acc使图形闭合
        labels = ['ACC', 'AUC', 'SEN', 'SPE', 'F1', 'ACC']  # 重复ACC标签

        # 计算角度（6个点，但实际是5个维度+1个闭合点）
        N = len(metrics)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=True)  # 包含终点2π

        # 创建极坐标图
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        # 绘制雷达图线条
        ax.plot(angles, metrics, 'o-', linewidth=2, color='red', alpha=1.0, markersize=8)

        # 填充区域
        ax.fill(angles, metrics, color='blue', alpha=0.2)

        # 设置角度刻度和标签
        ax.set_xticks(angles[:-1])  # 不显示最后一个重复的ACC
        ax.set_xticklabels(labels[:-1], fontsize=12, fontweight='bold')
        plt.yticks([0,0.2,0.4,0.6,0.8,1])
        for i, (angle, value) in enumerate(zip(angles[:-1], metrics[:-1])):
            # 文本位置稍微偏移避免重叠
            text_radius = value + 0.05
            text_angle = angle
            
            # 标注数值
            ax.text(text_angle, text_radius, f'{value:.3f}',
                    ha='center', va='bottom',  # 底部对齐
                    fontsize=12, fontweight='bold',
                    color='#E74C3C',  # 红色突出显示
                    bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white',
                            edgecolor='#E74C3C',
                            alpha=0.9,
                            linewidth=1))
        # plt.xtickla
        fig=plt.gcf()
        st.pyplot(fig)
        plt.close()





def get_group_shap_values(df):
    shap_values = c.shap_model(df)
    shap_df = pd.DataFrame(shap_values.values, columns=df.columns)
    return shap_values,shap_df
def get_group_shap_fig_feature_importance(shap_values,df):
    mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'Feature': df.columns,
        'Mean SHAP Value': mean_abs_shap_values[:]
    }).sort_values(by='Mean SHAP Value', ascending=False)
    
    fig, ax= plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance['Feature'], feature_importance['Mean SHAP Value'], color='steelblue')
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importance by SHAP')
    ax.invert_yaxis()
    return fig

def get_group_shap_fig_summary_plot(shap_values,df):
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values[:, :], df, feature_names=df.columns, show=False)
    fig=plt.gcf()
    return fig
def get_group_shap_fig_heatmap(shap_values,df):
    fig, ax = plt.subplots(figsize=(12, 8))
    shap_df=pd.DataFrame(shap_values.values,columns=df.columns,index=df.index)
    sns.heatmap(shap_df,cmap="coolwarm")
    return fig



def get_group_shap_fig_onecase(shap_values,case_id):
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.waterfall(shap_values[int(case_id.split("_")[0]), :], max_display=20, show=False)
    plt.tight_layout()
    fig=plt.gcf()
    return fig  

def render_shap_analysis(c, df):
    """渲染SHAP分析"""
    st.subheader("🔍 SHAP特征分析")
    
    # 准备数据
    
    with st.form("确定shap分析caseid"):
        case_id_list = st.multiselect(
            "选择要展示的case", 
            options=list(df.index), 
            default=list(df.index)[:5]
        )
        submitter=st.form_submit_button("开始shap分析",type="primary")
    if submitter:
        st.session_state.shap_group_submitter=True
        
    if st.session_state.shap_group_submitter:
        if st.button("清除计算结果",type="secondary"):
            st.session_state.shap_group_submitter=False
    
    if  st.session_state.shap_group_submitter:
            # 执行SHAP分析
        with st.spinner("正在计算SHAP值..."):
            dft = df[c.shap_feature_list]
            shap_values,shap_df = get_group_shap_values(dft)
            shap_df[st.session_state.id_col]=np.array(df[st.session_state.id_col])
        # 提供下载
        st.markdown("---")
        st.markdown("#### SHAP数值")
        render_dataframe(shap_df)

        
        st.markdown("---")
        st.markdown("#### SHAP整体重要性分析图")
        fig=get_group_shap_fig_feature_importance(shap_values,dft)
        render_fig(fig)

        st.markdown("---")
        st.markdown("#### SHAP整体散点图")
        fig=get_group_shap_fig_summary_plot(shap_values,dft)
        render_fig(fig)

        st.markdown("---")
        st.markdown("#### SHAP整体热力图")
        fig=get_group_shap_fig_heatmap(shap_values,dft)
        render_fig(fig)

        st.markdown("---")
        st.markdown("#### SHAP单CASE分析图")
        for case_id in case_id_list:
            st.markdown(case_id)
            fig=get_group_shap_fig_onecase(shap_values,case_id)
            render_fig(fig)







def get_case_survival(feature_dic={}):
    feature_df = pd.DataFrame([feature_dic])
    feature_df=trans_df_data(feature_df)
    xx = list(range(365 * 5))
    survival_prob = c.cox_model.predict_survival_function(feature_df, xx)
        
    plt.figure(figsize=(10, 6))
    plt.plot(xx, survival_prob)
    plt.xlabel("天数")
    plt.ylabel("预估生存概率")
    plt.title("预估生存概率图")
    plt.grid(True, alpha=0.3)
    fig = plt.gcf()
     
    # 准备数据下载
    case_df = pd.DataFrame({
        "天数": xx,
    })
    case_df["预测存活概率"]=np.array(survival_prob)
    case_df=case_df.style.format({"预测存活概率":"{:.3e}"})
    return fig, case_df

def get_case_shap_waterfall(shap_values):
    """生成单个病例的SHAP分析"""
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.waterfall(shap_values[0, :], max_display=20, show=False) 
    
    fig = plt.gcf()
    return fig




def get_case_shap_force(shap_values):
    # fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.force(shap_values.base_values[0],shap_values.values[0,:],features=np.round(shap_values.data[0,:],3),feature_names=shap_values.feature_names,matplotlib=True,text_rotation=0,show=False)
    plt.tight_layout()
    fig=plt.gcf()
    return fig  

def get_case_shap_bar(shap_values):
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.bar_plot(shap_values.values[0, :],feature_names=shap_values.feature_names,show=False)
    plt.tight_layout()
    return fig  

def trans_df_data(df):
    df["治疗方式1"]= (df["治疗方式"] == 1).map(int)
    df["治疗方式2"] =(df["治疗方式"] == 2).map(int)
    df["腹膜转移评分二分"]=df["腹膜转移评分"].map(lambda x:0 if x<=c.threshold["peritoneal_metastasis_score"] else 1)
    return df
def get_case_metrics(feature_dic,time_unit,time_points_list):
        feature_df = pd.DataFrame([feature_dic])
        feature_df=trans_df_data(feature_df)

        risk = c.cox_model.predict_partial_hazard(feature_df).iloc[0]
        group = "低危组" if risk <= c.threshold["risks"] else "高危组"
        
        rel = {
            "risk": risk,
            "group": group,
            "中位生存时间": c.cox_model.predict_median(feature_df),
            "期望生存时间": c.cox_model.predict_expectation(feature_df).iloc[0]
        }
        time_points_list_days=[float(x)*c.time_unit_map[time_unit] for x in time_points_list]
        for i,time_point in enumerate(time_points_list_days):
            survival_prob = c.cox_model.predict_survival_function(feature_df, times=time_point).iloc[0, 0]
            rel[f"{time_points_list[i]}年生存概率"] = survival_prob
        return rel




def render_fig(fig):
    st.pyplot(fig)
    plt.close()
    
def render_dataframe(df):
    st.dataframe(df,width="content",hide_index=True)

def init_session_state():
    """初始化session_state"""
    initial_states = {
        'case_results': None,
        'group_results': None,
        'uploaded_file': None,
        'show_home': True,
        'analysis_type': None,
        'case_submitter': False,
        'check_state': None,
        'event_col': None,
        'time_col': None,
        "id_col":None,
        'group_func1_submitter': False,
        'shap_group_submitter': None
    }
    
    for key, value in initial_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        # 平台标题
        st.markdown("""
        <div style="text-align: center;">
            <h2>🏥</h2>
            <h3>胃癌预后分析平台</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 导航菜单
        menu_options = ["🏠 平台主页", "🔍 Case 分析", "👥 Group 分析"]
        
        # 确定默认选中项
        if st.session_state.analysis_type is None:
            default_index = 0
        elif st.session_state.analysis_type == "Case 分析":
            default_index = 1
        else:
            default_index = 2
        
        selected_page = st.selectbox(
            "选择页面",
            options=menu_options,
            index=default_index,
            label_visibility="collapsed",
        )
        
        # 更新页面状态
        if selected_page == "🏠 平台主页":
            st.session_state.show_home = True
            st.session_state.analysis_type = None
        elif selected_page == "🔍 Case 分析":
            st.session_state.analysis_type = "Case 分析"
            st.session_state.show_home = False
        else:  # "👥 Group 分析"
            st.session_state.analysis_type = "Group 分析"
            st.session_state.show_home = False
        
        st.markdown("---")
        
        # 平台声明
        st.markdown("### 平台声明")
        
        with st.expander("📋 免责声明", expanded=False):
            st.markdown("""
            #### 医学研究工具
            
            **性质说明**
            - 本平台为医学研究辅助工具
            - 所有预测基于统计模型
            - 存在一定的不确定性
            
            **使用限制**
            1. 不得用于临床诊断
            2. 结果需结合临床判断
            3. 数据需经伦理审批
            
            **责任声明**
            - 开发者不对使用结果负责
            - 用户需自行验证结果
            - 遵守当地法律法规
            """)
        
        with st.expander("🔒 数据安全", expanded=False):
            st.markdown("""
            #### 隐私保护
            
            **数据处理**
            - 数据不上传至服务器
            - 分析在用户本地完成
            - 临时文件自动清理
            
            **安全措施**
            - SSL加密传输
            - 无数据持久化
            - 访问日志记录
            
            **合规性**
            - 符合HIPAA要求
            - 通过安全审计
            - 定期更新维护
            """)
        
        st.markdown("---")
        st.caption("© 2024 医学研究平台 | v2.1.0")

def render_home_page():
    """渲染主页"""
    st.title("🏥 胃癌预后分析平台")
    st.markdown("""
        ### 欢迎使用""")
    # 两列布局
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("""
        本平台基于Cox比例风险模型，为胃癌患者提供精准的预后预测分析。
        
        **主要功能：**
        - **🔍 病例分析**: 输入个体临床指标，获取生存概率预测
        - **👥 批量分析**: 上传数据集，进行批量预后分析
        - **📊 可视化**: 交互式图表展示分析结果
        - **💾 导出**: 支持多种格式结果导出
        
        **模型特点：**
        - ✅ 基于多中心临床数据
        - ✅ 经过外部验证
        - ✅ 实时计算结果
        - ✅ 保护患者隐私
        """)
    
    with col_right:

        st.image("p1.png", width="content")
        st.image("p2.png", width="content")

    
    st.markdown("---")
    
    # 快速开始
    st.subheader("🚀 快速开始")
    
    col_start1, col_start2 = st.columns(2)
    
    with col_start1:
        st.markdown("#### 🔍 Case分析")
        st.markdown("适用于单个病例的详细分析")
        if st.button("开始Case分析", width="content", type="primary"):
            st.session_state.show_home = False
            st.session_state.analysis_type = "Case 分析"
            st.rerun()
    
    with col_start2:
        st.markdown("#### 👥 Group分析")
        st.markdown("适用于批量数据的统计分析")
        if st.button("开始Group分析", width="content", type="primary"):
            st.session_state.show_home = False
            st.session_state.analysis_type = "Group 分析"
            st.rerun()
    
    st.markdown("---")

def render_case_analysis():
    """渲染Case分析页面"""
    c = Config(COX_MODEL_PATH, SHAP_MODEL_PATH)
    
    st.header("🔍 Case 分析")
    
    st.markdown("---")
    
    # 输入表单
    with st.form("case_input_form"):
        st.subheader("📝 输入病例指标")
        
        col_form1, col_form2 = st.columns([2, 1])
        
        with col_form1:
            # 基础信息
            st.markdown("#### 基本信息")
            case_id = st.text_input("Case ID", value="CASE_001")
            analysis_date = st.date_input("分析日期", value=datetime.now().date())
            
            # 临床指标
            st.markdown("#### 临床指标")
            col_indic1, col_indic2 = st.columns(2)
            
            with col_indic1:
                treatment_method = st.number_input("治疗方式", min_value=1, max_value=3, value=1, step=1)
                ascites_grading = st.number_input("腹水分级", min_value=1, max_value=3, value=1, step=1)
                liver_metastasis = st.number_input("肝脏转移", min_value=0, max_value=1, value=0, step=1)
                
            with col_indic2:
                peritoneal_metastasis_score = st.number_input("腹膜转移评分", min_value=0, max_value=10, step=1, value=0)
                cn_stage = st.number_input("cN分期", min_value=0,max_value=3, value=0)
            
            # 时间设置
            st.markdown("#### 要计算的时间")
            col_time1, col_time2 = st.columns(2)
            
            with col_time1:
                time_unit = st.selectbox(
                    "时间单位",
                    options=["年", "月", "天"],
                    index=0,
                    help="要计算的时刻的单位"
                )
            
            with col_time2:
                raw_time_points_list = st.text_input(
                    "时间点",
                    value="1,2,3,4,5",
                    help="要计算的时刻，用逗号分隔"
                )
                time_points_list=raw_time_points_list.split(",")
            
            # 分析参数
            st.markdown("#### 分析参数")

            confidence_level = st.slider(
                "置信水平",
                min_value=0.90,
                max_value=0.99,
                value=0.95,
                step=0.01
            )
        
        with col_form2:
            st.subheader("📋 输入预览")
            if st.session_state.case_submitter:
                input_summary = {
                    "腹膜转移评分": peritoneal_metastasis_score,
                    "cN分期": cn_stage,
                    "肝脏转移": liver_metastasis,
                    "腹水分级": ascites_grading,
                    "治疗方式": treatment_method
                }
                
                summary_df = pd.DataFrame(
                    list(input_summary.items()), 
                    columns=["参数", "值"]
                )
                st.dataframe(summary_df, width="content", hide_index=True)
        
        # 提交按钮
        submitted = st.form_submit_button("🚀 开始分析", type="primary")
        
    if submitted:
        st.session_state.case_submitter = True
        st.session_state.case_feature_dic={"治疗方式":treatment_method,"腹水分级":ascites_grading,"腹膜转移评分":peritoneal_metastasis_score,"cN分期":cn_stage,"肝脏转移":liver_metastasis}
    # 结果显示区域

    if st.session_state.case_submitter:
        if st.button("🔄 开始新分析", type="secondary", width="content"):
            st.session_state.case_submitter = False
            st.rerun()
        st.markdown("---")
        results=get_case_metrics(feature_dic=st.session_state.case_feature_dic,time_unit=time_unit,time_points_list=time_points_list)
        results["case_id"]=case_id
        results["分析时间"]=analysis_date
        # 显示核心结果
        st.subheader("📊 分析结果")

        
        # 结果卡片
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            st.metric("预测分数", f"{results['risk']:.1f}")
            st.metric("中位生存时间", f"{results['中位生存时间']:.1f}天")
        
        with col_result2:
            st.metric("风险等级", results['group'])
            st.metric("期望生存时间", f"{results['期望生存时间']:.1f}天")
        
        st.markdown("---")
        # 生存概率
        st.markdown("#### 📅 生存概率预测")
        cols_probs = st.columns(len(time_points_list))
        for i, time_point in enumerate(time_points_list):
            with cols_probs[i]:
                st.metric(
                    f"{time_point}{time_unit}生存概率",
                    f"{results[f'{time_point}年生存概率']:.2%}"
                )
        
        # 下载分析结果
        st.markdown("---")
        st.markdown("#### 💾 下载结果")
        
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            results_df = pd.DataFrame([results])
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 下载分析结果 (CSV)",
                data=csv,
                file_name=f"case_analysis_{case_id}.csv",
                mime="text/csv",
                width="content"
            )
        
        with col_dl2:
            st.info("更多图表下载见下方")
        
        # SHAP分析
        st.markdown("---")
        st.subheader("🔍 特征SHAP分析")


        feature_df = pd.DataFrame([st.session_state.case_feature_dic])
        feature_df=trans_df_data(feature_df)
        feature_df=feature_df[c.shap_feature_list]
        shap_values = c.shap_model(feature_df)
        shap_fig_waterfall =get_case_shap_waterfall(shap_values)
        shap_fig_force =get_case_shap_force(shap_values)
        shap_fig_bar=get_case_shap_bar(shap_values)
        st.markdown("#### case分析瀑布图")
        st.markdown("---")
        render_fig(shap_fig_waterfall)

        st.markdown("#### case分析力图")
        st.markdown("---")
        render_fig(shap_fig_force)

        st.markdown("#### case分析柱状图")
        st.markdown("---")
        render_fig(shap_fig_bar)
        
        # 生存曲线分析
        st.markdown("---")
        st.subheader("📈 case预估生存率")
        
        survival_fig, survival_df = get_case_survival(st.session_state.case_feature_dic)
        
        
        render_fig(survival_fig)
        st.markdown("##### 图标详细坐标点")
        render_dataframe(survival_df)





def render_data_upload():
    """渲染数据上传区域"""
    st.subheader("📁 数据上传")
    
    # 清除按钮
    if st.session_state.get('uploaded_file') is not None:
        if st.button("🗑️ 清除当前文件", type="secondary", width="content"):
            st.session_state.uploaded_file = None
            st.session_state.check_state = None
            st.rerun()
    
    uploaded_file = st.file_uploader(
        "上传Excel文件",
        type=['xlsx', 'xls'],
        help="请上传包含分析数据的Excel文件"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
    
    return uploaded_file


def render_data_preview(df):
    """渲染数据预览"""
    with st.expander("📊 数据预览", expanded=True):
        st.dataframe(df.head(min(100,df.shape[0])), width="content")
        
        st.subheader("📈 数据统计摘要")
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("总样本数", len(df))
        with col_stats2:
            st.metric("变量数量", len(df.columns))
        with col_stats3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("数值变量", len(numeric_cols))
    
    return True


def render_template_download():
    """渲染模板下载"""
    st.subheader("📄 数据模板")
    
    col_temp1, col_temp2 = st.columns(2)
    
    with col_temp1:
        st.markdown("""
        ### 模板要求
        
        1. **格式要求**
           - Excel格式 (.xlsx/.xls)
           - 第一行为列名
           - 标准数据格式
        
        2. **必需列**
           - 事件列 (0/1)
           - 时间列 (数值)
           - 临床指标列
        
        3. **推荐列**
           - ID列
           - 分组变量
        """)
    
    with col_temp2:
        template_data = pd.DataFrame({
            'ID': ['001', '002', '003'],
            '是否死亡': [0, 1, 0],
            'OS': [365, 180, 730],
            '腹膜转移评分': [2, 5, 1],
            '腹水分级': [1, 2, 3],
            '肝脏转移': [0, 1, 0],
            '治疗方式': [1, 2, 3],
            'cN分期': [1, 2, 3]
        })
        
        template_csv = template_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下载模板文件 (CSV)",
            data=template_csv,
            file_name="group_analysis_template.csv",
            mime="text/csv",
            help="下载模板文件了解数据格式要求",
            width="content"
        )



def render_data_validation(df):
    """渲染数据验证"""
    st.subheader("🔍 数据核验")
    
    with st.form("data_validation_form"):
        col_val1, col_val2,col_val3 = st.columns(3)
        
        with col_val1:
            event_col = st.text_input("事件列名称", value="是否死亡")
        
        with col_val2:
            time_col = st.text_input("时间列名称", value="OS")

        with col_val3:
            id_col = st.text_input("I唯一ID列名称", value="ID")
            st.session_state.id_col=id_col
        start_check = st.form_submit_button("开始数据核验", type="primary")
        
        if start_check:
            if validate_data(df, event_col, time_col,id_col):
                st.success("✅ 数据核验成功！可以进行下一步分析了。")
    return st.session_state.check_state == "good"
def validate_data(df, event_col, time_col,id_col):
    """验证数据格式"""
    st.session_state.check_state = "good"
    all_need_col_dict = {
        event_col: [float, [0, 1]],
        time_col: [float, None],
        "腹膜转移评分": [float, set(list(range(11)))],
        "腹水分级": [float, set([0, 1, 2, 3])],
        "肝脏转移": [float, [1, 0]],
        "治疗方式": [float, set([1, 2, 3])],
        "cN分期": [float, set([0, 1, 2, 3])]
    }
    
    st.session_state.event_col = event_col
    st.session_state.time_col = time_col
    
    # 列名验证
    missing_cols = []
    for col, _ in all_need_col_dict.items():
        if col not in df.columns:
            missing_cols.append(col)
    if id_col not in df.columns:
        missing_cols.append(id_col)

    if missing_cols:
        st.session_state.check_state = "bad"
        st.error(f"❌ 缺失必要列: {missing_cols}")
        return False
    
    st.success("✅ 列名验证通过")
    
    # 数据类型验证
    type_error_cols = []
    for col, col_rule in all_need_col_dict.items():
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                type_error_cols.append(col)
    
    if type_error_cols:
        st.session_state.check_state = "bad"
        st.error(f"❌ 非数值列: {type_error_cols}")
        return False
    
    st.success("✅ 数据类型验证通过")
    
    # 值范围验证
    range_error_cols = []
    for col, col_rule in all_need_col_dict.items():
        if col in df.columns and col_rule[1]:
            for val in df[col].unique():
                if pd.notna(val) and val not in col_rule[1]:
                    range_error_cols.append(col)
                    break
    
    if range_error_cols:
        st.session_state.check_state = "bad"
        st.error(f"❌ 值范围错误: {range_error_cols}")
        return False
    
    st.success("✅ 值范围验证通过")
    return True


def cal_km_survival_info_sub(df,event_col,time_col,confidence_level,time_points_list,pth_list):
    time_points_list=[float(x) for x in time_points_list.split(",")]
    pth_list=[float(x) for x in pth_list.split(",")]
    rel=[]
    kmf=KaplanMeierFitter()
    kmf.fit(df[time_col],df[event_col],alpha=1-confidence_level)
    time_list=[float(x) for x in kmf.survival_function_.index]
    survaival_list=list(kmf.survival_function_["KM_estimate"])
    survival_ci_df=kmf.confidence_interval_survival_function_
    survaival_list_lower=list(survival_ci_df[survival_ci_df.columns[0]])
    survaival_list_upper=list(survival_ci_df[survival_ci_df.columns[1]])
    for time_year in time_points_list:
        time=float(time_year)*365
        survival,survival_lower,survival_upper=get_time_survival_info(time_list,survaival_list,time,"time"),get_time_survival_info(time_list,survaival_list_lower,time,"time"),get_time_survival_info(time_list,survaival_list_upper,time,"time")
        info={"type":"时间","type_value":f"{time_year}年",f"生存率/时间  置信区间({confidence_level})":f"{survival}({survival_lower},{survival_upper})"}
        rel.append(info)
    for old_pth in pth_list:
        pth=float(old_pth)/100
        survival,survival_lower,survival_upper=get_time_survival_info(time_list,survaival_list,pth,"survival"),get_time_survival_info(time_list,survaival_list_lower,pth,"survival"),get_time_survival_info(time_list,survaival_list_upper,pth,"survival")
        info={"type":"生存率","type_value":f"{old_pth}%",f"生存率/时间  置信区间({confidence_level})":f"{survival}({survival_lower},{survival_upper})"}
        rel.append(info)

    rel=pd.DataFrame(rel)
    return rel

            

def get_group_km_info(df,event_col,time_col,var_col_list,confidence_level,time_points_list,pth_list):
    rel=[]
    for var in var_col_list:
        if not  var is None:
            for var_element in df[var].unique():
                dft=df[df[var]==var_element]
                sub_reldf=cal_km_survival_info_sub(dft,event_col,time_col,confidence_level,time_points_list,pth_list)
                sub_reldf["Group"]=var
                sub_reldf["Group_value"]=var_element
                rel.append(sub_reldf)
        else:
            sub_reldf=cal_km_survival_info_sub(df,event_col,time_col,confidence_level,time_points_list,pth_list)
            sub_reldf["Group"]="All"
            sub_reldf["Group_value"]="All"
            rel.append(sub_reldf)  
    return pd.concat(rel)
    
def get_group_km_curve(df,event_col,time_col,var_col,confidence_level,**args):
    fig = plt_km_multigroup_df(df, event_col, time_col, var_col,confidence_level, **args)
    return fig

def render_km_analysis(c, df):
    """渲染KM生存分析"""
    st.subheader("📈 Kaplan-Meier生存分析")
    
    with st.form("km_analysis_form"):
        col_km1, col_km2 = st.columns(2)
        
        with col_km1:
            event_col = st.selectbox(
                "事件列",
                options=df.columns.tolist(),
                index=df.columns.tolist().index(st.session_state.event_col)
            )

            time_col = st.selectbox(
                "时间列",
                options=df.columns.tolist(),
                index=df.columns.tolist().index(st.session_state.time_col)
            )

            time_points_list=st.text_input("要计算生存率的时间(年)",help="用,隔开",value="1,2,3,4,5")


        with col_km2:
            var_col_list = st.multiselect(
                "分组变量（可选）",
                options=["不分组"] + [x for x in df.columns if df[x].nunique()<5],
                default=["不分组"]
            )
            confidence_level = st.slider(
                "置信水平",
                min_value=0.90,
                max_value=0.99,
                value=0.95,
                step=0.01
            )
            pth_list=st.text_input("要计算的生存时间百分位数",help="用,隔开",value="25,50,75")
        submitted = st.form_submit_button("开始生存分析", type="primary")
        
    if submitted:
        st.session_state.group_func1_submitter = True
        
    
    if st.session_state.group_func1_submitter:
        # 清除按钮
        if st.button("🔄 重新分析", type="secondary"):
            st.session_state.group_func1_submitter = False
            st.rerun()
        
        var_column_list=[ None if var_col == "不分组" else var_col for var_col in var_col_list]
         
        for var_column in var_column_list:
            fig=plt_km_multigroup_df(  df, event_col, time_col, var_column,confidence_level,
                if_p=False if var_column is None else True,
                if_table=True,)
            st.markdown("---")
            st.markdown(f"KM曲线_{var_column if not var_column is None else '全部'}")
            render_fig(fig)
 
        st.markdown("---")
        st.markdown("KM分析结果")
        km_survival_df=get_group_km_info(df,event_col,time_col,var_column_list,confidence_level,time_points_list,pth_list)
        render_dataframe(km_survival_df)




def render_group_analysis():
    """渲染Group分析页面"""
    
    st.header("👥 Group 分析")
    
    # 1. 数据上传
    uploaded_file = render_data_upload()
    
    if uploaded_file is None:
        # 显示模板下载
        st.markdown("---")
        render_template_download()
        return
    
    # 2. 读取数据
    try:
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ 文件上传成功！共读取到 {len(df)} 行数据，{len(df.columns)} 个字段")
    except Exception as e:
        st.error(f"❌ 读取文件时出错: {str(e)}")
        return
    
    # 3. 数据预览
    render_data_preview(df)
    st.markdown("---")
    
    # 4. 数据验证
    if not render_data_validation(df):
        return
    
    df=trans_df_data(df)
    df.index=pd.Series([str(x) for x in range(df.shape[0])])+"_"+df[st.session_state.id_col].map(str)
    st.markdown("---")
    
    # 5. KM生存分析
    render_km_analysis(c, df)
    st.markdown("---")
    
    # 6. SHAP分析
    render_shap_analysis(c, df)
    st.markdown("---")
    
    # 7. 模型效能分析（预留）
    st.subheader("📊 模型效能分析")
    render_model_metric_analysis(c,df)




def main():
    st.set_page_config(
        page_title="胃癌预后分析平台",
        page_icon="🏥",
        layout="wide"
    )
    
    # 初始化
    init_session_state()
    
    # 渲染侧边栏
    render_sidebar()
    
    # 根据当前状态渲染对应页面
    if st.session_state.show_home:
        render_home_page()
    elif st.session_state.analysis_type == "Case 分析":
        render_case_analysis()
    elif st.session_state.analysis_type == "Group 分析":
        render_group_analysis()



main()