import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import shap
import optuna
import pickle
import io
import time

# ── Page config ──
st.set_page_config(page_title="XGBoost 电子温度预测", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stMetric { text-align: center; }
    .stMetric label { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──
for key in ['model', 'y_test', 'y_pred', 'y_all', 'y_pred_all',
            'X_train', 'X_test', 'y_train', 'X_all',
            'feature_names', 'shap_values', 'shap_interaction_values',
            'best_params', 'explainer', 'X_shap', 'trained']:
    if key not in st.session_state:
        st.session_state[key] = None

if 'trained' not in st.session_state:
    st.session_state.trained = False


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ═══════════════════════════════════════════════════════
# Sidebar — 配置区
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ 配置面板")

    # ── 1. 数据导入 ──
    st.header("1. 数据导入")
    uploaded_file = st.file_uploader("上传 CSV 文件", type=['csv'],
                                     help="CSV 第一行为列名")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            data.pop("Shot")
            data.pop("Wmhd")
            # df.pop("BetaN")
            # df.pop('Vloop')
            data.pop('PLHCD')
            data['PLHCD'] = data["P2.45LHW"] + data["P4.6LHW"]
            data.pop("P2.45LHW")
            data.pop("P4.6LHW")
            data.pop("POHM")
            # df.pop("PICRF")
            # df.pop('PLHCD')
            # df.pop("PECRH")
            # df.pop("PNBI")
            data.pop('Ptotal')
            # df.pop('Q95')
            # df.pop('Prad')
            # df.pop("Li")
            
            te_values = data.pop('Te')
            used_features_names = [r"$\beta_n$", r"$n_e$", r"$V_{loop}$", r"$I_p$", r"$P_{ICRF}$", r"$P_{ECRH}$", r"$B_t$", r"$\delta_{oben}$", r"$\delta_{untn}$", r"$\kappa$", r"$a$", r"$q_{95}$", r"$li$", r"$P_{rad}$", r"$P_{NBI}$", r"$P_{LHCD}$"]
            data.columns = used_features_names
            data['Te'] = te_values
            st.success(f"已加载 {data.shape[0]} 行 × {data.shape[1]} 列")
        except Exception as e:
            st.error(f"读取失败: {e}")
            st.stop()

        # ── 2. 列选择 ──
        st.header("2. 特征与目标选择")
        all_cols = data.columns.tolist()
        target_col = st.selectbox("目标列 (y)", all_cols,
                                  help="选择预测目标变量")
        available = [c for c in all_cols if c != target_col]
        feature_cols = st.multiselect(
            "特征列 (X)", available,
            default=available,
            help="可自由增删特征列")

        if not feature_cols:
            st.warning("请至少选择一个特征列")
            st.stop()

        # ── 3. 数据集划分 ──
        st.header("3. 数据集划分")
        c1, c2 = st.columns(2)
        with c1:
            test_size = st.slider("测试集比例", 0.10, 0.50, 0.30, 0.05)
        with c2:
            random_state = st.number_input("随机种子", 0, 9999, 42)

        # ── 4. 调参模式 ──
        st.header("4. 调参模式")
        tune_mode = st.radio("选择模式", ["手动调参", "Optuna 自动调参"],
                             horizontal=True)

        params = {}
        early_stopping_rounds = 5

        if tune_mode == "手动调参":
            st.subheader("XGBoost 超参数")

            with st.expander("树结构参数", expanded=True):
                params['n_estimators'] = st.slider("n_estimators (树数量)", 50, 500, 196, 10)
                params['max_depth'] = st.slider("max_depth (最大深度)", 3, 20, 15, 1)
                params['learning_rate'] = st.slider("learning_rate (学习率)", 0.01, 0.30, 0.034, 0.005)

            with st.expander("采样参数", expanded=False):
                params['subsample'] = st.slider("subsample (样本采样比)", 0.10, 1.00, 0.86, 0.05)
                params['colsample_bytree'] = st.slider("colsample_bytree (特征采样比/树)", 0.50, 1.00, 0.80, 0.05)
                params['colsample_bylevel'] = st.slider("colsample_bylevel (特征采样比/层)", 0.50, 1.00, 0.73, 0.05)
                params['colsample_bynode'] = st.slider("colsample_bynode (特征采样比/节点)", 0.50, 1.00, 0.72, 0.05)

            with st.expander("正则化参数", expanded=False):
                params['gamma'] = st.slider("gamma (最小损失减少)", 0.00, 2.00, 0.37, 0.05)
                params['reg_alpha'] = st.slider("reg_alpha (L1 正则化)", 0.00, 5.00, 0.31, 0.05)
                params['reg_lambda'] = st.slider("reg_lambda (L2 正则化)", 0.00, 5.00, 0.47, 0.05)

            early_stopping_rounds = st.slider("early_stopping_rounds", 1, 50, 5)

        else:  # Optuna自动调参
            sc1, sc2 = st.columns(2)
            with sc1:
                n_trials = st.number_input("搜索轮数", 10, 10000, 1000, 50)
            with sc2:
                cv_folds = st.selectbox("交叉验证折数", [3, 5, 10], index=1)

            with st.expander("搜索范围设置", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    lr_min = st.number_input("learning_rate 下限", 0.001, 0.1, 0.01, 0.005)
                with c2:
                    lr_max = st.number_input("learning_rate 上限", 0.01, 0.5, 0.1, 0.01)

                c1, c2 = st.columns(2)
                with c1:
                    depth_min = st.number_input("max_depth 下限", 3, 10, 5, 1)
                with c2:
                    depth_max = st.number_input("max_depth 上限", 5, 30, 20, 1)

                c1, c2 = st.columns(2)
                with c1:
                    n_min = st.number_input("n_estimators 下限", 50, 300, 80, 10)
                with c2:
                    n_max = st.number_input("n_estimators 上限", 100, 1000, 300, 10)

                c1, c2 = st.columns(2)
                with c1:
                    subsample_min = st.number_input("subsample 下限", 0.1, 1.0, 0.1, 0.05)

                c1, c2 = st.columns(2)
                with c1:
                    gamma_min = st.number_input("gamma 下限", 0.0, 5.0, 0.1, 0.05)
                with c2:
                    gamma_max = st.number_input("gamma 上限", 0.1, 10.0, 1.0, 0.1)

                c1, c2 = st.columns(2)
                with c1:
                    alpha_min = st.number_input("reg_alpha 下限", 0.0, 5.0, 0.1, 0.05)
                with c2:
                    alpha_max = st.number_input("reg_alpha 上限", 0.1, 10.0, 2.0, 0.1)

                c1, c2 = st.columns(2)
                with c1:
                    lambda_min = st.number_input("reg_lambda 下限", 0.0, 5.0, 0.1, 0.05)
                with c2:
                    lambda_max = st.number_input("reg_lambda 上限", 0.1, 10.0, 2.0, 0.1)

        st.markdown("---")
        train_clicked = st.button("🚀 开始训练", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════
st.title("XGBoost 电子温度预测与 SHAP 分析工具")

if uploaded_file is None:
    st.info("👈 请在左侧上传 CSV 数据文件开始")
    st.stop()

# ── 数据预览 ──
with st.expander("📊 数据预览", expanded=True):
    t1, t2 = st.tabs(["原始数据", "统计描述"])
    with t1:
        st.dataframe(data.head(100), use_container_width=True, height=300)
    with t2:
        st.dataframe(data.describe(), use_container_width=True)


# ═══════════════════════════════════════════════════════
# 训练逻辑
# ═══════════════════════════════════════════════════════
def run_training():
    X_all = data[feature_cols].copy()
    y_all = data[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, shuffle=True, random_state=random_state
    )

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.X_all = X_all
    st.session_state.y_all = y_all
    st.session_state.feature_names = feature_cols

    if tune_mode == "Optuna 自动调参":
        with st.status("🔍 Optuna 正在搜索最优参数...", expanded=True) as status:
            def objective(trial):
                optuna_params = {
                    'n_estimators': trial.suggest_int('n_estimators', n_min, n_max),
                    'max_depth': trial.suggest_int('max_depth', depth_min, depth_max),
                    'learning_rate': trial.suggest_float('learning_rate', lr_min, lr_max, log=True),
                    'subsample': trial.suggest_float('subsample', subsample_min, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                    'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', gamma_min, gamma_max),
                    'reg_alpha': trial.suggest_float('reg_alpha', alpha_min, alpha_max),
                    'reg_lambda': trial.suggest_float('reg_lambda', lambda_min, lambda_max),
                }
                model = xgb.XGBRegressor(**optuna_params, random_state=42)
                kf = KFold(cv_folds, shuffle=True, random_state=20)
                rmse_scores = np.sqrt(-cross_val_score(
                    model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf
                ))
                return rmse_scores.mean()

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction='minimize')
            progress_bar = st.progress(0)
            status_text = st.empty()

            class StreamlitCallback:
                def __call__(self, study, trial):
                    progress_bar.progress(trial.number / n_trials)
                    if trial.number % 50 == 0:
                        status_text.text(
                            f"第 {trial.number}/{n_trials} 轮, "
                            f"当前最优 RMSE: {study.best_value:.4f}"
                        )

            study.optimize(objective, n_trials=n_trials, callbacks=[StreamlitCallback()],
                           show_progress_bar=False)
            progress_bar.empty()
            status_text.empty()

            best_params = study.best_params
            best_params['random_state'] = 42
            st.session_state.best_params = best_params
            st.success(f"Optuna 搜索完成！最优 CV-RMSE: {study.best_value:.4f}")
            status.update(label="✅ Optuna 搜索完成", state="complete")

    else:
        best_params = params.copy()
        best_params['random_state'] = 42
        st.session_state.best_params = best_params

    # ── 训练最终模型（XGBoost 3.x API：eval_metric/early_stopping_rounds 在构造器，eval_set 在 fit）──
    with st.status("🏋️ 训练最终模型...", expanded=True) as status:
        model = xgb.XGBRegressor(
            **best_params,
            eval_metric='rmse',
            early_stopping_rounds=early_stopping_rounds,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )

        y_pred = model.predict(X_test)
        y_pred_all = model.predict(X_all)

        st.session_state.model = model
        st.session_state.y_pred = y_pred
        st.session_state.y_pred_all = y_pred_all
        st.session_state.trained = True
        status.update(label="✅ 训练完成", state="complete")


# ═══════════════════════════════════════════════════════
# 结果分析
# ═══════════════════════════════════════════════════════
def show_results():
    model = st.session_state.model
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred
    y_all = st.session_state.y_all
    y_pred_all = st.session_state.y_pred_all
    X_all = st.session_state.X_all
    X_test = st.session_state.X_test
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train

    # ── Metrics ──
    R_train = model.score(X_train, y_train)
    R_test = model.score(X_test, y_test)
    R_all = model.score(X_all, y_all)
    mae_test = mean_absolute_error(y_test, y_pred)
    rmse_test = rmse(y_test, y_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_pred)
    mae_all = mean_absolute_error(y_all, y_pred_all)
    rmse_all = rmse(y_all, y_pred_all)
    mape_all = mean_absolute_percentage_error(y_all, y_pred_all)

    # 误差统计
    err_test = y_pred - y_test.values
    err_all = y_pred_all - y_all.values
    err_mean_test = np.median(np.abs(err_test))
    err_min_test = err_test.min()
    err_max_test = err_test.max()
    err_mean_all = np.median(np.abs(err_all))
    err_min_all = err_all.min()
    err_max_all = err_all.max()

    st.header("📈 模型性能")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("训练集 R²", f"{R_train:.4f}")
    with c2:
        st.metric("测试集 R²", f"{R_test:.4f}")
    with c3:
        st.metric("ALL R²", f"{R_all:.4f}")
    with c4:
        st.metric("测试集 MAE", f"{mae_test:.4f}")
    with c5:
        st.metric("测试集 RMSE", f"{rmse_test:.4f}")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("测试集 MAPE", f"{mape_test:.4f}")
    with c2:
        st.metric("ALL MAE", f"{mae_all:.4f}")
    with c3:
        st.metric("ALL RMSE", f"{rmse_all:.4f}")
    with c4:
        st.metric("ALL MAPE", f"{mape_all:.4f}")

    bp = st.session_state.best_params
    st.metric("最优参数",
              f"n_estimators={bp.get('n_estimators','?')}  max_depth={bp.get('max_depth','?')}  learning_rate={bp.get('learning_rate','?'):.4f}  subsample={bp.get('subsample','?'):.2f}")

    # ── 误差统计 ──
    with st.expander("📏 预测误差统计", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**测试集**")
            st.write(f"误差绝对值中位数: {err_mean_test:.4f}")
            st.write(f"误差最小值: {err_min_test:.4f}")
            st.write(f"误差最大值: {err_max_test:.4f}")
        with c2:
            st.markdown("**全部数据**")
            st.write(f"误差绝对值中位数: {err_mean_all:.4f}")
            st.write(f"误差最小值: {err_min_all:.4f}")
            st.write(f"误差最大值: {err_max_all:.4f}")

    # ── 目标变量分布 ──
    st.header("📉 目标变量分布")
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.dpi = 150
    sns.histplot(y_train, label="Train", bins=40, stat='density', alpha=0.5, ax=ax)
    sns.histplot(y_test, label="Test", bins=40, stat='density', alpha=0.5, ax=ax)
    ax.set_xlabel(target_col, size=14)
    ax.set_ylabel("Density", size=14)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # ── 2D 彩色散点图 ──
    st.header("🎨 2D 彩色散点图")
    st.caption("选择两个特征作为 X/Y 轴，用第三维（目标值或预测值）着色")

    pred_df = pd.concat([X_all, pd.DataFrame(y_pred_all, columns=['Te_pred'], index=X_all.index)], axis=1)
    if target_col not in pred_df.columns:
        pred_df[target_col] = y_all.values

    cc1, cc2, cc3 = st.columns(3)
    all_plot_cols = X_all.columns.tolist() + ['Te_pred', target_col]
    with cc1:
        x_col = st.selectbox("X 轴特征", X_all.columns.tolist(), key='x_col')
    with cc2:
        y_col = st.selectbox("Y 轴特征", [c for c in X_all.columns.tolist() if c != x_col], key='y_col')
    with cc3:
        color_col = st.selectbox("颜色映射", all_plot_cols + ['Te_pred'],
                                 index=len(all_plot_cols) - 1 if 'Te_pred' in all_plot_cols else 0, key='color_col')

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.dpi = 150
    xv = pred_df[x_col].values
    yv = pred_df[y_col].values
    cv = pred_df[color_col].values

    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=cv.min(), vmax=cv.max())
    sc = ax.scatter(xv, yv, s=30, c=cv, cmap=cmap, norm=norm, marker='o', alpha=0.7)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(color_col, size=12)
    ax.set_xlabel(x_col, size=14)
    ax.set_ylabel(y_col, size=14)
    st.pyplot(fig)
    plt.close(fig)

    # ── 预测 vs 真实 ──
    st.header("📊 预测 vs 真实")
    tab_test, tab_all = st.tabs(["测试集", "全部数据"])

    for tab, xv, pv, rv in [
        (tab_test, y_test.values, y_pred, R_test),
        (tab_all, y_all.values, y_pred_all, R_all)
    ]:
        with tab:
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.dpi = 150
            xy = np.vstack([xv, pv])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            xs, ps, zs = xv[idx], pv[idx], z[idx]
            sc = ax.scatter(xs, ps, c=zs, marker='o', s=8, cmap=plt.cm.jet)
            lim_min = min(xv.min(), pv.min()) * 0.9
            lim_max = max(xv.max(), pv.max()) * 1.1
            ax.plot([lim_min, lim_max], [lim_min, lim_max], c='skyblue', ls='-', lw=1.5)
            ax.set_xlabel(f"Target {target_col}", size=14)
            ax.set_ylabel(f"Predict {target_col}", size=14)
            ax.text(0.05, 0.92, f"$R^2$ = {rv:.3f}", transform=ax.transAxes, fontsize=12)
            plt.colorbar(sc, ax=ax)
            st.pyplot(fig)
            plt.close(fig)

    # ── SHAP 分析 ──
    st.header("🧠 SHAP 模型解释")

    with st.spinner("正在计算 SHAP 值 (可能需要几分钟)..."):
        explainer = shap.TreeExplainer(model)
        shap_sample_size = min(300, X_all.shape[0])
        X_shap = X_all.sample(shap_sample_size, random_state=42)
        shap_values = explainer(X_shap)
        st.session_state.shap_values = shap_values
        st.session_state.X_shap = X_shap
        st.session_state.explainer = explainer

    st.success(f"SHAP 计算完成 (使用 {shap_sample_size} 个样本)")

    # SHAP tabs: 全局解释 vs 局部解释
    shap_tab1, shap_tab2 = st.tabs(["全局解释", "局部解释"])

    with shap_tab1:
        # Summary plot
        st.subheader("SHAP 特征摘要 (Summary Plot)")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.dpi = 150
        shap.summary_plot(shap_values, X_shap, show=False)
        st.pyplot(fig)
        plt.close(fig)

        # Bar plot
        st.subheader("SHAP 特征重要性 (Bar)")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.dpi = 150
        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
        st.pyplot(fig)
        plt.close(fig)

        # Dependence plot
        st.subheader("SHAP 特征依赖 (Dependence Plot)")
        dep_feat = st.selectbox("选择特征", X_all.columns.tolist(), key='dep_feat')
        inter_feat = st.selectbox("交互特征 (可选)", ["无"] + X_all.columns.tolist(), key='inter_feat')
        inter = None if inter_feat == "无" else inter_feat

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.dpi = 150
        shap.dependence_plot(dep_feat, shap_values.values, X_shap,
                             interaction_index=inter, ax=ax, show=False)
        st.pyplot(fig)
        plt.close(fig)

        # Interaction values
        st.subheader("SHAP 交互值 (Interaction Values)")
        st.caption("计算量较大，首次计算可能需要等待...")
        if st.button("计算/刷新 SHAP 交互值") or st.session_state.shap_interaction_values is not None:
            if st.session_state.shap_interaction_values is None:
                with st.spinner("正在计算 SHAP 交互值..."):
                    shap_inter = explainer.shap_interaction_values(X_shap)
                    st.session_state.shap_interaction_values = shap_inter
            else:
                shap_inter = st.session_state.shap_interaction_values

            plt.figure(figsize=(16, 4), dpi=120)
            shap.summary_plot(shap_inter, X_shap, plot_type="violin", show=False)
            st.pyplot(plt.gcf())
            plt.close('all')

        # Heatmap
        st.subheader("SHAP 热力图 (Heatmap)")
        shap.plots.heatmap(shap_values, show=False)
        st.pyplot(plt.gcf())
        plt.close('all')

    with shap_tab2:
        sample_idx = st.slider("选择样本索引", 0, shap_sample_size - 1, 0, 1)

        # Force plot
        st.subheader("SHAP 力图 (Force Plot)")
        fig = shap.plots.force(shap_values[sample_idx], matplotlib=True, show=False)
        st.pyplot(fig)
        plt.close(fig)

        # Waterfall
        st.subheader("SHAP 瀑布图 (Waterfall)")
        shap.plots.waterfall(shap_values[sample_idx], show=False)
        st.pyplot(plt.gcf())
        plt.close('all')

        # Bar (single sample)
        st.subheader("SHAP 单样本特征贡献 (Bar)")
        shap.plots.bar(shap_values[sample_idx], show=False)
        st.pyplot(plt.gcf())
        plt.close('all')

        # Decision plot (single sample) — notebook cell 41
        st.subheader("SHAP 单样本决策图 (Single Decision Plot)")
        st.caption("展示单个样本从基准值到最终预测的决策路径")
        single_sample = shap_values[sample_idx]
        shap.decision_plot(explainer.expected_value, single_sample.values,
                           X_shap.iloc[[sample_idx]],
                           feature_names=X_all.columns.tolist(), show=False)
        st.pyplot(plt.gcf())
        plt.close('all')

        # Decision plot (multi-sample)
        st.subheader("SHAP 决策图 (Decision Plot)")
        dec_start = st.number_input("起始样本", 0, shap_sample_size - 7, 0, 1, key='dec_start')
        dec_end = st.number_input("结束样本 (≤起始+20)", dec_start + 1,
                                   min(dec_start + 20, shap_sample_size), dec_start + 7, 1, key='dec_end')
        dec_indices = list(range(dec_start, dec_end))
        dec_features = X_shap.iloc[dec_indices]
        dec_shap_vals = explainer(dec_features)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.dpi = 150
        shap.decision_plot(explainer.expected_value, dec_shap_vals.values, dec_features,
                           feature_names=X_all.columns.tolist(), show=False)
        st.pyplot(fig)
        plt.close(fig)

        # ── 全样本堆积力图 (notebook cell 44) ──
        st.subheader("SHAP 全样本堆积力图 (Stacked Force Plot)")
        st.caption("所有样本的 SHAP 解释叠加视图，可观察整体预测趋势")
        shap_html = f"<head>{shap.getjs()}</head><body>{shap.plots.force(explainer.expected_value, shap_values.values, X_shap).html()}</body>"
        st.components.v1.html(shap_html, height=400, scrolling=True)

        # ── 多样本力图集 (notebook cells 48-54) ──
        st.subheader("SHAP 多样本力图集 (Multi-Sample Force Gallery)")
        gal_start = st.number_input("起始样本号", 0, shap_sample_size - 20, sample_idx, 1, key='gal_start')
        gal_count = st.slider("展示样本数", 1, min(15, shap_sample_size - gal_start), 5, 1, key='gal_count')
        gal_indices = list(range(gal_start, gal_start + gal_count))

        for sample_id in gal_indices:
            st.caption(f"样本 #{sample_id}")
            rounded_features = X_shap.iloc[sample_id].round(2)
            plt.rcParams.update({'font.size': 5})
            shap.plots.force(shap_values[sample_id], matplotlib=True, show=False,
                             figsize=(16, 3), features=rounded_features)
            plt.gcf().set_dpi(100)
            st.pyplot(plt.gcf())
            plt.close('all')
            plt.rcParams.update({'font.size': 14})


def show_model_export():
    st.header("💾 导出模型")
    if st.session_state.model is not None:
        buf = io.BytesIO()
        pickle.dump(st.session_state.model, buf)
        buf.seek(0)
        st.download_button(
            label="下载模型 (pickle)",
            data=buf,
            file_name="xgboost_model.pkl",
            mime="application/octet-stream"
        )


if train_clicked:
    run_training()


if st.session_state.trained and st.session_state.model is not None:
    show_results()
    show_model_export()
