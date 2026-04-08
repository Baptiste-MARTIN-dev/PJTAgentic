"""
app.py — Streamlit Dashboard for PJT MARS
==========================================
4-tab interface for the Wear-Aware Milling Optimization Digital Twin.

Tabs:
  1. Data Explorer    — Visualize physics model datasets
  2. Surrogate Model  — Train / load ensemble, inspect R² and predictions
  3. Optimizer        — Run all 3 optimizers, compare results, Pareto front
  4. Checkpoint Logic — Get action recommendation based on VB

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="PJT MARS",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project module imports (top-level so pickle can resolve class paths) ──────
from data_generator import generate_all_datasets, load_all_datasets, VB_LEVELS
from surrogate_model import train_ensemble, predict, save_model, load_model, DEFAULT_GBR_PARAMS
from candidate_space import build_candidate_space, GRID_CONFIG
from optimizers import run_all_optimizers
from checkpoint_logic import evaluate_checkpoint, get_action_color_map, VB_DERATE, VB_INSPECT, VB_REPLACE

DATA_DIR = Path(".")
MODEL_PATH = Path("ensemble_model.joblib")

# ── Session state initialisation ──────────────────────────────────────────────
for key, default in [
    ("ensemble", None),
    ("df_all", None),
    ("last_optimizer_results", None),
    ("train_r2", None),
    ("test_r2", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────
st.title("PJT MARS — Milling Optimization Digital Twin")

tab1, tab2, tab3, tab4 = st.tabs([
    "Data Explorer",
    "Surrogate Model",
    "Optimizer",
    "Checkpoint Logic",
])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Data Explorer
# ─────────────────────────────────────────────────────────────────────────────
def render_data_explorer():
    st.header("Dataset Explorer")

    st.latex(r"Ra = C \cdot f_z^{\alpha} \cdot V_c^{\beta} \cdot a_p^{\gamma} \cdot (1 + \delta \cdot VB + \eta \cdot VB \cdot f_z) + \varepsilon")
    st.caption(
        r"C=50.0,  α=1.6,  β=−0.5,  γ=0.15,  δ=2.0,  η=5.0,  ε ~ N(0, 0.02 µm)"
    )
    st.divider()

    col_gen, col_load = st.columns([1, 1])
    with col_gen:
        n_samples = st.number_input("Samples per VB level", min_value=100,
                                    max_value=5000, value=500, step=100)
        if st.button("Generate Datasets", type="primary"):
            with st.spinner("Generating datasets..."):
                generate_all_datasets(output_dir=DATA_DIR, n_samples=int(n_samples))
            st.success(f"Generated 4 CSVs ({n_samples} rows each).")
            st.session_state["df_all"] = None

    with col_load:
        if st.button("Load Existing CSVs"):
            try:
                st.session_state["df_all"] = load_all_datasets(DATA_DIR)
                st.success(f"Loaded {len(st.session_state['df_all'])} rows.")
            except FileNotFoundError as e:
                st.error(str(e))

    # Auto-load on first visit if CSVs exist
    if st.session_state["df_all"] is None:
        try:
            st.session_state["df_all"] = load_all_datasets(DATA_DIR)
        except FileNotFoundError:
            st.info("No datasets found. Click 'Generate Datasets' to create them.")
            return

    df = st.session_state["df_all"]

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.divider()
    cols = st.columns(4)
    for i, VB in enumerate(VB_LEVELS):
        sub = df[df["VB"] == VB]
        cols[i].metric(f"VB = {VB} mm", f"{len(sub)} rows",
                       delta=f"Ra̅ = {sub['Ra'].mean():.4f} µm")

    # ── Plot: Ra vs fz per VB level ───────────────────────────────────────────
    st.subheader("Ra vs fz per Wear Level")

    selected_vb = st.multiselect(
        "VB levels to display",
        options=VB_LEVELS,
        default=VB_LEVELS,
        format_func=lambda v: f"VB = {v} mm",
    )

    if selected_vb:
        fig_line = go.Figure()
        colors = px.colors.qualitative.Set1

        for j, VB in enumerate(selected_vb):
            sub = df[df["VB"] == VB].copy()
            sub["fz_bin"] = pd.cut(sub["fz"], bins=10)
            binned = sub.groupby("fz_bin", observed=True)["Ra"].agg(["mean", "std"]).reset_index()
            binned["fz_mid"] = binned["fz_bin"].apply(lambda x: x.mid)

            fig_line.add_trace(go.Scatter(
                x=binned["fz_mid"], y=binned["mean"],
                mode="lines+markers",
                name=f"VB={VB}",
                line=dict(color=colors[j % len(colors)]),
                error_y=dict(type="data", array=binned["std"].fillna(0).tolist(),
                             visible=True, color=colors[j % len(colors)]),
            ))

        fig_line.update_layout(
            xaxis_title="fz [mm/tooth]",
            yaxis_title="Ra [µm]",
            height=400,
            legend_title="Wear Level",
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # ── 3D scatter ────────────────────────────────────────────────────────────
    st.subheader("3D Scatter: Vc / fz / Ra")
    sub3d = df[df["VB"].isin(selected_vb)].copy() if selected_vb else df.copy()
    if len(sub3d) > 800:
        sub3d = sub3d.groupby("VB", group_keys=False).apply(
            lambda g: g.sample(min(200, len(g)), random_state=0)
        )

    fig3d = px.scatter_3d(
        sub3d, x="Vc", y="fz", z="Ra",
        color="VB",
        color_continuous_scale="RdYlGn_r",
        opacity=0.7,
        labels={"Vc": "Vc [m/min]", "fz": "fz [mm/tooth]", "Ra": "Ra [µm]"},
        height=500,
    )
    st.plotly_chart(fig3d, use_container_width=True)

    with st.expander("Raw data table"):
        st.dataframe(df.round(4), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Surrogate Model
# ─────────────────────────────────────────────────────────────────────────────
def render_surrogate_model():
    st.header("Bootstrap Ensemble Surrogate")

    col_train, col_load = st.columns([2, 1])

    with col_train:
        st.subheader("Train")
        n_bootstrap = st.slider("Bootstrap runs", min_value=10, max_value=100,
                                value=50, step=10)
        n_estimators = st.slider("GBR estimators", min_value=100, max_value=500,
                                 value=300, step=50)

        if st.button("Train Model", type="primary"):
            if st.session_state["df_all"] is None:
                try:
                    st.session_state["df_all"] = load_all_datasets(DATA_DIR)
                except FileNotFoundError:
                    st.error("No datasets found. Go to the Data Explorer tab first.")
                    return

            df = st.session_state["df_all"]
            gbr_params = {**DEFAULT_GBR_PARAMS, "n_estimators": n_estimators}
            progress_bar = st.progress(0, text="Training...")

            try:
                with st.spinner(f"Training {n_bootstrap} bootstrap models..."):
                    ens, tr2, te2 = train_ensemble(
                        df, n_bootstrap=n_bootstrap,
                        gbr_params=gbr_params, verbose=False
                    )

                progress_bar.progress(100, text="Done.")
                st.session_state["ensemble"] = ens
                st.session_state["train_r2"] = tr2
                st.session_state["test_r2"] = te2
                save_model(ens, MODEL_PATH)
                st.success(f"Model saved to {MODEL_PATH}")

            except ValueError as e:
                st.error(str(e))
                return

    with col_load:
        st.subheader("Load")
        if st.button("Load Saved Model"):
            if MODEL_PATH.exists():
                try:
                    ens = load_model(MODEL_PATH)
                    st.session_state["ensemble"] = ens
                    st.session_state["train_r2"] = ens.train_r2
                    st.session_state["test_r2"] = ens.test_r2
                    st.success(f"Loaded from {MODEL_PATH}")
                except Exception as e:
                    st.error(f"Failed to load: {e}")
            else:
                st.warning(f"{MODEL_PATH} not found.")

    # ── R² display ────────────────────────────────────────────────────────────
    if st.session_state["ensemble"] is not None:
        st.divider()
        ens = st.session_state["ensemble"]
        tr2 = st.session_state["train_r2"]
        te2 = st.session_state["test_r2"]

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Bootstrap runs", ens.n_bootstrap)
        col_m2.metric("Train R²", f"{tr2:.4f}")

        r2_status = "PASS" if te2 >= 0.80 else "FAIL"
        col_m3.metric("Test R²", f"{te2:.4f}", delta=f"{r2_status} (>=0.80 required)")

        # ── Ra vs fz validation curves ────────────────────────────────────────
        st.subheader("Predicted Ra vs fz per VB Level")
        st.caption("Vc = 150 m/min, ap = 1.5 mm fixed. Dots = binned data mean, dashed = model mean, band = ±2σ.")

        if st.session_state["df_all"] is not None:
            df = st.session_state["df_all"]
            fz_range = np.linspace(0.05, 0.30, 50)
            fig_valid = go.Figure()
            colors = px.colors.qualitative.Set1

            for j, VB in enumerate(VB_LEVELS):
                sub = df[df["VB"] == VB].copy()
                sub["fz_bin"] = pd.cut(sub["fz"], bins=8)
                binned = sub.groupby("fz_bin", observed=True)["Ra"].mean().reset_index()
                binned["fz_mid"] = binned["fz_bin"].apply(lambda x: x.mid)

                fig_valid.add_trace(go.Scatter(
                    x=binned["fz_mid"], y=binned["Ra"],
                    mode="markers",
                    name=f"Data VB={VB}",
                    marker=dict(color=colors[j % len(colors)], size=8, symbol="circle"),
                    showlegend=True,
                ))

                X_pred = pd.DataFrame({
                    "Vc": np.full(50, 150.0),
                    "fz": fz_range,
                    "ap": np.full(50, 1.5),
                    "VB": np.full(50, VB),
                })
                Ra_mean, Ra_sigma = predict(ens, X_pred)
                fig_valid.add_trace(go.Scatter(
                    x=fz_range, y=Ra_mean,
                    mode="lines",
                    name=f"Model VB={VB}",
                    line=dict(color=colors[j % len(colors)], dash="dash"),
                    showlegend=True,
                ))
                fig_valid.add_trace(go.Scatter(
                    x=np.concatenate([fz_range, fz_range[::-1]]),
                    y=np.concatenate([Ra_mean + 2*Ra_sigma, (Ra_mean - 2*Ra_sigma)[::-1]]),
                    fill="toself",
                    fillcolor=colors[j % len(colors)],
                    opacity=0.1,
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ))

            fig_valid.update_layout(
                xaxis_title="fz [mm/tooth]",
                yaxis_title="Ra [µm]",
                height=450,
                legend_title="Legend",
            )
            st.plotly_chart(fig_valid, use_container_width=True)

        # ── Single-point prediction ────────────────────────────────────────────
        st.subheader("Single-Point Prediction")
        pc1, pc2, pc3, pc4 = st.columns(4)
        p_Vc = pc1.slider("Vc [m/min]", 50.0, 300.0, 150.0, step=10.0)
        p_fz = pc2.slider("fz [mm/tooth]", 0.05, 0.30, 0.15, step=0.01)
        p_ap = pc3.slider("ap [mm]", 0.5, 3.0, 1.5, step=0.1)
        p_VB = pc4.slider("VB [mm]", 0.0, 0.35, 0.0, step=0.05)

        X_pt = pd.DataFrame({"Vc": [p_Vc], "fz": [p_fz], "ap": [p_ap], "VB": [p_VB]})
        Ra_mu, Ra_sig = predict(ens, X_pt)
        Ra_upper_pt = float(Ra_mu[0]) + 2.0 * float(Ra_sig[0])

        pr1, pr2, pr3 = st.columns(3)
        pr1.metric("Ra mean [µm]", f"{Ra_mu[0]:.4f}")
        pr2.metric("Ra sigma [µm]", f"{Ra_sig[0]:.4f}")
        pr3.metric("Ra + 2σ [µm]", f"{Ra_upper_pt:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Optimizer
# ─────────────────────────────────────────────────────────────────────────────
def render_optimizer():
    st.header("Optimizer")

    if st.session_state["ensemble"] is None:
        st.warning("No model loaded. Train or load a model in the Surrogate Model tab first.")
        return

    ens = st.session_state["ensemble"]

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        opt_VB = st.slider("Flank wear VB [mm]", 0.0, 0.35, 0.1, step=0.05)
    with col_in2:
        opt_Ra_target = st.slider("Ra target [µm]", 0.10, 2.0, 0.5, step=0.05)

    if st.button("Run All Optimizers", type="primary"):
        with st.spinner("Running optimizers..."):
            results = run_all_optimizers(ens, opt_VB, opt_Ra_target)
        st.session_state["last_optimizer_results"] = results

    if st.session_state["last_optimizer_results"] is None:
        st.info("Click 'Run All Optimizers' to see results.")
        return

    results = st.session_state["last_optimizer_results"]

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Results")

    rows = []
    for method, res in results.items():
        rows.append({
            "Method": method,
            "Feasible": "Yes" if res.feasible else "No",
            "Vc [m/min]": f"{res.best_Vc:.1f}" if res.best_Vc else "—",
            "fz [mm/t]":  f"{res.best_fz:.3f}" if res.best_fz else "—",
            "ap [mm]":    f"{res.best_ap:.2f}" if res.best_ap else "—",
            "MRR":        f"{res.best_MRR:.4f}" if res.best_MRR else "—",
            "Ra_mean":    f"{res.best_Ra_mean:.4f}" if res.best_Ra_mean else "—",
            "Ra+2σ":      (f"{res.best_Ra_mean + 2*res.best_Ra_sigma:.4f}"
                           if res.best_Ra_mean is not None else "—"),
            "Candidates OK": res.n_feasible,
            "Time [s]":   f"{res.solve_time_s:.3f}",
        })

    results_df = pd.DataFrame(rows)

    def highlight_feasible(row):
        color = "background-color: #d4edda" if row["Feasible"] == "Yes" else "background-color: #f8d7da"
        return [color] * len(row)

    st.dataframe(
        results_df.style.apply(highlight_feasible, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # ── MRR comparison bar chart ──────────────────────────────────────────────
    st.subheader("MRR Comparison")
    bar_data = []
    for method, res in results.items():
        bar_data.append({"Method": method,
                         "MRR": res.best_MRR if res.feasible else 0,
                         "Status": "Feasible" if res.feasible else "Infeasible"})

    fig_bar = px.bar(
        pd.DataFrame(bar_data),
        x="Method", y="MRR", color="Status",
        color_discrete_map={"Feasible": "#28a745", "Infeasible": "#dc3545"},
        text_auto=".4f",
        height=350,
        title=f"MRR by optimizer — VB={opt_VB} mm, Ra_target={opt_Ra_target} µm",
    )
    fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Pareto front (NSGA-II) ────────────────────────────────────────────────
    nsga_result = results.get("NSGA-II")
    if nsga_result and nsga_result.pareto_front is not None:
        st.subheader("NSGA-II Pareto Front")
        pf = nsga_result.pareto_front.copy()

        fig_pareto = px.scatter(
            pf,
            x="MRR", y="Ra_mean",
            color="feasible",
            size=pf["Ra_sigma"].clip(lower=0.001),
            color_discrete_map={True: "#28a745", False: "#dc3545"},
            labels={"MRR": "MRR", "Ra_mean": "Ra_mean [µm]",
                    "feasible": "Meets Ra_target", "Ra_sigma": "Ra σ"},
            title=f"Pareto front — bubble size = Ra uncertainty (VB={opt_VB})",
            height=450,
        )
        fig_pareto.add_hline(y=opt_Ra_target, line_dash="dash", line_color="orange",
                             annotation_text=f"Ra_target = {opt_Ra_target} µm",
                             annotation_position="top right")
        st.plotly_chart(fig_pareto, use_container_width=True)

        with st.expander("Pareto front data"):
            st.dataframe(pf.round(4), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Checkpoint Logic
# ─────────────────────────────────────────────────────────────────────────────
def render_checkpoint_logic():
    st.header("Tool Condition Checkpoint")

    if st.session_state["ensemble"] is None:
        st.warning("No model loaded. Train or load a model in the Surrogate Model tab first.")
        return

    ens = st.session_state["ensemble"]

    col_ck1, col_ck2 = st.columns(2)
    with col_ck1:
        ck_VB = st.number_input("Flank wear VB [mm]", min_value=0.0, max_value=0.5,
                                value=0.15, step=0.01, format="%.2f")
        ck_Ra_target = st.number_input("Ra target [µm]", min_value=0.05, max_value=5.0,
                                       value=0.5, step=0.05, format="%.3f")

    with col_ck2:
        with st.expander("Current operating parameters (optional)"):
            ck_Vc = st.number_input("Current Vc [m/min]", 50.0, 300.0, 150.0, step=10.0)
            ck_fz = st.number_input("Current fz [mm/tooth]", 0.05, 0.30, 0.15, step=0.01)
            ck_ap = st.number_input("Current ap [mm]", 0.5, 3.0, 1.5, step=0.1)
            use_current = st.checkbox("Include current params in analysis", value=False)

    if st.button("Evaluate Checkpoint", type="primary"):
        with st.spinner("Evaluating..."):
            decision = evaluate_checkpoint(
                VB=ck_VB,
                ensemble=ens,
                Ra_target=ck_Ra_target,
                Vc=ck_Vc if use_current else None,
                fz=ck_fz if use_current else None,
                ap=ck_ap if use_current else None,
            )

        st.divider()
        action_fn = {
            "CONTINUE": st.success,
            "DERATE":   st.warning,
            "INSPECT":  st.warning,
            "REPLACE":  st.error,
        }
        action_fn[decision.action](f"**{decision.action}** — {decision.message}")

        if decision.recommended_params:
            st.subheader("Recommended Cutting Parameters")
            rec = decision.recommended_params
            rc1, rc2, rc3, rc4, rc5 = st.columns(5)
            rc1.metric("Vc [m/min]", f"{rec['Vc']:.1f}")
            rc2.metric("fz [mm/tooth]", f"{rec['fz']:.3f}")
            rc3.metric("ap [mm]", f"{rec['ap']:.2f}")
            rc4.metric("MRR", f"{rec['MRR']:.4f}")
            rc5.metric("Ra + 2σ [µm]", f"{rec['Ra_upper']:.4f}")

        # ── VB wear gauge ─────────────────────────────────────────────────────
        st.subheader("Wear Level")
        fig_gauge = go.Figure()
        zones = [
            (0.0,       VB_DERATE,  "rgba(40,167,69,0.2)",  "CONTINUE"),
            (VB_DERATE, VB_INSPECT, "rgba(255,165,0,0.3)",  "DERATE"),
            (VB_INSPECT,VB_REPLACE, "rgba(255,165,0,0.4)",  "INSPECT"),
            (VB_REPLACE, 0.40,      "rgba(220,53,69,0.3)",  "REPLACE"),
        ]
        for x0, x1, color, label in zones:
            fig_gauge.add_vrect(x0=x0, x1=x1, fillcolor=color,
                                line_width=0, annotation_text=label,
                                annotation_position="top left")

        fig_gauge.add_vline(x=ck_VB, line_color="black", line_width=3,
                            annotation_text=f"VB={ck_VB:.2f}",
                            annotation_position="top")

        fig_gauge.update_layout(
            xaxis=dict(title="Flank wear VB [mm]", range=[0, 0.42]),
            yaxis=dict(visible=False),
            height=150,
            margin=dict(t=40, b=20, l=20, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Decision table for all VB checkpoints ────────────────────────────
        with st.expander("Decision for all standard VB checkpoints"):
            rows_ck = []
            for VB_ck in VB_LEVELS:
                d = evaluate_checkpoint(VB_ck, ens, ck_Ra_target)
                rows_ck.append({
                    "VB [mm]": VB_ck,
                    "Action": d.action,
                    "Feasible": "Yes" if d.feasible else "No",
                    "Candidates OK": d.n_feasible,
                    "Recommended Vc": f"{d.recommended_params['Vc']:.1f}" if d.recommended_params else "—",
                    "Recommended fz": f"{d.recommended_params['fz']:.3f}" if d.recommended_params else "—",
                    "Recommended MRR": f"{d.recommended_params['MRR']:.4f}" if d.recommended_params else "—",
                })
            st.dataframe(pd.DataFrame(rows_ck), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Render tabs
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    render_data_explorer()

with tab2:
    render_surrogate_model()

with tab3:
    render_optimizer()

with tab4:
    render_checkpoint_logic()
