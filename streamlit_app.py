import os
import streamlit as st
import pandas as pd
from core.config import MODELS, MAX_RETRIES_PER_STEP, CLEANED_DIR, DATA_DIR
from core.models import SessionState
from core.session import save_session
from core.logger import start_step, log_prompt, log_response, log_code, log_error, log_summary, flush_logs
from core.storage import save_df, load_df
from core.errors import *
from io.ingest import ingest_file
from utils.df_utils import detect_schema
from io.schema import apply_schema_edits
from io.convert import to_preview
from quality.profile import basic_quality_report, export_ydata_profile, export_sweetviz
from quality.correction import apply_corrections
from llm.codegen import generate_code
from llm.summarizer import summarize_strategy
from exec.executor import exec_generated_code
from exec.pipeline_builder import build_pipeline

st.set_page_config(page_title="LLM Data Cleaner", layout="wide")

if "state" not in st.session_state:
    st.session_state["state"] = SessionState(model_key=DEFAULT_MODEL_KEY)

state: SessionState = st.session_state["state"]

st.title("LLM-augmented Data Cleaning Pipeline")

with st.sidebar:
    st.header("Session")
    st.write(f"Session ID: {state.session_id}")
    model_key = st.selectbox("Model", options=list(MODELS.keys()), index=list(MODELS.keys()).index(state.model_key or DEFAULT_MODEL_KEY))
    state.model_key = model_key
    mode = st.radio("Mode", ["Human Reviewed", "Automatic"], index=0 if state.mode=="Human Reviewed" else 1)
    state.mode = mode

st.subheader("1) File Ingestion")
uploaded = st.file_uploader("Upload CSV / Excel / JSON / Text", type=["csv","xlsx","xls","json","txt"])

if uploaded is not None and st.button("Ingest"):
    path = os.path.join(DATA_DIR, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    try:
        source_type, dfs, sheets = ingest_file(path)
        state.dataset_path = path
        state.dataset_name = uploaded.name
        state.source_type = source_type
        if source_type == "excel":
            sheet = st.selectbox("Select sheet", sheets, key="sheet_select")
            state.active_sheet = sheet
            df = dfs[sheet]
        else:
            state.active_sheet = "default"
            df = dfs["default"]
        st.write("Preview:", to_preview(df))
        schema = detect_schema(df, source_type, sheets)
        state.schema = schema
        st.success("Ingested and schema detected.")
        save_session(state)
    except Exception as e:
        st.error(f"Ingestion failed: {e}")

if state.schema:
    st.subheader("2) Schema Detection & Editing")
    # Display schema
    sch_cols = []
    for c in state.schema.columns:
        sch_cols.append([c.name, c.dtype, c.is_categorical, c.date_format, c.null_count, ", ".join(map(str, c.example_values))])
    st.dataframe(pd.DataFrame(sch_cols, columns=["name","dtype","is_categorical","date_format","null_count","examples"]))

    with st.expander("Edit Schema"):
        edits = {}
        for c in state.schema.columns:
            col = c.name
            new_dtype = st.selectbox(f"{col} dtype", options=["keep","int64","float64","object","datetime"], index=0, key=f"dtype_{col}")
            date_fmt = st.text_input(f"{col} date format (optional)", value=c.date_format or "", key=f"dfmt_{col}")
            if new_dtype != "keep":
                edits[col] = {"dtype": new_dtype, "date_format": date_fmt}
        if st.button("Apply Edits"):
            try:
                # Load DataFrame
                if state.source_type == "excel":
                    df = pd.read_excel(state.dataset_path, sheet_name=state.active_sheet)
                elif state.source_type == "csv":
                    df = pd.read_csv(state.dataset_path)
                elif state.source_type == "json":
                    df = pd.read_json(state.dataset_path, lines=False)
                elif state.source_type == "text":
                    df = pd.read_csv(state.dataset_path)  # placeholder
                else:
                    df = pd.read_csv(state.dataset_path)
                df2 = apply_schema_edits(df, edits)
                st.write("Preview after edits:", to_preview(df2))
                # Update schema
                state.schema = detect_schema(df2, state.source_type, state.schema.sheets)
                # Save snapshot
                out_path = save_df(df2, f"{state.session_id}_schema_updated.parquet")
                state.df_snapshot_path = out_path
                st.success("Schema updated and dataset snapshot saved.")
                save_session(state)
            except Exception as e:
                st.error(f"Schema edit failed: {e}")

    st.subheader("3) Data Quality Profiling")
    if st.button("Run Profiling"):
        step = start_step(state, "Profiling")
        try:
            # Load current df
            df = load_df(state.df_snapshot_path) if state.df_snapshot_path else (
                pd.read_excel(state.dataset_path, sheet_name=state.active_sheet) if state.source_type=="excel" else
                pd.read_csv(state.dataset_path) if state.source_type in ["csv","txt"] else
                pd.read_json(state.dataset_path)
            )
            basic = basic_quality_report(df)
            st.json(basic)
            try:
                ypath = export_ydata_profile(df, state.session_id)
                st.info(f"ydata-profiling report saved: {ypath}")
            except Exception as e:
                st.warning(f"ydata-profiling failed: {e}")
            try:
                spath = export_sweetviz(df, state.session_id)
                st.info(f"Sweetviz report saved: {spath}")
            except Exception as e:
                st.warning(f"sweetviz failed: {e}")

            # Ask LLM for deeper profiling code
            issues = basic
            task = "Generate Python code to perform deeper data profiling and quality checks on df. Produce variables 'df_out' if modifications, and print or assign metrics to variables."
            code = generate_code(task, schema=state.schema.__dict__, issues=issues, user_instructions=None, prev_errors=None, model_key=state.model_key)
            log_code(step, code)
            st.code(code, language="python")
            summary = summarize_strategy("Deeper quality profiling code generated.", state.model_key)
            log_summary(step, summary)
            st.write("LLM summary:", summary)
            agree = True
            if state.mode == "Human Reviewed":
                agree = st.checkbox("Approve and execute profiling code?", value=False)
            if agree:
                try:
                    df2 = exec_generated_code(df, code)
                    st.success("LLM profiling code executed.")
                    out_path = save_df(df2, f"{state.session_id}_profiled.parquet")
                    state.df_snapshot_path = out_path
                    state.executed_code_blocks.append(code)
                    save_session(state)
                except Exception as e:
                    log_error(step, str(e))
                    st.error(f"Execution error: {e}")
            flush_logs(state)
        except Exception as e:
            log_error(step, str(e))
            st.error(f"Profiling failed: {e}")
            flush_logs(state)

    st.subheader("4) Data Correction")
    user_instr = st.text_area("Optional correction instructions (NL)", placeholder="e.g., fill missing age with median, drop duplicates, normalize city names.")
    if st.button("Propose and Apply Corrections"):
        step = start_step(state, "Correction")
        try:
            df = load_df(state.df_snapshot_path) if state.df_snapshot_path else (
                pd.read_excel(state.dataset_path, sheet_name=state.active_sheet) if state.source_type=="excel" else
                pd.read_csv(state.dataset_path) if state.source_type in ["csv","txt"] else
                pd.read_json(state.dataset_path)
            )
            # Build correction code via LLM
            issues = basic_quality_report(df)
            task = "Generate Python code to correct data quality issues on df based on issues and user instructions. Set df_out as the cleaned DataFrame."
            code = generate_code(task, schema=state.schema.__dict__, issues=issues, user_instructions=user_instr, prev_errors=None, model_key=state.model_key)
            log_code(step, code)
            st.code(code, language="python")
            summary = summarize_strategy("Data correction plan generated and will be applied.", state.model_key)
            log_summary(step, summary)
            st.write("LLM summary:", summary)
            agree = True
            if state.mode == "Human Reviewed":
                agree = st.checkbox("Approve and execute correction code?", value=False, key="approve_correction")
            if agree:
                try:
                    df2 = exec_generated_code(df, code)
                    st.success("Corrections executed.")
                    out_path = save_df(df2, f"{state.session_id}_cleaned.parquet")
                    state.df_snapshot_path = out_path
                    state.executed_code_blocks.append(code)
                    # Update schema after cleaning
                    state.schema = detect_schema(df2, state.source_type, state.schema.sheets)
                    save_session(state)
                except Exception as e:
                    log_error(step, str(e))
                    st.error(f"Execution error: {e}")
            flush_logs(state)
        except Exception as e:
            log_error(step, str(e))
            st.error(f"Correction failed: {e}")
            flush_logs(state)

    st.subheader("5) Finalization")
    if st.button("Build Final Pipeline (.py)"):
        try:
            pipeline_path = build_pipeline(state.executed_code_blocks, state.session_id)
            st.success(f"Pipeline built at: {pipeline_path}")
        except Exception as e:
            st.error(f"Pipeline build failed: {e}")

    if state.df_snapshot_path:
        st.download_button("Download Cleaned Dataset (.parquet)", data=open(state.df_snapshot_path, "rb").read(), file_name=os.path.basename(state.df_snapshot_path))
    logs_path = os.path.join(DATA_DIR, "logs", f"{state.session_id}.json")
    if os.path.exists(logs_path):
        st.download_button("Download Logs (.json)", data=open(logs_path, "rb").read(), file_name=os.path.basename(logs_path))
    pipeline_candidate = os.path.join(DATA_DIR, "pipeline", f"pipeline_{state.session_id}.py")
    if os.path.exists(pipeline_candidate):
        st.download_button("Download Final Pipeline (.py)", data=open(pipeline_candidate, "rb").read(), file_name=os.path.basename(pipeline_candidate))
