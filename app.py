import gc
import hashlib
import json
import logging
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import altair as alt
import gspread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from google.oauth2.service_account import Credentials
from gspread.exceptions import WorksheetNotFound
from PIL import Image
from sklearn.metrics import confusion_matrix
from streamlit_gsheets import GSheetsConnection

# ==============================================================================
# ==== STREAMLIT CLOUD LIVE ANSI-COLOR GRID LOGGING CONFIGURATION ====
# ==============================================================================

class CloudLogFormatter(logging.Formatter):
    # ANSI Terminal Palette Codes
    RESET = "\033[0m"
    ORANGE = "\033[33m"
    GREEN = "\033[32m"
    MAX_USER_LENGTH = 10
    MAX_COMP_LENGTH = 10

    def format(self, record):
        level_map = {
            "DEBUG": "DEBUG",
            "INFO": "INFO",
            "WARNING": "WARN",
            "ERROR": "ERROR",
            "CRITICAL": "FATAL",
        }
        raw_user = str(getattr(record, "user", "SYSTEM"))
        user_formatted = (
            raw_user[:self.MAX_USER_LENGTH]
            if len(raw_user) > self.MAX_USER_LENGTH
            else raw_user.ljust(self.MAX_USER_LENGTH)
        )

        raw_comp = str(getattr(record, "comp", "CORE"))
        comp_formatted = (
            raw_comp[:self.MAX_COMP_LENGTH]
            if len(raw_comp) > self.MAX_COMP_LENGTH
            else raw_comp.ljust(self.MAX_COMP_LENGTH)
        )

        asctime = self.formatTime(record, self.datefmt)
        levelname = level_map.get(record.levelname, f"{record.levelname:<5}")
        msg = record.getMessage()

        if "waitlisted" in msg.lower():
            color_prefix = self.ORANGE
        elif "completed" in msg.lower() or "success" in msg.lower():
            color_prefix = self.GREEN
        else:
            color_prefix = ""

        # Assemble the final log stream grid string
        if color_prefix:
            return f"{asctime} {levelname} - {user_formatted} {comp_formatted} {color_prefix}{msg}{self.RESET}"
        return f"{asctime} {levelname} - {user_formatted} {comp_formatted} {msg}"

# Instantiate stream handlers bound directly to sys.stdout
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(CloudLogFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [log_handler]

# Force stdout to be completely unbuffered on Streamlit Cloud containers
sys.stdout.reconfigure(line_buffering=True)

# ==============================================================================

ALLOWED_MODELS = [
    "Custom",
    "DenseNet121",
    "DenseNet169",
    "DenseNet201",
    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetB2",
    "EfficientNetB3",
    "EfficientNetB4",
    "EfficientNetV2B0",
    "EfficientNetV2B1",
    "EfficientNetV2B2",
    "EfficientNetV2B3",
    "EfficientNetV2S",
    "InceptionV3",
    "MobileNet",
    "MobileNetV2",
    "MobileNetV3Small",
    "MobileNetV3Large",
    "NASNetMobile",
    "ResNet50",
    "ResNet50V2",
    "VGG16",
    "VGG19",
    "Xception",
 ]

TEST_IMAGE_DIR = "test_images"
CLASS_NAMES = ["A", "B", "C"]
REQUIRED_COLUMNS = ["participant", "accuracy", "submission_time", "batch", "model_type"]

help_leaderboard_toggle = """By default, the leaderboard displays one entry per participant and model type.\n\n
Toggle if you prefer to see only one entry per participant.
"""

help_preprocessing = """Select whether or not your model is performing
the necessary preprocessing steps on its own.\n
If you select `Yes`, the app assumes your pipeline includes either
- a normal rescaling layer or
- a keras preprocessing layer of the format
`preprocessor = Lambda(_________.preprocess_input)`\n
If you select `No`, the app will apply
- rescaling to [0,1] in case of custom models
- the model family's own preprocessor for pre-trained models
"""


@st.cache_resource
def get_global_store() -> dict:
    return {
        "submissions": {},
        "alltime_submissions": None,
        "leaderboards": {},
        "alltime_leaderboard": None,
        "batches": None,
        "batches_last_updated": None,
        "gsheet_conn": None,
        "configured_batches": set(),
        "eval_lock": threading.Lock(),
    }


def state_inits() -> None:
    if "user_name" not in st.session_state:
        st.session_state.user_name = None
    if "code_input" not in st.session_state:
        st.session_state.code_input = None
    if "batch" not in st.session_state:
        st.session_state.batch = None
    if "alltime" not in st.session_state:
        st.session_state.alltime = None

    store = get_global_store()

    if store["gsheet_conn"] is None:
        configure_gsheet(_store=store)

    if st.session_state.alltime and store["alltime_submissions"] is None:
        load_alltime_data(store)


def load_alltime_data(store: dict) -> None:
    try:
        batches_df = store["gsheet_conn"].read(worksheet="Batches", ttl=0)
        batches = batches_df["Batch"].tolist()
        worksheet_titles = [b for b in batches if b not in ["Batches", "anonymous"]]

        dfs = []
        for ws_name in worksheet_titles:
            try:
                df = store["gsheet_conn"].read(worksheet=ws_name, ttl=0)
                if df is not None and not df.empty:
                    dfs.append(df)
            except:
                pass

        store["alltime_submissions"] = (
            pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        )
        build_leaderboards()
    except Exception:
        store["alltime_submissions"] = pd.DataFrame()


@st.cache_resource
def get_gsheet_connection() -> GSheetsConnection:
    return st.connection("gsheets", type=GSheetsConnection)


@st.cache_resource
def configure_gsheet(_store: dict, batch: str | None = None) -> str:
    try:
        if _store["gsheet_conn"] is None:
            _store["gsheet_conn"] = get_gsheet_connection()

        if batch and batch not in _store["configured_batches"]:
            ensure_batch_sheet_exists(batch, _store["gsheet_conn"])
            _store["configured_batches"].add(batch)

        return "Successful"
    except Exception as e:
        return f"Error: {e}"


def display_admin() -> None:
    st.divider()
    st.subheader("🛠️ Admin Settings", anchor=False)
    if st.button("Clear cached resources"):
        get_global_store.clear()
        configure_gsheet.clear()
        get_gsheet_connection.clear()
        st.success("Cache cleared successfully! Refreshing app...")
        st.rerun()


def _open_spreadsheet() -> gspread.Spreadsheet:
    creds_dict = dict(st.secrets["connections"]["gsheets"])
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    return client.open_by_url(creds_dict["spreadsheet"])


def ensure_batch_sheet_exists(batch: str, conn: GSheetsConnection) -> None:
    try:
        conn.read(worksheet=batch, ttl=0)
    except WorksheetNotFound:
        sh = _open_spreadsheet()
        sh.add_worksheet(title=batch, rows="1000", cols="10")
        empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        conn.update(worksheet=batch, data=empty_df)
    except Exception as e:
        st.error(f"Failed to verify/create worksheet: {e}")


def generate_leaderboard_dataframe(
    submissions_df: pd.DataFrame,
    *,
    reduce_leaderboard: bool,
) -> pd.DataFrame:
    if submissions_df.empty:
        return pd.DataFrame()

    if reduce_leaderboard:
        groupby = ["participant", "batch"]
    else:
        groupby = ["participant", "batch", "model_type"]

    return (
        submissions_df.assign(
            attempts=lambda df_: df_.groupby(groupby)["participant"].transform("count"),
        )
        .sort_values(["accuracy", "submission_time"], ascending=[False, True])
        .drop_duplicates(subset=groupby, keep="first")
        .assign(
            position=lambda df_: df_["accuracy"]
            .rank(method="min", ascending=False)
            .astype(int)
        )
        .assign(
            position=lambda df_: df_["position"]
            .where(df_["position"].ne(df_["position"].shift()), "")
            .astype(str)
        )
        .set_index("position")
        .filter(["participant", "batch", "model_type", "accuracy", "attempts"])
    )


def build_leaderboards() -> None:
    store = get_global_store()
    for batch, df in store["submissions"].items():
        if batch != "anonymous" and df is not None and not df.empty:
            store["leaderboards"][batch] = generate_leaderboard_dataframe(
                df,
                reduce_leaderboard=False,
            )
        else:
            store["leaderboards"][batch] = pd.DataFrame()

    if (
        store["alltime_submissions"] is not None
        and not store["alltime_submissions"].empty
    ):
        store["alltime_leaderboard"] = generate_leaderboard_dataframe(
            store["alltime_submissions"],
            reduce_leaderboard=False,
        )


def update_submissions(participant_results: pd.DataFrame) -> None:
    store = get_global_store()
    batch = st.session_state.batch

    current_submissions = store["submissions"].get(batch, pd.DataFrame())
    updated_df = pd.concat(
        [current_submissions, participant_results],
        ignore_index=True,
    )

    try:
        store["gsheet_conn"].update(worksheet=batch, data=updated_df)
        store["submissions"][batch] = updated_df

        if batch != "anonymous" and store["alltime_submissions"] is not None:
            store["alltime_submissions"] = pd.concat(
                [store["alltime_submissions"], participant_results],
                ignore_index=True,
            )

        build_leaderboards()
    except Exception as e:
        st.error(f"Could not update Google Sheets: {e}")


def get_participant_info() -> None:
    store = get_global_store()

    if st.session_state.user_name and st.session_state.batch:
        if st.session_state.batch not in store["submissions"]:
            try:
                configure_gsheet(_store=store, batch=st.session_state.batch)
                store["submissions"][st.session_state.batch] = store[
                    "gsheet_conn"
                ].read(worksheet=st.session_state.batch, ttl=0)
                build_leaderboards()
            except Exception as e:
                st.error(f"Error loading batch data: {e}")
                st.stop()

        st.info(
            f"Logged in as: **{st.session_state.user_name}** from **{st.session_state.batch}**",
        )
    else:
        st.write("Please log in with the details provided by your instructor.")
        st.divider()

        if (
            store["batches"] is None
            or (pd.Timestamp.now() - store["batches_last_updated"]).seconds > 600
        ):
            try:
                if store["gsheet_conn"] is None:
                    configure_gsheet(_store=store)
                store["batches"] = store["gsheet_conn"].read(worksheet="Batches", ttl=0)
                store["batches_last_updated"] = pd.Timestamp.now()
            except Exception:
                st.error("Database connection failed.")
                st.stop()

        user_name = st.text_input("Username (Real Name or Alias):")
        code_input = st.text_input("Secret Batch Code:", type="password")

        if user_name and code_input:
            batches_df = store["batches"]
            code_map = batches_df.set_index("Code")

            if code_input in code_map.index:
                row = code_map.loc[code_input]
                st.session_state.user_name = user_name
                st.session_state.code_input = code_input
                st.session_state.batch = row["Batch"]
                st.session_state.alltime = row["Show All-time?"]
                st.rerun()
            else:
                st.error("Invalid Code.")


def plot_submissions(participant_name: str) -> None:
    store = get_global_store()
    batch = st.session_state.batch
    if batch not in store["submissions"]:
        return

    participant_submissions = (
        store["submissions"][batch]
        .query("participant == @participant_name")
        .filter(["model_type", "submission_time", "accuracy"])
        .copy()
    )

    if len(participant_submissions) > 1:
        st.divider()
        st.subheader("📊 Your progress over time", anchor=False)
        participant_submissions["submission_time"] = pd.to_datetime(
            participant_submissions["submission_time"],
            format="ISO8601",
        )
        participant_submissions = participant_submissions.sort_values("submission_time")
        line = (
            alt.Chart(participant_submissions)
            .mark_line()
            .encode(
                x="submission_time:T",
                y="accuracy:Q",
            )
        )

        points = (
            alt.Chart(participant_submissions)
            .mark_point(filled=True, size=150)
            .encode(
                x="submission_time:T",
                y="accuracy:Q",
                color="model_type:N",
                tooltip=["submission_time:T", "model_type:N", "accuracy:Q"],
            )
        )

        chart = alt.layer(line, points).interactive()
        st.altair_chart(chart, width="stretch")
    elif len(participant_submissions):
        st.success("First submission recorded! Submit more models to see your progress chart.")


@st.fragment(run_every=10)
def show_leaderboard() -> None:
    if st.session_state.batch == "anonymous":
        st.info("You are currently in an anonymous session. You won't see or appear on any public leaderboards.")
        return

    store = get_global_store()
    batch = st.session_state.batch

    st.divider()
    st.header(f"🏆 {batch} Leaderboard", anchor=False)

    submissions_df = store["submissions"].get(batch, pd.DataFrame())

    if not submissions_df.empty:
        reduce_leaderboard = st.toggle(
            "Reduce leaderboard to one entry per participant?",
            help=help_leaderboard_toggle,
        )
        view = generate_leaderboard_dataframe(
            submissions_df,
            reduce_leaderboard=reduce_leaderboard,
        )
        st.dataframe(view.drop("batch", axis=1, errors="ignore"), width="stretch")
    else:
        st.write("No submissions yet for this batch.")

    if (
        st.session_state.alltime
        and store["alltime_submissions"] is not None
        and not store["alltime_submissions"].empty
    ):
        st.divider()
        st.header("👑 All-time Global Leaderboard", anchor=False)
        reduce_leaderboard = st.toggle(
            "Reduce all-time leaderboard to one entry per participant?",
            help=help_leaderboard_toggle,
        )
        at_view = generate_leaderboard_dataframe(
            store["alltime_submissions"],
            reduce_leaderboard=reduce_leaderboard,
        )
        st.dataframe(at_view, width="stretch")


def render_html_metric_banner(score, baseline_score, label):
    diff = score - baseline_score
    if diff > 0.05:
        color = "#155724"
        bg = "#d4edda"
    elif diff >= -0.015:
        color = "#28a745"
        bg = "#e2f0d9"
    elif diff >= -0.035:
        color = "#ffc107"
        bg = "#fff3cd"
    elif diff >= -0.075:
        color = "#fd7e14"
        bg = "#ffe8d6"
    else:
        color = "#dc3545"
        bg = "#f8d7da"

    st.markdown(
        f"""
        <div style="background-color:{bg}; padding:12px; border-radius:5px; border-left:5px solid {color}; margin-bottom:10px;">
            <h4 style="margin:0 0 5px 0; color:#333;">{label}</h4>
            <p style="margin:0; font-size:18px; font-weight:bold; color:{color};">
                Total Accuracy: {score:.2%}
                <br>
                ({diff:+.2%} vs Baseline)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_matrix_and_metric(y_true, y_pred, label, score, baseline_score):
    if not y_true or not y_pred:
        st.warning(f"⚠️ No samples available to compile data evaluation matrix for: {label}")
        return

    render_html_metric_banner(score, baseline_score, label)

    fig, ax = plt.subplots(figsize=(2, 2), facecolor="black")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="copper",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        cbar=False,
        annot_kws={"color": "white", "fontsize": 8},
    )
    ax.set_xlabel("Predicted", color="white", fontsize=8)
    ax.set_ylabel("True Label", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=8, which="both", length=0)
    st.pyplot(fig, width="content")
    plt.close(fig)


def run_evaluation_process(model_path, model_type, apply_preprocess, flip_val, rot_val, zoom_val="normal"):
    results_path = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    try:
        process = subprocess.run([
            sys.executable, "evaluator.py",
            "--model_path", model_path,
            "--model_type", model_type,
            "--apply_preprocess", str(apply_preprocess),
            "--flip", flip_val,
            "--rotate", rot_val,
            "--zoom", zoom_val,
            "--output_json", results_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if process.returncode != 0:
            raise RuntimeError(f"Subprocess failed: {process.stderr}")

        with open(results_path, "r") as f:
            return json.load(f)
    finally:
        Path(results_path).unlink(missing_ok=True)


def main() -> None:
    st.set_page_config(page_title="Sign Language Showdown", page_icon="✊")

    if st.session_state.get("batch") == "Instructor":
        display_admin()

    st.title("Sign Language Model Showdown!", anchor=False, text_alignment="center")
    st.title("✊🖐️🤏", anchor=False, text_alignment="center")

    state_inits()
    get_participant_info()

    store = get_global_store()

    if st.session_state.user_name and st.session_state.batch:
        st.subheader("📤 Submit Your Model", anchor=False)
        cols = st.columns(2, gap="large")
        with cols[0]:
            model_type = st.selectbox(
                "Select the exact model used:",
                options=ALLOWED_MODELS,
                index=None,
                help="Note: Large models are disabled for stability."
            )
        with cols[1]:
            apply_preprocess = st.radio(
                "Does your model handle preprocessing?",
                options=["Yes", "No"],
                index=0,
                help=help_preprocessing,
            ) == "No"

        if model_type:
            uploaded_file = st.file_uploader("Select a Keras model file", type=["keras"])

            if uploaded_file:
                file_bytes = uploaded_file.getvalue()
                file_hash = hashlib.sha256(file_bytes).hexdigest()
                file_size_mb = len(file_bytes) / (1024 * 1024)

                if st.session_state.get("last_processed_hash") != file_hash:
                    logger.info(
                        f"{model_type} uploaded successfully. Size: {file_size_mb:.2f} MB",
                        extra={"user": st.session_state.user_name, "comp": "UPLOADER"},
                    )
                    st.session_state.last_processed_hash = file_hash
                    st.session_state.baseline_run_data = None
                    st.session_state.deep_handedness_data = None
                    st.session_state.deep_perp_data = None
                    st.session_state.deep_zoom_data = None
                    st.session_state.deep_inversion_data = None

                    # Polling State Waitlist Initializations
                    st.session_state.waiting_for_baseline = False
                    st.session_state.trigger_baseline_eval = False
                    st.session_state.waiting_for_handedness = False
                    st.session_state.trigger_handedness_eval = False
                    st.session_state.waiting_for_perp = False
                    st.session_state.trigger_perp_eval = False
                    st.session_state.waiting_for_inversion = False
                    st.session_state.trigger_inversion_eval = False
                    st.session_state.waiting_for_zoom = False
                    st.session_state.trigger_zoom_eval = False

                if st.session_state.get("baseline_run_data") is None:
                    # ---- BASELINE NON-BLOCKING POLLING STATE VALVE ----
                    if st.session_state.get("waiting_for_baseline", False):
                        if store["eval_lock"].locked():
                            st.warning("⏳ Server busy: Another user is running evaluations. Retrying automatically...")
                            time.sleep(20)
                            st.rerun()
                        else:
                            st.session_state.waiting_for_baseline = False
                            st.session_state.trigger_baseline_eval = True
                            st.rerun()

                    # Intercept execution thread if lock is currently held
                    if store["eval_lock"].locked() and not st.session_state.get("trigger_baseline_eval", False):
                        logger.warning(
                            f"User is waitlisted.",
                            extra={"user": st.session_state.user_name, "comp": "BASELINE"},
                        )
                        st.session_state.waiting_for_baseline = True
                        st.rerun()

                    st.session_state.trigger_baseline_eval = False

                    try:
                        with st.spinner("Analyzing model performance..."):
                            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmpf:
                                tmpf.write(uploaded_file.getbuffer())
                                saved_model_path = tmpf.name

                            st.session_state.saved_model_path = saved_model_path

                            logger.info(
                                f"Evaluation started: {model_type}",
                                extra={"user": st.session_state.user_name, "comp": "BASELINE"},
                            )
                            with store["eval_lock"]:
                                baseline_run = run_evaluation_process(
                                    saved_model_path, model_type, apply_preprocess, "False", "0"
                                )
                            st.session_state.baseline_run_data = baseline_run

                            acc = baseline_run["overall_accuracy"]
                            result = pd.DataFrame([
                                {
                                    "accuracy": round(acc, 4),
                                    "participant": st.session_state.user_name,
                                    "batch": st.session_state.batch,
                                    "submission_time": pd.Timestamp.now().isoformat(),
                                    "model_type": model_type,
                                },
                            ])
                            update_submissions(result)
                            logger.info(
                                f"Completed. Accuracy: {acc:.4f}",
                                extra={"user": st.session_state.user_name, "comp": "BASELINE"},
                            )
                        st.rerun()
                    except Exception as e:
                        logger.error(
                            f"Baseline evaluation failed: {str(e)}",
                            extra={"user": st.session_state.user_name, "comp": "BASELINE"},
                        )
                        st.error(f"Error evaluating model baseline: {e}")
                    finally:
                        uploaded_file = None
                        gc.collect()

                if st.session_state.get("baseline_run_data") is not None:
                    baseline_run = st.session_state.baseline_run_data
                    acc = baseline_run["overall_accuracy"]
                    y_pred = [pred["y_pred"] for pred in baseline_run["predictions"].values()]
                    y_test = [pred["y_true"] for pred in baseline_run["predictions"].values()]

                    st.success(f"Success! Model Accuracy: {acc:.2%}")

                    st.subheader("🧮 Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(2, 2), facecolor="black")
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="copper",
                        xticklabels=CLASS_NAMES,
                        yticklabels=CLASS_NAMES,
                        ax=ax,
                        cbar=False,
                        annot_kws={"color": "white", "fontsize": 8},
                    )
                    ax.set_xlabel("Predicted", color="white", fontsize=8)
                    ax.set_ylabel("True Label", color="white", fontsize=8)
                    ax.tick_params(colors="white", labelsize=8, which="both", length=0)
                    st.pyplot(fig, width="content")
                    plt.close(fig)

                    gc.collect()

        plot_submissions(st.session_state.user_name)
        show_leaderboard()

        # ==== ADVANCED DIAGNOSTICS CONTROL CENTER PANEL ====
        if st.session_state.get("baseline_run_data") is not None:
            st.divider()

            show_diagnostics = st.toggle("🔍 Show Advanced Diagnostic Robustness Stress-Testing", value=False)

            if show_diagnostics:
                st.header("🔍 Advanced Diagnostic: Robustness Stress-Testing", anchor=False)
                st.write("Analyze your network's vulnerabilities against variance in "
                         "handedness, horizontal hand orientations, "
                         "total canvas inversions, and scaling.")

                saved_model_path = st.session_state.saved_model_path
                baseline_run = st.session_state.baseline_run_data
                acc = baseline_run["overall_accuracy"]

                # --- STEP 1: HANDEDNESS INVARIANT VERIFICATION ---
                st.markdown("### 🖐️ Handedness Invariant Verification")
                if st.session_state.get("deep_handedness_data") is None:
                    # ---- HANDEDNESS NON-BLOCKING POLLING AUTOMATION VALVE ----
                    if st.session_state.get("waiting_for_handedness", False):
                        if store["eval_lock"].locked():
                            st.warning("⏳ Server busy: Another user is running evaluations. Retrying automatically...")
                            time.sleep(20)
                            st.rerun()
                        else:
                            st.session_state.waiting_for_handedness = False
                            st.session_state.trigger_handedness_eval = True
                            st.rerun()

                    if st.button("Evaluate Handedness Robustness") or st.session_state.get("trigger_handedness_eval", False):
                        if store["eval_lock"].locked() and not st.session_state.get("trigger_handedness_eval", False):
                            logger.warning(
                                f"User is waitlisted.",
                                extra={"user": st.session_state.user_name, "comp": "HANDEDNESS"}
                            )
                            st.session_state.waiting_for_handedness = True
                            st.rerun()

                        st.session_state.trigger_handedness_eval = False
                        progress_bar = st.progress(0)
                        status_text = st.info("Running Handedness matrix configuration...")
                        
                        logger.info(
                            f"Evaluation started: {model_type}",
                            extra={"user": st.session_state.user_name, "comp": "HANDEDNESS"},
                        )
                        with store["eval_lock"]:
                            flipped_0 = run_evaluation_process(saved_model_path, model_type, apply_preprocess, "True", "0")
                        
                        progress_bar.progress(1.0)
                        status_text.empty()
                        progress_bar.empty()
                        st.session_state.deep_handedness_data = {"flipped_0": flipped_0}
                        logger.info(
                            f"Completed.",
                            extra={"user": st.session_state.user_name, "comp": "HANDEDNESS"},
                        )
                        st.rerun()
                else:
                    handedness_slices = st.session_state.deep_handedness_data
                    left_y_true, left_y_pred = [], []
                    right_y_true, right_y_pred = [], []

                    real_left_true, real_left_pred = [], []
                    sim_left_true, sim_left_pred = [], []
                    real_right_true, real_right_pred = [], []
                    sim_right_true, sim_right_pred = [], []

                    base_left_total, base_left_correct = 0, 0
                    base_right_total, base_right_correct = 0, 0

                    for fname, p in baseline_run["predictions"].items():
                        hand = str(p.get("native_handedness", "Unknown")).strip().capitalize()
                        if hand == "Left":
                            left_y_true.append(p["y_true"])
                            left_y_pred.append(p["y_pred"])
                            real_left_true.append(p["y_true"])
                            real_left_pred.append(p["y_true"])
                            real_left_pred[-1] = p["y_pred"]
                            base_left_total += 1
                            if p["correct"]: base_left_correct += 1
                        elif hand == "Right":
                            right_y_true.append(p["y_true"])
                            right_y_pred.append(p["y_pred"])
                            real_right_true.append(p["y_true"])
                            real_right_pred.append(p["y_true"])
                            real_right_pred[-1] = p["y_pred"]
                            base_right_total += 1
                            if p["correct"]: base_right_correct += 1

                    for fname, p in handedness_slices["flipped_0"]["predictions"].items():
                        hand = str(p.get("native_handedness", "Unknown")).strip().capitalize()
                        if hand == "Left":
                            right_y_true.append(p["y_true"])
                            right_y_pred.append(p["y_pred"])
                            sim_right_true.append(p["y_true"])
                            sim_right_pred.append(p["y_pred"])
                        elif hand == "Right":
                            left_y_true.append(p["y_true"])
                            left_y_pred.append(p["y_pred"])
                            sim_left_true.append(p["y_true"])
                            sim_left_pred.append(p["y_pred"])

                    left_acc = np.mean(np.array(left_y_true) == np.array(left_y_pred)) if left_y_true else 0.0
                    right_acc = np.mean(np.array(right_y_true) == np.array(right_y_pred)) if right_y_true else 0.0

                    real_left_acc = np.mean(np.array(real_left_true) == np.array(real_left_pred)) if real_left_true else 0.0
                    sim_left_acc = np.mean(np.array(sim_left_true) == np.array(sim_left_pred)) if sim_left_true else 0.0
                    real_right_acc = np.mean(np.array(real_right_true) == np.array(real_right_pred)) if real_right_true else 0.0
                    sim_right_acc = np.mean(np.array(sim_right_true) == np.array(sim_right_pred)) if sim_right_true else 0.0

                    col_h1, col_h2 = st.columns(2)
                    with col_h1:
                        render_matrix_and_metric(
                            left_y_true, left_y_pred, "Left-Handed Images", left_acc, acc
                        )
                        st.caption(f"• Real Left-Handed Samples: **{real_left_acc:.2%}** ({base_left_correct}/{base_left_total})")
                        st.caption(f"• Simulated Left-Handed Samples (Flipped Rights): **{sim_left_acc:.2%}**")
                    with col_h2:
                        render_matrix_and_metric(
                            right_y_true, right_y_pred, "Right-Handed Images", right_acc, acc
                        )
                        st.caption(f"• Real Right-Handed Samples: **{real_right_acc:.2%}** ({base_right_correct}/{base_right_total})")
                        st.caption(f"• Simulated Right-Handed Samples (Flipped Lefts): **{sim_right_acc:.2%}**")

                # --- STEP 2: HORIZONTAL ORIENTATIONS OF HANDS ---
                st.write("")
                st.markdown("### 🫱 Horizontal Orientations of Hands (90° & -90°)")
                if st.session_state.get("deep_perp_data") is None:
                    # ---- HORIZONTAL ORIENTATIONS NON-BLOCKING POLLING AUTOMATION VALVE ----
                    if st.session_state.get("waiting_for_perp", False):
                        if store["eval_lock"].locked():
                            st.warning("⏳ Server busy: Another user is running evaluations. Retrying automatically...")
                            time.sleep(20)
                            st.rerun()
                        else:
                            st.session_state.waiting_for_perp = False
                            st.session_state.trigger_perp_eval = True
                            st.rerun()

                    if st.button("Evaluate Perpendicular Robustness") or st.session_state.get("trigger_perp_eval", False):
                        if store["eval_lock"].locked() and not st.session_state.get("trigger_perp_eval", False):
                            logger.warning(
                                f"User is waitlisted.",
                                extra={"user": st.session_state.user_name, "comp": "HORIZ_ROT"},
                            )
                            st.session_state.waiting_for_perp = True
                            st.rerun()

                        st.session_state.trigger_perp_eval = False
                        perp_configs = [
                            ("unflipped_90", "False", "90"),
                            ("unflipped_270", "False", "270"),
                            ("flipped_90", "True", "90"),
                            ("flipped_270", "True", "270")
                        ]
                        slices_p = {}
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        logger.info(
                            f"Evaluation started: {model_type}",
                            extra={"user": st.session_state.user_name, "comp": "HORIZ_ROT"},
                        )
                        with store["eval_lock"]:
                            for idx, (s_name, f_v, r_v) in enumerate(perp_configs):
                                status_text.info(f"Processing evaluation slice [{idx + 1}/{len(perp_configs)}]: Flip={f_v}, Rotate={r_v}°")
                                slices_p[s_name] = run_evaluation_process(saved_model_path, model_type, apply_preprocess, f_v, r_v)
                                progress_bar.progress((idx + 1) / len(perp_configs))

                        status_text.empty()
                        progress_bar.empty()
                        st.session_state.deep_perp_data = slices_p
                        logger.info(
                            f"Completed.",
                            extra={"user": st.session_state.user_name, "comp": "HORIZ_ROT"}
                        )
                        st.rerun()
                else:
                    perp_slices = st.session_state.deep_perp_data
                    rot_p90_true, rot_p90_pred = [], []
                    for run in [perp_slices["unflipped_90"], perp_slices["flipped_90"]]:
                        for p in run["predictions"].values():
                            rot_p90_true.append(p["y_true"])
                            rot_p90_pred.append(p["y_pred"])
                    rot_p90_acc = np.mean(np.array(rot_p90_true) == np.array(rot_p90_pred)) if rot_p90_true else 0.0

                    rot_n90_true, rot_n90_pred = [], []
                    for run in [perp_slices["unflipped_270"], perp_slices["flipped_270"]]:
                        for p in run["predictions"].values():
                            rot_n90_true.append(p["y_true"])
                            rot_n90_pred.append(p["y_pred"])
                    rot_n90_acc = np.mean(np.array(rot_n90_true) == np.array(rot_n90_pred)) if rot_n90_true else 0.0

                    rot_90_270_true = rot_p90_true + rot_n90_true
                    rot_90_270_pred = rot_p90_pred + rot_n90_pred
                    rot_90_270_acc = np.mean(np.array(rot_90_270_true) == np.array(rot_90_270_pred)) if rot_90_270_true else 0.0

                    render_html_metric_banner(rot_90_270_acc, acc, "Combined Accuracy")

                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        render_matrix_and_metric(
                            rot_p90_true, rot_p90_pred, "Clockwise (+90°)", rot_p90_acc, acc
                        )
                    with col_r2:
                        render_matrix_and_metric(
                            rot_n90_true, rot_n90_pred, "Counter-Clockwise (-90°)", rot_n90_acc, acc
                        )

                # --- STEP 3: UPSIDE-DOWN HANDS ---
                st.write("")
                st.markdown("### 🙃 Upside-down Hands")
                if st.session_state.get("deep_inversion_data") is None:
                    # ---- INVERSION NON-BLOCKING POLLING AUTOMATION VALVE ----
                    if st.session_state.get("waiting_for_inversion", False):
                        if store["eval_lock"].locked():
                            st.warning("⏳ Server busy: Another user is running evaluations. Retrying automatically...")
                            time.sleep(20)
                            st.rerun()
                        else:
                            st.session_state.waiting_for_inversion = False
                            st.session_state.trigger_inversion_eval = True
                            st.rerun()

                    if st.button("Evaluate Inversion Robustness") or st.session_state.get("trigger_inversion_eval", False):
                        if store["eval_lock"].locked() and not st.session_state.get("trigger_inversion_eval", False):
                            logger.warning(
                                f"User is waitlisted.",
                                extra={"user": st.session_state.user_name, "comp": "INVERSION"}
                            )
                            st.session_state.waiting_for_inversion = True
                            st.rerun()

                        st.session_state.trigger_inversion_eval = False
                        inv_configs = [
                            ("unflipped_180", "False", "180"),
                            ("flipped_180", "True", "180")
                        ]
                        slices_i = {}
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        logger.info(
                            f"Evaluation started: {model_type}",
                            extra={"user": st.session_state.user_name, "comp": "INVERSION"},
                        )
                        with store["eval_lock"]:
                            for idx, (s_name, f_v, r_v) in enumerate(inv_configs):
                                status_text.info(f"Processing evaluation slice [{idx + 1}/{len(inv_configs)}]: Flip={f_v}, Rotate={r_v}°")
                                slices_i[s_name] = run_evaluation_process(saved_model_path, model_type, apply_preprocess, f_v, r_v)
                                progress_bar.progress((idx + 1) / len(inv_configs))

                        status_text.empty()
                        progress_bar.empty()
                        st.session_state.deep_inversion_data = slices_i
                        logger.info(
                            f"Completed.",
                            extra={"user": st.session_state.user_name, "comp": "INVERSION"},
                        )
                        st.rerun()
                else:
                    inv_slices = st.session_state.deep_inversion_data
                    rot_180_true, rot_180_pred = [], []
                    for run in [inv_slices["unflipped_180"], inv_slices["flipped_180"]]:
                        for p in run["predictions"].values():
                            rot_180_true.append(p["y_true"])
                            rot_180_pred.append(p["y_pred"])
                    rot_180_acc = np.mean(np.array(rot_180_true) == np.array(rot_180_pred)) if rot_180_true else 0.0

                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        render_matrix_and_metric(
                            rot_180_true, rot_180_pred, "Upside-Down Accuracy", rot_180_acc, acc
                        )

                # --- STEP 4: DISTANCE & FRAMING INVARIANCE (ZOOM) ---
                st.write("")
                st.markdown("### 📏 Distance & Framing Invariance (Scale/Zoom)")
                if st.session_state.get("deep_zoom_data") is None:
                    # ---- ZOOM NON-BLOCKING POLLING AUTOMATION VALVE ----
                    if st.session_state.get("waiting_for_zoom", False):
                        if store["eval_lock"].locked():
                            st.warning("⏳ Server busy: Another user is running evaluations. Retrying automatically...")
                            time.sleep(20)
                            st.rerun()
                        else:
                            st.session_state.waiting_for_zoom = False
                            st.session_state.trigger_zoom_eval = True
                            st.rerun()

                    if st.button("Evaluate Scale Robustness") or st.session_state.get("trigger_zoom_eval", False):
                        if store["eval_lock"].locked() and not st.session_state.get("trigger_zoom_eval", False):
                            logger.warning(
                                f"User is waitlisted.",
                                extra={"user": st.session_state.user_name, "comp": "ZOOM"},
                            )
                            st.session_state.waiting_for_zoom = True
                            st.rerun()

                        st.session_state.trigger_zoom_eval = False
                        zoom_configs = [
                            ("zoomed_in", "in"),
                            ("zoomed_out", "out")
                        ]
                        slices_z = {}
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        logger.info(
                            f"Evaluation started: {model_type}",
                            extra={"user": st.session_state.user_name, "comp": "ZOOM"},
                        )
                        with store["eval_lock"]:
                            for idx, (s_name, zoom_flag) in enumerate(zoom_configs):
                                status_text.info(f"Processing scaling evaluation matrix [{idx + 1}/{len(zoom_configs)}]: Zoom Mode='{zoom_flag}'")
                                slices_z[s_name] = run_evaluation_process(saved_model_path, model_type, apply_preprocess, "False", "0", zoom_flag)
                                progress_bar.progress((idx + 1) / len(zoom_configs))

                        status_text.empty()
                        progress_bar.empty()
                        st.session_state.deep_zoom_data = slices_z
                        logger.info(
                            f"Completed.",
                            extra={"user": st.session_state.user_name, "comp": "ZOOM"},
                        )
                        st.rerun()
                else:
                    zoom_slices = st.session_state.deep_zoom_data
                    zin_true, zin_pred = [], []
                    for p in zoom_slices["zoomed_in"]["predictions"].values():
                        zin_true.append(p["y_true"])
                        zin_pred.append(p["y_pred"])
                    zin_acc = np.mean(np.array(zin_true) == np.array(zin_pred)) if zin_true else 0.0

                    zout_true, zout_pred = [], []
                    for p in zoom_slices["zoomed_out"]["predictions"].values():
                        zout_true.append(p["y_true"])
                        zout_pred.append(p["y_pred"])
                    zout_acc = np.mean(np.array(zout_true) == np.array(zout_pred)) if zout_true else 0.0

                    total_zoom_true = zin_true + zout_true
                    total_zoom_pred = zin_pred + zout_pred
                    total_zoom_acc = np.mean(np.array(total_zoom_true) == np.array(total_zoom_pred)) if total_zoom_true else 0.0

                    render_html_metric_banner(total_zoom_acc, acc, "Combined Framing Invariance Accuracy")

                    col_z1, col_z2 = st.columns(2)
                    with col_z1:
                        render_matrix_and_metric(
                            zin_true, zin_pred, "Zoomed-In (Close Framing)", zin_acc, acc
                        )
                    with col_z2:
                        render_matrix_and_metric(
                            zout_true, zout_pred, "Zoomed-Out (Distant Framing)", zout_acc, acc
                        )


if __name__ == "__main__":
    main()
