import gc
import re
import tempfile
import time
from pathlib import Path
import json
import subprocess
import sys

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

# ==== CONFIGURATION & CONSTANTS ====
TEST_IMAGE_DIR = "test_images"
CLASS_NAMES = ["A", "B", "C"]
REQUIRED_COLUMNS = ["participant", "accuracy", "submission_time", "batch", "model_type"]


help_leaderboard_toggle = """By default, the leaderboard displays one entry per participant and model type.\n\n
Toggle if you prefer to see only one entry per participant.
"""

help_model_selection = """Please select your used model.\n\n
Select the model family in case you used transfer learning or select "Custom" otherwise.\n\n
The app uses this information to handle preprocessing as well as for leaderboard purposes.
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

# ==== GLOBAL STORE & STATE ====


@st.cache_resource
def get_global_store() -> dict:
    """Initializes the global in-memory store for the application session."""
    return {
        "submissions": {},
        "alltime_submissions": None,
        "leaderboards": {},
        "alltime_leaderboard": None,
        "batches": None,
        "batches_last_updated": None,
        "gsheet_conn": None,
        "configured_batches": set(),
        "is_evaluating": False,
        "eval_start_time": None,
    }


def state_inits() -> None:
    """Initializes session state variables and the initial GSheet connection."""
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
    """Aggregates data from all batches (EXCEPT anonymous) for the global leaderboard."""
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


# ==== CACHED HELPER FUNCTIONS ====


@st.cache_resource
def get_gsheet_connection() -> GSheetsConnection:
    """Returns the Streamlit GSheets connection object."""
    return st.connection("gsheets", type=GSheetsConnection)


@st.cache_resource
def configure_gsheet(_store: dict, batch: str | None = None) -> str:
    """Configures the GSheet connection and ensures specific worksheets exist."""
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
    """Provides UI for instructors to clear the global cache."""
    st.divider()
    st.subheader("🛠️ Admin Settings", anchor=False)
    if st.button("Clear cached resources"):
        get_global_store.clear()
        configure_gsheet.clear()
        get_gsheet_connection.clear()
        st.success("Cache cleared successfully! Refreshing app...")
        st.rerun()


# ==== GSHEETS UTILS ====


def _open_spreadsheet() -> gspread.Spreadsheet:
    """Opens the raw gspread client for structural changes."""
    creds_dict = dict(st.secrets["connections"]["gsheets"])
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    return client.open_by_url(creds_dict["spreadsheet"])


def ensure_batch_sheet_exists(batch: str, conn: GSheetsConnection) -> None:
    """Checks if a worksheet exists for the batch; creates it if not."""
    try:
        conn.read(worksheet=batch, ttl=0)
    except WorksheetNotFound:
        sh = _open_spreadsheet()
        sh.add_worksheet(title=batch, rows="1000", cols="10")
        empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        conn.update(worksheet=batch, data=empty_df)
    except Exception as e:
        st.error(f"Failed to verify/create worksheet: {e}")


# ==== LEADERBOARD LOGIC ====


def generate_leaderboard_dataframe(
    submissions_df: pd.DataFrame,
    *,
    reduce_leaderboard: bool,
) -> pd.DataFrame:
    """Processes submission data into a leaderboard format."""
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
        .assign(position=lambda df_: range(1, len(df_) + 1))
        .set_index("position")
        .filter(["participant", "batch", "model_type", "accuracy", "attempts"])
    )


def build_leaderboards() -> None:
    """Rebuilds the processed leaderboards in the store."""
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
    """Writes a new result to GSheets and updates memory."""
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


# ==== UI COMPONENTS ====


def get_participant_info() -> None:
    """Handles Login and Batch authentication."""
    store = get_global_store()

    if st.session_state.user_name and st.session_state.batch:
        # Check if we need to load the submissions for chart plotting/recording
        if st.session_state.batch not in store["submissions"]:
            try:
                configure_gsheet(_store=store, batch = st.session_state.batch)
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
    """Plot submission accuracy for a participant over time."""
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
        participant_submissions = participant_submissions.sort_values(
            "submission_time",
        )  # .set_index("submission_time")
        line = (
            alt.Chart(participant_submissions)
            .mark_line()
            .encode(
                x="submission_time:T",
                y="accuracy:Q",
            )
        )

        # Large colored points on top of the line
        points = (
            alt.Chart(participant_submissions)
            .mark_point(filled=True, size=150)  # size controls how big the dots are
            .encode(
                x="submission_time:T",
                y="accuracy:Q",
                color="model_type:N",
                tooltip=["submission_time:T", "model_type:N", "accuracy:Q"],
            )
        )

        # Layer line + points
        chart = alt.layer(line, points).interactive()

        st.altair_chart(chart, width="stretch")
    elif len(participant_submissions):
        st.success(
            "First submission recorded! Submit more models to see your progress chart.",
        )


@st.fragment(run_every=10)
def show_leaderboard() -> None:
    """Displays the interactive leaderboard with toggle logic."""
    if st.session_state.batch == "anonymous":
        st.info(
            "You are currently in an anonymous session. You won't see or appear on any public leaderboards.",
        )
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
        st.dataframe(
            view.drop("batch", axis=1, errors="ignore"),
            width="stretch",
        )
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


# ==== MAIN ====


def main() -> None:
    st.set_page_config(page_title="Sign Language Showdown", page_icon="✊")

    if st.session_state.get("batch") == "Instructor":
        display_admin()

    st.title("Sign Language Model Showdown!", anchor=False, text_alignment="center")
    st.title("✊🖐️🤏", anchor=False, text_alignment="center")

    state_inits()
    get_participant_info()

    if st.session_state.user_name and st.session_state.batch:
        st.subheader("📤 Submit Your Model", anchor=False)
        cols = st.columns(
                2,
                gap="large",
            )
        with cols[0]:
            model_type = st.selectbox(
                "Select the exact model used:",
                options=ALLOWED_MODELS, # Now shows specific models
                index=None,
                help="Note: Large models (ConvNeXt, EfficientNetL, etc.) are disabled for stability."
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
                store = get_global_store()
                now = time.time()
                TIMEOUT_SECONDS = 300
                waiting_placeholder = st.empty()
                while store["is_evaluating"]:
                    start_time = store.get("eval_start_time")
                    if start_time and (now - start_time) > TIMEOUT_SECONDS:
                        waiting_placeholder.info("⚠️ Previous evaluation timed out or crashed. Recovering...")
                        break
                    waiting_placeholder.warning(
                        "⏳ Another user is currently evaluating a model. "
                        "Please wait, your evaluation will start automatically when the server is free..."
                    )
                    time.sleep(5)
                    now = time.time()
                store["is_evaluating"] = True
                store["eval_start_time"] = time.time()
                waiting_placeholder.empty()
                try:
                    with st.spinner("Analyzing model performance..."):  # noqa: SIM117
                        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmpf:
                            tmpf.write(uploaded_file.getbuffer())
                            model_path = tmpf.name

                        uploaded_file = None
                        gc.collect()

                        try:

                            results_path = tempfile.NamedTemporaryFile(
                                suffix=".json",
                                delete=False,
                            ).name

                            progress_path = results_path.replace(".json", "_progress.json")

                            process = subprocess.Popen(
                                [
                                    sys.executable,
                                    "evaluator.py",
                                    model_path,
                                    model_type,
                                    str(apply_preprocess),
                                    results_path,
                                    progress_path,
                                ],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                            )

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            status_text.info("Loading model...")

                            while process.poll() is None:

                                if Path(progress_path).exists():
                                    try:
                                        try:
                                            with open(progress_path, "r") as f:
                                                progress = json.load(f)
                                        except (json.JSONDecodeError, FileNotFoundError):
                                            continue

                                        progress_bar.progress(progress["progress"])

                                        status_text.write(
                                            f"Image {progress['done']}/{progress['total']} "
                                            f"| Current accuracy: {progress['accuracy']:.2%}"
                                        )

                                    except Exception:
                                        pass

                                time.sleep(0.5)

                            stdout, stderr = process.communicate()
                            progress_bar.progress(1.0)
                            status_text.success("Evaluation complete!")

                            if process.returncode != 0:
                                st.code(stderr)
                                raise RuntimeError("Evaluation failed")

                            with open(results_path) as f:
                                results = json.load(f)

                            acc = results["accuracy"]

                            y_pred = np.array(results["y_pred"])
                            y_test = np.array(results["y_true"])

                            result = pd.DataFrame(
                                [
                                    {
                                        "accuracy": round(acc, 4),
                                        "participant": st.session_state.user_name,
                                        "batch": st.session_state.batch,
                                        "submission_time": pd.Timestamp.now().isoformat(),
                                        "model_type": model_type,
                                    },
                                ],
                            )

                            update_submissions(result)
                            st.success(f"Success! Model Accuracy: {acc:.2%}")
                            st.subheader("🧮 Confusion Matrix")
                            fig, ax = plt.subplots(
                                figsize=(2, 2), facecolor="black",
                            )
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
                            ax.set_xlabel("Predicted", color="white")
                            ax.set_ylabel("True Label", color="white")
                            ax.tick_params(colors="white", labelsize=8)
                            ax.tick_params(
                                which="both",
                                length=0,
                            )
                            st.pyplot(fig, width="content")
                            plt.close(fig)

                            gc.collect()
                            Path(model_path).unlink(missing_ok=True)
                            Path(results_path).unlink(missing_ok=True)
                        except Exception as e:
                            st.error(f"Error evaluating model: {e}")
                            gc.collect()
                except:
                    pass
                finally:
                    store["is_evaluating"] = False
                    store["eval_start_time"] = None
                    gc.collect()
                    if 'model_path' in locals():
                        Path(model_path).unlink(missing_ok=True)
                    if 'results_path' in locals():
                        Path(results_path).unlink(missing_ok=True)
                    if 'progress_path' in locals():
                        Path(progress_path).unlink(missing_ok=True)

        plot_submissions(st.session_state.user_name)
        show_leaderboard()


if __name__ == "__main__":
    main()
