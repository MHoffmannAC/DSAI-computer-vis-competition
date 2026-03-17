import tempfile
from pathlib import Path

import gspread
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from google.oauth2.service_account import Credentials
from gspread.exceptions import WorksheetNotFound
from PIL import Image
from streamlit_gsheets import GSheetsConnection

# ==== CONFIGURATION & CONSTANTS ====
TEST_IMAGE_DIR = "test_images"
CLASS_NAMES = ["A", "B", "C"]
REQUIRED_COLUMNS = ["participant", "accuracy", "submission_time", "batch"]

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
        "configured_batches": set(),  # Track which batches have been verified/created
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
def configure_gsheet(batch: str | None = None, _store: dict | None = None) -> str:
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
    submissions_df: pd.DataFrame, *, show_best_only: bool = True,
) -> pd.DataFrame:
    """Processes submission data into a leaderboard format."""
    if submissions_df.empty:
        return pd.DataFrame()

    if show_best_only:
        df = (
            submissions_df.assign(
                attempts=lambda df_: df_.groupby("participant")[
                    "participant"
                ].transform("count"),
            )
            .sort_values(["accuracy", "submission_time"], ascending=[False, True])
            .drop_duplicates(subset=["participant"], keep="first")
            .assign(position=lambda df_: range(1, len(df_) + 1))
            .set_index("position")
            .filter(["participant", "accuracy", "attempts", "batch"])
        )
    else:
        df = (
            submissions_df.sort_values("submission_time", ascending=False)
            .assign(position=lambda df_: range(1, len(df_) + 1))
            .set_index("position")
            .filter(["participant", "accuracy", "submission_time", "batch"])
        )
    return df


def build_leaderboards() -> None:
    """Rebuilds the processed leaderboards in the store."""
    store = get_global_store()
    for batch, df in store["submissions"].items():
        if batch != "anonymous" and df is not None and not df.empty:
            store["leaderboards"][batch] = generate_leaderboard_dataframe(df)
        else:
            store["leaderboards"][batch] = pd.DataFrame()

    if (
        store["alltime_submissions"] is not None
        and not store["alltime_submissions"].empty
    ):
        store["alltime_leaderboard"] = generate_leaderboard_dataframe(
            store["alltime_submissions"],
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


# ==== MODEL EVALUATION ====


@st.cache_data(show_spinner="Loading test set...")
def load_raw_test_images() -> tuple[list[Image.Image], np.ndarray]:
    images, labels = [], []
    base_dir = Path(TEST_IMAGE_DIR)

    for idx, cls in enumerate(CLASS_NAMES):
        folder = base_dir / cls
        if not folder.exists():
            continue
        for fpath in sorted(folder.iterdir()):
            if not fpath.is_file():
                continue

            img = Image.open(fpath).convert("RGB")
            images.append(img)
            labels.append(idx)

    return images, np.array(labels)


def evaluate_model(model, pil_images, y, input_size) -> float:
    resized_imgs = [np.array(img.resize(input_size)) for img in pil_images]
    x = np.stack(resized_imgs)
    preds = model.predict(x)
    y_pred = np.argmax(preds, axis=1)
    return (y_pred == y).mean()


# ==== UI COMPONENTS ====


def get_participant_info() -> None:
    """Handles Login and Batch authentication."""
    store = get_global_store()

    if st.session_state.user_name and st.session_state.batch:
        # Check if we need to load the submissions for chart plotting/recording
        if st.session_state.batch not in store["submissions"]:
            try:
                configure_gsheet(st.session_state.batch, _store=store)
                store["submissions"][st.session_state.batch] = store[
                    "gsheet_conn"
                ].read(worksheet=st.session_state.batch, ttl=0)
                build_leaderboards()
            except Exception as e:
                st.error(f"Error loading batch data: {e}")
                st.stop()

        st.info(
            f"Logged in as: **{st.session_state.user_name}** | Batch: **{st.session_state.batch}**",
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
            elif st.button("Log In"):
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
        .filter(["submission_time", "accuracy"])
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
        ).set_index("submission_time")
        st.line_chart(participant_submissions)
    elif len(participant_submissions):
        st.success(
            "First submission recorded! Submit more models to see your progress chart.",
        )


@st.fragment(run_every=10)
def show_leaderboard() -> None:
    """Displays the interactive leaderboard with toggle logic."""
    if st.session_state.batch == "anonymous":
        st.info(
            "You are currently in an anonymous session. Your results are being recorded for instructors, but you won't see or appear on any public leaderboards.",
        )
        return

    store = get_global_store()
    batch = st.session_state.batch

    st.divider()
    st.header(f"🏆 {batch} Leaderboard", anchor=False)

    submissions_df = store["submissions"].get(batch, pd.DataFrame())

    if not submissions_df.empty:
        show_best = st.toggle(
            "Only show best attempt per user",
            value=True,
            key="batch_toggle",
        )
        view = generate_leaderboard_dataframe(submissions_df, show_best_only=show_best)
        st.dataframe(
            view.drop("batch", axis=1, errors="ignore"),
            use_container_width=True,
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
        at_show_best = st.toggle(
            "Only show best attempt per user",
            value=True,
            key="at_toggle",
        )
        at_view = generate_leaderboard_dataframe(
            store["alltime_submissions"],
            show_best_only=at_show_best,
        )
        st.dataframe(at_view, use_container_width=True)


# ==== MAIN ====


def main() -> None:
    st.set_page_config(page_title="Sign Language Showdown", page_icon="✊")
    st.title("Sign Language Model Showdown! ✊🖐️🤏", anchor=False)

    state_inits()
    get_participant_info()

    if st.session_state.user_name and st.session_state.batch:
        st.subheader("📤 Submit Your Model", anchor=False)
        uploaded_file = st.file_uploader("Select a Keras model file", type=["keras"])

        if uploaded_file:  # noqa: SIM102
            if st.button("Evaluate Model", type="primary"):
                raw_images, y_test = load_raw_test_images()
                with st.spinner("Analyzing model performance..."):
                    with tempfile.NamedTemporaryFile(
                        suffix=".keras",
                        delete=True,
                    ) as tmpf:
                        tmpf.write(uploaded_file.read())
                        tmpf.flush()
                        try:
                            model = tf.keras.models.load_model(tmpf.name)
                            input_shape = model.input_shape
                            if len(input_shape) == 4 and input_shape[-1] == 3:
                                input_size = (input_shape[1], input_shape[2])
                                acc = evaluate_model(
                                    model,
                                    raw_images,
                                    y_test,
                                    input_size,
                                )

                                result = pd.DataFrame(
                                    [
                                        {
                                            "accuracy": round(acc, 4),
                                            "participant": st.session_state.user_name,
                                            "batch": st.session_state.batch,
                                            "submission_time": pd.Timestamp.now().isoformat(),
                                        },
                                    ],
                                )

                                # Recording now happens for everyone, including anonymous
                                update_submissions(result)
                                st.success(f"Success! Model Accuracy: {acc:.2%}")
                            else:
                                st.error("Incompatible model shape.")
                        except Exception as e:
                            st.error(f"Error evaluating model: {e}")

        plot_submissions(st.session_state.user_name)
        show_leaderboard()

        if st.session_state.batch == "Instructor":
            display_admin()


if __name__ == "__main__":
    main()
