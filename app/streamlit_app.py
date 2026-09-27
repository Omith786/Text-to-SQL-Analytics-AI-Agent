"""Streamlit chat interface for the Text-to-SQL agent.

Run with: ``streamlit run app/streamlit_app.py``

The UI is a thin layer over :class:`text2sql.agent.Agent`: it keeps a chat
history, sends each question to the agent, and renders the generated SQL, the
result table and (where the shape allows) a bar chart. Connection settings come
from the same environment variables the CLI uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text2sql.agent import Agent  # noqa: E402
from text2sql.chart import is_chartable  # noqa: E402
from text2sql.config import DEFAULT_DB_PATH, AgentConfig, LLMConfig  # noqa: E402
from text2sql.db import Database  # noqa: E402
from text2sql.llm import OpenAIChatClient  # noqa: E402
from text2sql.prompts import STRATEGIES  # noqa: E402
from text2sql.sample_data import ensure  # noqa: E402


@st.cache_resource
def get_agent(model: str, strategy_unused: str) -> Agent:
    """Create (and cache) an agent bound to the bundled database.

    Cached on the model name so switching models rebuilds the client while the
    read-only database connection is reused across reruns.
    """
    ensure()
    db = Database(DEFAULT_DB_PATH)
    config = LLMConfig.from_env()
    if model:
        config = LLMConfig(**{**config.__dict__, "model": model})
    return Agent(db, OpenAIChatClient(config), AgentConfig())


def main() -> None:
    st.set_page_config(page_title="Text-to-SQL Analytics Agent", layout="wide")
    st.title("Text-to-SQL Analytics Agent")
    st.caption("Ask analytical questions in plain English over a bundled shipping and energy database.")

    with st.sidebar:
        st.header("Settings")
        default_model = LLMConfig.from_env().model
        model = st.text_input("Model", value=default_model,
                              help="Any model available on your OpenAI-compatible endpoint.")
        strategy = st.selectbox("Prompting strategy", STRATEGIES, index=len(STRATEGIES) - 1)
        self_correction = st.checkbox("Self-correction (retry on error)", value=True)
        st.markdown("---")
        with st.expander("Database schema"):
            st.code(get_agent(model, strategy).db.schema_text(), language="text")

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("e.g. Which port received the most calls?")
    if not question:
        return

    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    agent = get_agent(model, strategy)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.answer(question, strategy=strategy, self_correction=self_correction)

        if result.sql:
            st.code(result.sql, language="sql")

        if result.success and result.result is not None:
            df = pd.DataFrame(result.result.rows, columns=result.result.columns)
            st.dataframe(df, use_container_width=True)
            if is_chartable(result.result):
                st.bar_chart(df.set_index(df.columns[0]))
            summary = (f"{result.result.row_count} row(s), {result.retries} retries, "
                       f"{result.total_tokens} tokens, {result.latency_s:.2f}s")
            st.caption(summary)
            st.session_state.history.append({"role": "assistant", "content": summary})
        else:
            last_error = result.attempts[-1].error if result.attempts else "no attempts"
            msg = f"Could not answer that. Last error: {last_error}"
            st.error(msg)
            st.session_state.history.append({"role": "assistant", "content": msg})


if __name__ == "__main__":
    main()
