"""Research Assistant. Launch with run.bat / run.sh, or: python -m streamlit run app.py"""
import streamlit as st

import core
from config import settings

st.set_page_config(page_title="Research Assistant", page_icon="📚", layout="wide")
core.setup_page()

library = core.get_library()
core.get_embedder()

# Guard against mixing vectors from different embedding models.
stored_model = library.get_meta("embed_model")
if stored_model != settings.embed_model:
    if stored_model is not None and library.list_documents():
        st.error(
            f"Your library was indexed with '{stored_model}' but EMBED_MODEL is "
            f"'{settings.embed_model}'. Restore the original setting, or delete the "
            "data folder to re-index from scratch."
        )
        st.stop()
    library.set_meta("embed_model", settings.embed_model)

@st.cache_resource
def _startup() -> bool:
    library.interrupt_stale_jobs()  # threads don't survive a restart
    return True


_startup()

pages = st.navigation({
    "Research": [
        st.Page("views/research.py", title="Research agent", icon="🔬", default=True),
        st.Page("views/chat.py", title="Chat", icon="💬"),
        st.Page("views/reader.py", title="Reader", icon="📖"),
        st.Page("views/tables.py", title="Tables", icon="📊"),
        st.Page("views/notes.py", title="Notes", icon="📝"),
    ],
    "Papers": [
        st.Page("views/library.py", title="Library", icon="📚"),
        st.Page("views/discover.py", title="Discover papers", icon="🔎"),
        st.Page("views/settings.py", title="Settings", icon="⚙️"),
    ],
})

# Page-specific sidebar content goes in this slot, above the shared model settings.
st.session_state["sidebar_slot"] = st.sidebar.container()
core.model_sidebar()
pages.run()
