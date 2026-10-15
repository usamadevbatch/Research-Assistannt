import streamlit as st

import settings_store
import sources

st.markdown('<div class="ra-hero"><h1>⚙️ Settings</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="ra-muted">Optional API keys for more databases. Keys are stored only on this computer '
            '(data/settings.json) and are sent only to the service they belong to.</p>', unsafe_allow_html=True)

values = settings_store.load()
with st.form("keys"):
    new = {}
    for key, (name, help_text, url) in settings_store.KEYS.items():
        label = f"{name}" + (f" · [get a key]({url})" if url else "")
        st.markdown(f"**{label}**")
        new[key] = st.text_input(name, value=values.get(key, ""), type="password", help=help_text,
                                 label_visibility="collapsed", placeholder=help_text)
    if st.form_submit_button("Save keys", type="primary"):
        settings_store.save({**values, **{k: v.strip() for k, v in new.items()}})
        st.success("Saved.")

st.markdown("#### Available sources")
for s in sources.SOURCES:
    ok = sources.available(s)
    st.markdown(f"{'🟢' if ok else '⚪'} **{s.label}** · {s.group}" + ("" if ok else " · needs a key above"))
st.info("Paywalled PDFs can't be downloaded automatically. For those papers, the Research page gives you a link "
        "to download the PDF through your university access, then you can attach it to the paper.")
