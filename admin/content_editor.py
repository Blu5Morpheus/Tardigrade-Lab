"""Tab 2 — content editor: edit Astro `src/content/*` markdown via GitHub API."""

from __future__ import annotations

import streamlit as st

from lib.cache import clear_all_data_caches
from lib.github_client import committer_dict, github_configured

COLLECTIONS = {
    "demos": "src/content/demos",
    "papers": "src/content/papers",
    "products": "src/content/products",
    "preprints": "src/content/preprints",
}


def render() -> None:
    st.subheader("Content editor")
    st.caption(
        "Commits land on the default branch. Render will redeploy the static site automatically."
    )

    if not github_configured():
        st.info(
            "GitHub not yet configured. Add `GITHUB_TOKEN` and `GITHUB_REPO` to Render's "
            "environment variables (or `secrets.toml` locally). The token must be a fine-grained "
            "PAT scoped to the tardigrade-site repo, contents: read+write."
        )
        return

    from lib.github_client import get_repo
    try:
        repo = get_repo()
    except Exception as e:
        st.error(f"Could not open repo: {e}")
        return

    collection = st.radio("Collection", list(COLLECTIONS), horizontal=True)
    path = COLLECTIONS[collection]

    try:
        contents = repo.get_contents(path)
    except Exception as e:
        st.error(f"Could not list `{path}`: {e}")
        return

    files = [c for c in contents if getattr(c, "name", "").endswith(".md")]
    file_names = [f.name for f in files]
    selected = st.selectbox("File", ["+ new file", *file_names])

    if selected == "+ new file":
        new_name = st.text_input("Filename (e.g., new-demo.md)")
        body = st.text_area(
            "Content (markdown with YAML frontmatter)",
            value="---\ntitle: \nslug: \n---\n\n",
            height=400,
        )
        if st.button("Create file", type="primary", disabled=not new_name):
            full = f"{path}/{new_name}"
            try:
                repo.create_file(
                    full,
                    f"admin: create {full}",
                    body,
                    committer=committer_dict(),
                )
                st.success(f"Created `{full}`. Site will rebuild shortly.")
                clear_all_data_caches()
            except Exception as e:
                st.error(f"Create failed: {e}")
        return

    f = next(f for f in files if f.name == selected)
    current = f.decoded_content.decode("utf-8")
    edited = st.text_area("Content", value=current, height=500, key=f"edit-{f.path}")

    diff_dirty = edited != current
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Save", type="primary", disabled=not diff_dirty):
            try:
                repo.update_file(
                    f.path, f"admin: edit {f.path}", edited, f.sha,
                    committer=committer_dict(),
                )
                st.success("Saved. Site will rebuild shortly.")
                clear_all_data_caches()
            except Exception as e:
                st.error(f"Save failed: {e}")
    with col2:
        if st.button("Reset", disabled=not diff_dirty):
            st.rerun()
    with col3:
        confirm_key = f"confirm_delete_{f.path}"
        if st.button("Delete"):
            if st.session_state.get(confirm_key):
                try:
                    repo.delete_file(
                        f.path, f"admin: delete {f.path}", f.sha,
                        committer=committer_dict(),
                    )
                    st.success(f"Deleted `{f.path}`.")
                    st.session_state.pop(confirm_key, None)
                    clear_all_data_caches()
                except Exception as e:
                    st.error(f"Delete failed: {e}")
            else:
                st.session_state[confirm_key] = True
                st.warning("Click **Delete** again to confirm.")

    if diff_dirty:
        with st.expander("Diff preview"):
            import difflib
            diff = difflib.unified_diff(
                current.splitlines(keepends=True),
                edited.splitlines(keepends=True),
                fromfile="current",
                tofile="edited",
                n=3,
            )
            st.code("".join(diff) or "(identical)", language="diff")
