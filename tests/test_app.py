import importlib.util
import sys
import types


def test_streamlit_app_warns_on_empty_text(monkeypatch):
    calls = {"warnings": [], "button": True}
    st = types.ModuleType("streamlit")
    st.set_page_config = lambda **_kwargs: None
    st.title = lambda *_args, **_kwargs: None
    st.write = lambda *_args, **_kwargs: None
    st.text_area = lambda *_args, **_kwargs: "   "
    st.button = lambda *_args, **_kwargs: calls["button"]
    st.warning = lambda message: calls["warnings"].append(message)
    st.caption = lambda *_args, **_kwargs: None
    st.markdown = lambda *_args, **_kwargs: None
    st.expander = lambda *_args, **_kwargs: __import__("contextlib").nullcontext()
    st.columns = lambda count: [types.SimpleNamespace(metric=lambda *_args, **_kwargs: None) for _ in range(count)]
    st.subheader = lambda *_args, **_kwargs: None
    st.error = lambda *_args, **_kwargs: None
    st.divider = lambda: None
    st.cache_resource = lambda **_kwargs: (lambda function: function)
    monkeypatch.setitem(sys.modules, "streamlit", st)

    spec = importlib.util.spec_from_file_location("streamlit_app_under_test", "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert calls["warnings"] == ["Enter a movie review before analyzing sentiment."]
