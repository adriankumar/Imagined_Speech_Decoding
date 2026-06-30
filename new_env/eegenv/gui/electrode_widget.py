import threading
import webview

#module-level reference to the popup window, none when closed
_popup = None
_popup_lock = threading.Lock()

#open the electrode selector as a standalone pywebview popup window
#host and port must match the running flask app so the popup can reach the same routes
#action_context is a string passed to the popup so it knows which buttons to show
#e.g. "exclude", "reference", or "spread"
def open_selector(host="127.0.0.1", port=5000, action_context="exclude", title="select electrodes"):
    global _popup
    with _popup_lock:
        if _popup is not None:
            #bring the existing popup to focus rather than opening a second one
            try:
                _popup.show()
            except Exception:
                pass
            return

        url = f"http://{host}:{port}/electrode_widget?context={action_context}"
        _popup = webview.create_window(
            title, url,
            width=520, height=560,
            resizable=True,
            on_top=True,
        )
        #clear the module reference when the popup is closed
        _popup.events.closed += _on_closed

#called when the popup window is closed, either by the user or by confirm_and_close
def _on_closed():
    global _popup
    with _popup_lock:
        _popup = None

#close the popup from Python, called after an action is committed to the session
def close_selector():
    global _popup
    with _popup_lock:
        if _popup is not None:
            _popup.destroy()
            _popup = None