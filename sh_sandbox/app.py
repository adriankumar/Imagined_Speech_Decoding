import webview
import threading
import functools
import http.server
import socketserver
from pathlib import Path

#absolute path to the web folder, used as the static server's document root
_WEB_DIR = (Path(__file__).parent / "web").resolve()

#start a background static file server rooted at the web folder, return its port
def _serve_web():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(_WEB_DIR))
    #port 0 lets the os pick a free port, avoiding collisions
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    #daemon thread so the server dies with the app
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return port

#open the pywebview window onto the locally served sandbox page
def launch(width=1100, height=820):
    port = _serve_web()
    webview.create_window(title="spherical harmonics sandbox",
                          url=f"http://127.0.0.1:{port}/index.html",
                          width=width, height=height)
    webview.start(debug=False) #change to true if you want to see console log outputs