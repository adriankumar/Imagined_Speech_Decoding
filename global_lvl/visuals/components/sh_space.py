import numpy as np 
from ...constants import (DEFAULT_CMAP, DELTA_CMAP)

#returns figure
def basis_matrix_fig(Y, subtitle=None):
    import matplotlib.pyplot as plt  #lazy import

    a = np.abs(Y).max()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    m = ax.imshow(Y, origin="upper", cmap=DELTA_CMAP, vmin=-a, vmax=a, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    rows, cols = Y.shape

    ax.set_title(subtitle or f"Y  ({rows} modes x {cols} electrodes)", fontsize=10)
    fig.tight_layout()
    return fig



#------------
#interactive basis sphere; one mode at a time, sampled at the montage's electrodes
#------------
_WEB_DIR = None #resolved lazily so the module imports without pathlib work at import time
 
#unit-sphere xyz from spherical angles, three.js is y-up so the polar axis maps to y
def _to_xyz(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.cos(theta)
    z = np.sin(theta) * np.sin(phi)
    return np.stack([x, y, z], axis=-1) #(..., 3)
 
#marching-squares zero contour of one mode over the (theta, phi) grid
#returns flat xyz pairs, two vertices per segment
def _nodal_segments(v, theta, phi):
    nt, npz = v.shape #(n_theta, n_phi)

    #nudge exact zeros off the grid lines; a node landing on a sample point gives
    #a*b == 0, which the sign test misses, dropping that whole nodal line
    eps = 1e-9 * np.abs(v).max()
    v = np.where(np.abs(v) < eps, eps, v)

    #crossings on horizontal edges, between (i, j) and (i, j+1) -> (nt, npz-1)
    a, b = v[:, :-1], v[:, 1:]
    h_hit = (a * b) < 0
    t = np.divide(a, a - b, out=np.zeros_like(a), where=(a - b) != 0)
    h_th = np.broadcast_to(theta[:, None], (nt, npz - 1))
    h_ph = phi[None, :-1] + t * (phi[None, 1:] - phi[None, :-1])
 
    #crossings on vertical edges, between (i, j) and (i+1, j) -> (nt-1, npz)
    c, d = v[:-1, :], v[1:, :]
    v_hit = (c * d) < 0
    s = np.divide(c, c - d, out=np.zeros_like(c), where=(c - d) != 0)
    v_th = theta[:-1, None] + s * (theta[1:, None] - theta[:-1, None])
    v_ph = np.broadcast_to(phi[None, :], (nt - 1, npz))
 
    #each cell owns four edges, ordered top, bottom, left, right -> (n_cells, 4)
    hit = np.stack([h_hit[:-1, :], h_hit[1:, :], v_hit[:, :-1], v_hit[:, 1:]], axis=-1).reshape(-1, 4)
    cth = np.stack([h_th[:-1, :], h_th[1:, :], v_th[:, :-1], v_th[:, 1:]], axis=-1).reshape(-1, 4)
    cph = np.stack([h_ph[:-1, :], h_ph[1:, :], v_ph[:, :-1], v_ph[:, 1:]], axis=-1).reshape(-1, 4)
 
    keep = hit.sum(axis=1) >= 2 #saddles give four, first two are taken
    if not keep.any():
        return []
 
    hit, cth, cph = hit[keep], cth[keep], cph[keep]
    order = np.argsort(~hit, axis=1, kind="stable")[:, :2] #crossing edges first
 
    th_pair = np.take_along_axis(cth, order, axis=1) #(n_seg, 2)
    ph_pair = np.take_along_axis(cph, order, axis=1) #(n_seg, 2)
 
    return _to_xyz(th_pair, ph_pair).reshape(-1).round(4).tolist() #flat xyz, 6 per segment
 
#evaluate every mode on a sphere grid and pack the whole viewer payload
def _build_payload(Y, thetas, phis, n_theta, n_phi):
    from ...helpers import build_sh_basis #lazy, visuals sits below helpers in the package
 
    n_modes, n_chns = Y.shape
    L = int(round(np.sqrt(n_modes))) - 1
    assert (L + 1)**2 == n_modes, f"Y has {n_modes} rows, not a perfect square (L+1)^2"
 
    #phi endpoint duplicated so the seam closes; poles are degenerate and draw nothing
    grid_theta = np.linspace(0.0, np.pi, n_theta)
    grid_phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    tt, pp = np.meshgrid(grid_theta, grid_phi, indexing="ij") #(n_theta, n_phi)
 
    #same function that built Y, so the sphere and the electrode dots cannot disagree
    Y_grid = build_sh_basis(theta=tt.ravel(), phi=pp.ravel(), L=L) #(n_modes, n_theta*n_phi)
 
    modes = []
    for k in range(n_modes):
        field = Y_grid[k].reshape(n_theta, n_phi) #(n_theta, n_phi)
        absmax = float(np.abs(field).max())
        norm = field / absmax if absmax > 0 else field #shipped in [-1, 1]
 
        l = int(np.floor(np.sqrt(k)))
        m = int(k - l * l - l)
 
        modes.append({
            "l": l, "m": m,
            "absmax": round(absmax, 5),
            "values": norm.reshape(-1).round(4).tolist(), #row-major over (n_theta, n_phi)
            "nodes": _nodal_segments(field, grid_theta, grid_phi),
        })
 
    return {
        "n_theta": n_theta,
        "n_phi": n_phi,
        "n_chns": n_chns,
        "L": L,
        "modes": modes,
        "electrodes": _to_xyz(np.asarray(thetas), np.asarray(phis)).round(5).tolist(), #(n_chns, 3)
        "Y": np.asarray(Y).round(5).tolist(), #(n_modes, n_chns)
    }
 
#serve the web folder plus the in-memory payload, return the running server and its port
def _serve(payload_bytes):
    import functools, http.server, socketserver, threading
    from pathlib import Path
 
    web_dir = (Path(__file__).parent / "web").resolve()
 
    class _Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] == "/payload.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload_bytes)))
                self.end_headers()
                self.wfile.write(payload_bytes)
                return
            return super().do_GET()
 
        def log_message(self, *args):
            pass #keep the console clean
 
    handler = functools.partial(_Handler, directory=str(web_dir))
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    server.allow_reuse_address = True
 
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]
 
#blocking; opens a window showing one basis mode at a time on the fitted sphere,
#with the montage's electrodes drawn as the sampled columns of that mode's row in Y
def view_basis_sphere(Y, thetas, phis, n_theta=48, n_phi=96, width=1280, height=820, subtitle=None):
    import json
    import webview #lazy import
 
    assert Y.shape[1] == len(thetas) == len(phis), f"Y has {Y.shape[1]} columns but {len(thetas)} angles given"
 
    payload = _build_payload(Y=Y, thetas=thetas, phis=phis, n_theta=n_theta, n_phi=n_phi)
    payload_bytes = json.dumps(payload).encode("utf-8")
 
    server, port = _serve(payload_bytes)

    try:
        webview.create_window(title=subtitle or "spherical harmonic basis",
                              url=f"http://127.0.0.1:{port}/index.html",
                              width=width, height=height)
        webview.start(debug=False) #blocks until the window closes

    finally:
        server.shutdown()
        server.server_close()