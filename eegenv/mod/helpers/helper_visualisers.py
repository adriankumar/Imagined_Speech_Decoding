import numpy as np

#===================================================================
# primitives, pure figure builders over arrays, no env, no show, no save
#===================================================================
#lay out F panels, call draw_fn(ax, i) per feature, attach a per-panel colourbar so each feature keeps its own scale
def _panel_grid(F, names, draw_fn, suptitle=None):
    import matplotlib.pyplot as plt #lazy import
    cols = min(F, 3)
    rows = int(np.ceil(F / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < F:
            m = draw_fn(ax, i) #the draw returns its scalar mappable
            ax.set_title(names[i] if names is not None else f"f{i}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig

#a single matrix as one heatmap, aspect auto since the two axes rarely match in length
def _single_heatmap(matrix, title=None, xlabel=None, ylabel=None, col_names=None):
    import matplotlib.pyplot as plt #lazy import
    fig, ax = plt.subplots(figsize=(6, 4))
    m = ax.imshow(matrix, origin="lower", cmap="viridis", aspect="auto")
    if title:
        ax.set_title(title, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if col_names is not None:
        ax.set_xticks(range(len(col_names)))
        ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=8)
    fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

#===================================================================
# per-panel draws, image is the interpolated field, stack is the electrode scatter
#===================================================================
#one interpolated feature map per panel, anterior up via origin lower
def _draw_image(image):
    def draw(ax, i):
        return ax.imshow(image[:, :, i], origin="lower", cmap="viridis")
    return draw

#one feature per panel as the electrode scatter at the 2d projection
def _draw_stack(stack, pos_2d):
    def draw(ax, i):
        sc = ax.scatter(pos_2d[:, 0], pos_2d[:, 1], c=stack[:, i], cmap="viridis", s=30)
        ax.set_aspect("equal")
        return sc
    return draw

#===================================================================
# figure builders, one logical view each, return the figure
#===================================================================
#render a feature representation, kind image is the interpolated (H, W, F) field, kind stack is the (n_channels, F) scatter
def build_features_figure(array, names, kind="image", pos_2d=None, suptitle=None):
    if kind == "image":
        return _panel_grid(array.shape[2], names, _draw_image(array), suptitle)
    if kind == "stack":
        if pos_2d is None:
            raise ValueError("stack kind needs pos_2d for the electrode positions")
        return _panel_grid(array.shape[1], names, _draw_stack(array, pos_2d), suptitle)
    raise ValueError(f"kind {kind} must be 'image' or 'stack'")

#render the raw window (n_channels, window_size), rows are channels, columns are time
def build_raw_window_figure(window, title="raw window"):
    return _single_heatmap(window, title=title, xlabel="time (samples)", ylabel="channel")

#render the harmonic coefficients ((L+1)^2, F), rows are modes, columns are features
def build_compressed_figure(coeffs, names=None, title="harmonic coefficients"):
    return _single_heatmap(coeffs, title=title, xlabel="feature", ylabel="mode", col_names=names)

#===================================================================
# env-facing convenience, gather the tensors the env provides then build and show or save
#===================================================================
#read-only feature stack at a start, defaults to the cursor, never advances the live stream
def _features_at(env, start, feature_toggles):
    start = env.window_cursor if start is None else start
    stack = env.preview_window_features(start, feature_toggles=feature_toggles)
    names = env.features.enabled_names(feature_toggles)
    return stack, names

#assemble the decode slots for a kind, current then delta then decoded, and actual only when the delta is not the true delta
def _decode_slots(env, preview, kind, names, show_actual):
    slots = [("delta coefficients", build_compressed_figure(preview.delta_coeffs, names))]
    if kind == "image":
        slots.insert(0, ("current", build_features_figure(preview.before_image, names, "image")))
        slots.append(("decoded next", build_features_figure(preview.after_image, names, "image")))
        if show_actual:
            slots.append(("actual next", build_features_figure(env.to_image(preview.target_stack), names, "image")))
    else:
        pos = env.electrode_pos_2d
        slots.insert(0, ("current", build_features_figure(preview.current_stack, names, "stack", pos)))
        slots.append(("decoded next", build_features_figure(preview.predicted_stack, names, "stack", pos)))
        if show_actual:
            slots.append(("actual next", build_features_figure(preview.target_stack, names, "stack", pos)))
    return slots

#--- raw window ---
def view_raw_window(env, start=None):
    import matplotlib.pyplot as plt
    build_raw_window_figure(env.get_raw_window(start))
    plt.show()

def save_raw_window(env, path, start=None):
    import matplotlib.pyplot as plt
    fig = build_raw_window_figure(env.get_raw_window(start))
    fig.savefig(path); plt.close(fig)

#--- feature representation, image or stack ---
def view_features(env, start=None, kind="image", feature_toggles=None):
    import matplotlib.pyplot as plt
    stack, names = _features_at(env, start, feature_toggles)
    array = env.to_image(stack) if kind == "image" else stack
    build_features_figure(array, names, kind=kind, pos_2d=env.electrode_pos_2d)
    plt.show()

def save_features(env, path, start=None, kind="image", feature_toggles=None):
    import matplotlib.pyplot as plt
    stack, names = _features_at(env, start, feature_toggles)
    array = env.to_image(stack) if kind == "image" else stack
    fig = build_features_figure(array, names, kind=kind, pos_2d=env.electrode_pos_2d)
    fig.savefig(path); plt.close(fig)

#--- compressed coefficients ---
def view_compressed(env, start=None, feature_toggles=None):
    import matplotlib.pyplot as plt
    stack, names = _features_at(env, start, feature_toggles)
    build_compressed_figure(env.to_sh_compression(stack), names=names)
    plt.show()

def save_compressed(env, path, start=None, feature_toggles=None):
    import matplotlib.pyplot as plt
    stack, names = _features_at(env, start, feature_toggles)
    fig = build_compressed_figure(env.to_sh_compression(stack), names=names)
    fig.savefig(path); plt.close(fig)

#--- decode at the cursor, true delta shows three slots, a supplied delta adds the actual-next slot ---
def view_decode(env, delta_coeffs=None, kind="image"):
    import matplotlib.pyplot as plt
    preview = env.preview_decode_at_cursor(delta_coeffs)
    names = env.features.enabled_names()
    for label, fig in _decode_slots(env, preview, kind, names, delta_coeffs is not None):
        fig.suptitle(label)
    plt.show()

def save_decode(env, path_prefix, delta_coeffs=None, kind="image"):
    import matplotlib.pyplot as plt
    preview = env.preview_decode_at_cursor(delta_coeffs)
    names = env.features.enabled_names()
    for label, fig in _decode_slots(env, preview, kind, names, delta_coeffs is not None):
        fig.suptitle(label)
        fig.savefig(f"{path_prefix}_{label.replace(' ', '_')}.png"); plt.close(fig)