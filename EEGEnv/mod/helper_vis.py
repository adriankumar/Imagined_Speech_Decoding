import numpy as np
from .feature_stack import FeatureStack
from .helper_maths import encode_image

#===================================================================
# panel scaffold, shared by image and stack, static and animated
#===================================================================
#lay out F panels and call draw_fn(ax, i) for each enabled feature and attach its colourbar, return the figure and the per-panel mappables
def _panel_grid(F, names, draw_fn):
    import matplotlib.pyplot as plt
    cols = min(F, 3)
    rows = int(np.ceil(F / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3))
    axes = np.atleast_1d(axes).ravel()
    mappables = []
    for i, ax in enumerate(axes):
        if i < F:
            m = draw_fn(ax, i)  #the draw returns its scalar mappable
            ax.set_title(names[i], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)  #per-panel value scale for interpretation
            mappables.append(m)
        else:
            ax.axis("off")
    fig.tight_layout()
    return fig, mappables

#per-feature colour ranges across a set of frames, so an animation keeps a fixed scale
def _clims(frames, kind):
    F = frames[0].shape[-1]
    out = []
    for i in range(F):
        vals = np.concatenate([(f[:, :, i].ravel() if kind == "image" else f[:, i]) for f in frames])
        out.append((float(vals.min()), float(vals.max())))
    return out

#===================================================================
# renderers, image is the interpolated field, stack is the raw electrodes
#===================================================================
#draw the (H, W, F) interpolated field, one map per feature, anterior up
def render_image(image, names, clims=None):
    F = image.shape[2]
    def draw(ax, i):
        vmin, vmax = clims[i] if clims else (None, None)
        return ax.imshow(image[:, :, i], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    return _panel_grid(F, names, draw)

#draw the (n_channels, F) raw values at their 2d electrode positions, one scatter per feature
def render_stack(stack, pos_2d, names, clims=None):
    F = stack.shape[1]
    def draw(ax, i):
        vmin, vmax = clims[i] if clims else (None, None)
        sc = ax.scatter(pos_2d[:, 0], pos_2d[:, 1], c=stack[:, i], cmap="viridis", s=30, vmin=vmin, vmax=vmax)
        ax.set_aspect("equal")
        return sc
    return _panel_grid(F, names, draw)

#build the right figure for a kind, image or stack, returns the figure and the per-panel mappables
def _figure(env, array, names, kind, clims=None):
    if kind == "image":
        return render_image(array, names, clims)
    return render_stack(array, env.SH_dict['pos_2d'], names, clims)

#===================================================================
# single window, view and save, read-only against the live stream
#===================================================================

#compute one window read-only and shape it for the chosen kind, accumulator peeked without state advance
def _one_window(env, start, length, feature_toggles, kind):
    stack = env._compute_window(start, length, env.features, feature_toggles=feature_toggles,
                                update=False)
    
    base_names = env.features.enabled_names(feature_toggles)
    names = base_names

    array = encode_image(env.SH_dict['interpol_operator'], stack, env.img_res) if kind == "image" else stack
    return array, names

def view_window(env, start, length, feature_toggles=None, kind="image"):
    import matplotlib.pyplot as plt #lazy import

    array, names = _one_window(env, start, length, feature_toggles, kind)
    _figure(env, array, names, kind)
    plt.show()

def save_window(env, path, start, length, feature_toggles=None, kind="image"):
    import matplotlib.pyplot as plt #lazy import

    array, names = _one_window(env, start, length, feature_toggles, kind)
    fig, _ = _figure(env, array, names, kind)
    fig.savefig(path)
    plt.close(fig)


#===================================================================
# animation, view and save, isolated lag so the live stream is untouched
#===================================================================
#precompute consecutive frames on a private feature stack and accumulator clone, never touching env state
def _frames(env, start, length, n_frames, step, feature_toggles, kind):

    #unpack only the eight base toggle keys the constructor signature accepts, ema toggles are derived from accum
    base_toggle = {k: env.features.feature_toggle[k]
                   for k in ("raw", "median", "iqr", "mobility", "complexity",
                              "raw_lag", "median_lag", "iqr_lag")}
    cache = FeatureStack(**base_toggle,
                         scale=env.features.scale,
                         weight=env.features.weight,
                         accum=env.features.accum,
                         alpha=env.features.alpha)  #mirror the full feature config including ema state

    M, img_res = env.SH_dict['interpol_operator'], env.img_res
    frames, s = [], start

    base_names = cache.enabled_names(feature_toggles)
    names = base_names

    for _ in range(n_frames):
        if s + length > env.time_points:
            break
        stack = env._compute_window(s, length, cache, feature_toggles=feature_toggles,
                                    update=True)
        frames.append(encode_image(M, stack, img_res) if kind == "image" else stack)
        s += step

    return frames, names

#build the figure and the per-frame updater for an animation
def _build_animation(env, frames, names, kind):
    from matplotlib.animation import FuncAnimation

    clims = _clims(frames, kind)
    fig, mappables = _figure(env, frames[0], names, kind, clims)

    if kind == "image":
        update = lambda k: [mappables[i].set_data(frames[k][:, :, i]) for i in range(len(mappables))]
    else:
        update = lambda k: [mappables[i].set_array(frames[k][:, i]) for i in range(len(mappables))]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=False)

    return fig, anim

def view_animation(env, start, length, n_frames, step, feature_toggles=None, kind="image"):
    import matplotlib.pyplot as plt #lazy import

    frames, names = _frames(env, start, length, n_frames, step, feature_toggles, kind)
    fig, anim = _build_animation(env, frames, names, kind)  #anim kept alive through show
    plt.show()

def save_animation(env, path, start, length, n_frames, step, feature_toggles=None, kind="image", fps=10):
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter #lazy imports

    frames, names = _frames(env, start, length, n_frames, step, feature_toggles, kind)
    fig, anim = _build_animation(env, frames, names, kind)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)