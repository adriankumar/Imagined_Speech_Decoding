//local selection set, channel names only, committed on action button press
let selected = new Set();
let snapshot = null;

//fetch the current state from the main flask app to get electrode positions
async function loadState() {
    const response = await fetch("/state");
    snapshot = await response.json();
    if (!snapshot.loaded) {
        document.getElementById("widget-status").textContent = "no source loaded";
        return;
    }
    renderSvg();
    renderActions();
}

//draw electrodes, clicking toggles selection, already-target-ref channels shown in purple
function renderSvg() {
    const svg = document.getElementById("electrode-svg");
    svg.innerHTML = "";

    const positions = snapshot.geometry.pos_2d;
    const names = snapshot.geometry.channel_names;
    const targetRef = snapshot.config.target_ref;
    //target_ref is either "average" or a list of channel names
    const refSet = Array.isArray(targetRef) ? new Set(targetRef) : new Set();

    let maxR = 0;
    for (const [x, y] of positions) {
        const r = Math.sqrt(x * x + y * y);
        if (r > maxR) maxR = r;
    }
    const scale = 1.0 / maxR;

    for (let i = 0; i < positions.length; i++) {
        const name = names[i];
        const [x, y] = positions[i];
        const cx = x * scale;
        const cy = -y * scale;

        const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        c.setAttribute("cx", cx);
        c.setAttribute("cy", cy);
        c.setAttribute("r", 0.035);
        c.dataset.name = name;

        //set visual class: selected > target-ref > default, in priority order
        if (selected.has(name)) {
            c.setAttribute("class", "electrode selected");
        } else if (refSet.has(name)) {
            c.setAttribute("class", "electrode target-ref");
        } else {
            c.setAttribute("class", "electrode");
        }

        //click toggles selection and redraws only this circle's class
        c.addEventListener("click", () => {
            if (selected.has(name)) {
                selected.delete(name);
                c.setAttribute("class", refSet.has(name) ? "electrode target-ref" : "electrode");
            } else {
                selected.add(name);
                c.setAttribute("class", "electrode selected");
            }
            updateCount();
        });

        const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        title.textContent = name;
        c.appendChild(title);
        svg.appendChild(c);
    }
}

//render action buttons appropriate for the current context
function renderActions() {
    const actions = document.getElementById("widget-actions");
    actions.innerHTML = "";

    if (ACTION_CONTEXT === "exclude") {
        addButton(actions, "exclude selected", () => commitAction("exclude"), true);
        addButton(actions, "reset exclusions", () => commitAction("reset"), false);
    }
    if (ACTION_CONTEXT === "reference") {
        addButton(actions, "set as reference target", () => commitAction("reference"), true);
        addButton(actions, "use average reference", () => commitAction("average"), false);
    }

    //clear selection button always present
    addButton(actions, "clear selection", () => {
        selected.clear();
        renderSvg();
        updateCount();
    }, false);
}

function addButton(parent, label, handler, requiresSelection) {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.dataset.requiresSelection = requiresSelection;
    btn.disabled = requiresSelection && selected.size === 0;
    btn.addEventListener("click", handler);
    parent.appendChild(btn);
    return btn;
}

function updateCount() {
    document.getElementById("selection-count").textContent = `${selected.size} selected`;
    //update disabled state on buttons that require a selection
    for (const btn of document.querySelectorAll("#widget-actions button[data-requires-selection='true']")) {
        btn.disabled = selected.size === 0;
    }
}

//post the current selection and action type to the backend, close on success
async function commitAction(action) {
    const status = document.getElementById("widget-status");
    status.textContent = "applying...";
    try {
        const response = await fetch("/electrode_action", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                action: action,
                channels: Array.from(selected),
            }),
        });
        const data = await response.json();
        if (!response.ok) {
            status.textContent = "error: " + (data.message || "unknown");
            return;
        }
        //python closes this window after responding, nothing to do here
        status.textContent = "done";
    } catch (err) {
        status.textContent = "request failed: " + err.message;
    }
}

//pywebview and plain-browser fallback, same pattern as main page
window.addEventListener("pywebviewready", loadState);
if (!window.pywebview) {
    setTimeout(() => { if (!snapshot) loadState(); }, 200);
}