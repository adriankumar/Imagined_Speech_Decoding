//electrode selector popup: draw the resolved electrodes at their 2d positions, click to select, commit an action
//reads the running app's snapshot for positions and the current target re-reference, posts the selection back
//the same server backs this window, so committing re-resolves through the env and the main window's poll picks it up
let selected = new Set();
let snapshot = null;

//context-aware header so the popup states what the selection is for
const TITLES = {exclude: "select electrodes to exclude", reference: "select target re-reference electrodes"};

//fetch the current state from the main flask app to get electrode positions and the current reference
async function loadState() {
    document.getElementById("widget-title").textContent = TITLES[ACTION_CONTEXT] || "select electrodes";
    const response = await fetch("/state");
    snapshot = await response.json();
    if (!snapshot.loaded) {
        document.getElementById("widget-status").textContent = "no source loaded";
        return;
    }
    renderSvg();
    renderActions();
}

//draw electrodes, clicking toggles selection, channels in the current target re-reference shown in purple
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
    const scale = maxR > 0 ? 1.0 / maxR : 1.0;

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

        //visual class priority: selected, then target-ref, then default
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
        addButton(actions, "set target re-reference", () => commitAction("reference"), true);
        addButton(actions, "use average", () => commitAction("average"), false);
    }

    //clear selection is always available
    addButton(actions, "clear selection", () => {
        selected.clear();
        renderSvg();
        updateCount();
    }, false);
}

function addButton(parent, label, handler, requiresSelection) {
    const btn = document.createElement("button");
    btn.className = "seg";
    btn.textContent = label;
    btn.dataset.requiresSelection = requiresSelection;
    btn.disabled = requiresSelection && selected.size === 0;
    btn.addEventListener("click", handler);
    parent.appendChild(btn);
    return btn;
}

function updateCount() {
    document.getElementById("selection-count").textContent = `${selected.size} selected`;
    //update the disabled state on buttons that require a selection
    for (const btn of document.querySelectorAll("#widget-actions button[data-requires-selection='true']")) {
        btn.disabled = selected.size === 0;
    }
}

//post the current selection and action type to the backend, python closes this window on success
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
            //the env refused, e.g. a re-reference clash or a harmonic-capacity violation, the window stays open
            status.textContent = "error: " + (data.message || "unknown");
            return;
        }
        //python closes this window after responding, the main window's poll reflects the change
        status.textContent = "done";
    } catch (err) {
        status.textContent = "request failed: " + err.message;
    }
}

//pywebview and plain-browser fallback, same pattern as the main page
window.addEventListener("pywebviewready", loadState);
if (!window.pywebview) {
    setTimeout(() => { if (!snapshot) loadState(); }, 200);
}