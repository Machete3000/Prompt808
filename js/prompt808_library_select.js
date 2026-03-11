import { app } from "../../scripts/app.js";

/**
 * Prompt808 Library Select — custom node with dynamic library slots.
 *
 * Follows rgthree Power Lora Loader's exact widget lifecycle pattern:
 * - configure() clears all widgets, restores data, then adds static UI
 * - addNonSlotWidgets() always called last to keep button at bottom
 * - New slots inserted before a spacer widget to maintain ordering
 */

const MARGIN = 10;
const INNER_MARGIN = Math.round(MARGIN * 0.33);
const TOGGLE_WIDTH_RATIO = 1.5;

/** Cached library names, refreshed on node creation and sidebar events. */
let _libraryNames = [];

/** Track live Select Libraries nodes for refresh events. */
const _liveSelectNodes = new Set();

async function fetchLibraryNames() {
  try {
    const resp = await fetch("/prompt808/api/libraries");
    if (!resp.ok) return;
    const data = await resp.json();
    _libraryNames = (data.libraries || []).map((l) => l.name);
  } catch {
    // Server not reachable
  }
}

/** Move an array element to a specific index. */
function moveArrayItem(arr, item, toIndex) {
  const fromIndex = arr.indexOf(item);
  if (fromIndex < 0 || fromIndex === toIndex) return;
  arr.splice(fromIndex, 1);
  arr.splice(toIndex, 0, item);
}

/** Truncate text with ellipsis to fit within maxWidth. */
function fitString(ctx, str, maxWidth) {
  if (ctx.measureText(str).width <= maxWidth) return str;
  let t = str;
  while (t.length > 1 && ctx.measureText(t + "\u2026").width > maxWidth) {
    t = t.slice(0, -1);
  }
  return t + "\u2026";
}

/** Draw a rounded rectangle. */
function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

// ------------------------------------------------------------------
// LibrarySlotWidget — one row: [library name]  ... [toggle]
// ------------------------------------------------------------------

class LibrarySlotWidget {
  constructor(name) {
    this.name = name;
    this.type = "custom";
    this._value = { on: true, name: "" };
    this.options = {};
    this.y = 0;
    this.last_y = 0;
    this._hitAreas = {};
  }

  get value() {
    return this._value;
  }

  set value(v) {
    this._value = typeof v === "object" && v ? v : { on: true, name: "" };
  }

  computeSize(width) {
    return [width, LiteGraph.NODE_WIDGET_HEIGHT];
  }

  serializeValue() {
    return this._value;
  }

  draw(ctx, node, w, posY, height) {
    this.last_y = posY;
    const h = height || LiteGraph.NODE_WIDGET_HEIGHT;
    const midY = posY + h / 2;
    const rowX = MARGIN;
    const rowW = w - MARGIN * 2;

    const alpha = (app.canvas && app.canvas.editor_alpha) || 1;
    const bgColor = LiteGraph.WIDGET_BGCOLOR || "#2a2a2a";
    const outlineColor = LiteGraph.WIDGET_OUTLINE_COLOR || "#333";
    const textColor = LiteGraph.WIDGET_TEXT_COLOR || "#ddd";

    // ---- Row background (rounded pill) ----
    ctx.fillStyle = bgColor;
    ctx.strokeStyle = outlineColor;
    ctx.lineWidth = 1;
    roundRect(ctx, rowX, posY, rowW, h, h * 0.5);
    ctx.fill();
    ctx.stroke();

    // ---- Toggle (right side, iOS-style pill) ----
    const toggleH = h - 8;
    const toggleW = toggleH * TOGGLE_WIDTH_RATIO;
    const toggleX = rowX + rowW - INNER_MARGIN - toggleW - 4;
    const toggleY = posY + 4;
    const toggleR = toggleH * 0.5;
    const circleR = toggleH * 0.36;

    // Toggle pill background
    ctx.save();
    ctx.globalAlpha = alpha * 0.25;
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    roundRect(ctx, toggleX, toggleY, toggleW, toggleH, toggleR);
    ctx.fill();
    ctx.restore();

    // Toggle knob
    const knobX = this._value.on
      ? toggleX + toggleW - toggleR
      : toggleX + toggleR;
    ctx.beginPath();
    ctx.arc(knobX, toggleY + toggleH / 2, circleR, 0, Math.PI * 2);
    ctx.fillStyle = this._value.on ? "#89B" : "#888";
    ctx.fill();

    this._hitAreas.toggle = {
      x: toggleX - 4,
      y: posY,
      w: toggleW + 8,
      h: h,
    };

    // ---- Library name label (left side, clickable for dropdown) ----
    const labelX = rowX + INNER_MARGIN + 8;
    const labelW = toggleX - labelX - 6;
    const displayName = this._value.name || "(select library)";

    ctx.save();
    ctx.globalAlpha = alpha * (this._value.on ? 1 : 0.4);
    ctx.fillStyle = this._value.name ? textColor : "#888";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(fitString(ctx, displayName, labelW), labelX, midY);
    ctx.restore();

    this._hitAreas.label = {
      x: rowX,
      y: posY,
      w: toggleX - rowX,
      h: h,
    };
  }

  mouse(event, pos, node) {
    if (event.type !== "pointerdown") return false;
    const localX = pos[0];
    const localY = pos[1];

    const t = this._hitAreas.toggle;
    if (t && localX >= t.x && localX <= t.x + t.w && localY >= t.y && localY <= t.y + t.h) {
      this._value = { ...this._value, on: !this._value.on };
      node.setDirtyCanvas(true, true);
      return true;
    }

    const l = this._hitAreas.label;
    if (l && localX >= l.x && localX <= l.x + l.w && localY >= l.y && localY <= l.y + l.h) {
      this._showDropdown(node, event);
      return true;
    }

    return false;
  }

  _showDropdown(node, mouseEvent) {
    const names = _libraryNames.length ? _libraryNames : ["(no libraries)"];
    new LiteGraph.ContextMenu(
      names.map((n) => ({ content: n })),
      {
        event: mouseEvent,
        callback: (item) => {
          if (item && item.content && item.content !== "(no libraries)") {
            this._value = { ...this._value, name: item.content };
            node.setDirtyCanvas(true, true);
          }
        },
        parentMenu: null,
      },
    );
  }
}

// ------------------------------------------------------------------
// ButtonWidget — "Add Library" button (matches rgthree pattern)
// ------------------------------------------------------------------

class AddButtonWidget {
  constructor(label, callback) {
    this.name = label;
    this.type = "custom";
    this.value = "";
    this.options = { serialize: false };
    this.y = 0;
    this.last_y = 0;
    this._callback = callback;
    this._isPressed = false;
  }

  computeSize(width) {
    return [width, LiteGraph.NODE_WIDGET_HEIGHT];
  }

  serializeValue() {
    return undefined;
  }

  draw(ctx, node, w, posY, height) {
    this.last_y = posY;
    const h = height || LiteGraph.NODE_WIDGET_HEIGHT;
    const btnX = 15;
    const btnW = w - 30;
    const btnY = posY;

    // Shadow
    ctx.fillStyle = "#000000aa";
    roundRect(ctx, btnX + 1, btnY + 1, btnW, h, 4);
    ctx.fill();

    // Background
    ctx.fillStyle = this._isPressed ? "#444" : (LiteGraph.WIDGET_BGCOLOR || "#2a2a2a");
    roundRect(ctx, btnX, btnY, btnW, h, 4);
    ctx.fill();

    // Outer border
    ctx.strokeStyle = "#00000044";
    ctx.lineWidth = 1;
    roundRect(ctx, btnX, btnY, btnW, h, 4);
    ctx.stroke();

    // Inner highlight
    ctx.strokeStyle = "#ffffff11";
    roundRect(ctx, btnX + 1, btnY + 1, btnW - 2, h - 2, 3);
    ctx.stroke();

    // Text
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#ddd";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(this.name, btnX + btnW / 2, btnY + h / 2);

    this._btnBounds = { x: btnX, y: btnY, w: btnW, h: h };
  }

  mouse(event, pos, node) {
    const b = this._btnBounds;
    if (!b) return false;
    const inside = pos[0] >= b.x && pos[0] <= b.x + b.w &&
                   pos[1] >= b.y && pos[1] <= b.y + b.h;

    if (event.type === "pointerdown" && inside) {
      this._isPressed = true;
      node.setDirtyCanvas(true, true);
      return true;
    }
    if (event.type === "pointerup") {
      if (this._isPressed && inside) {
        this._isPressed = false;
        this._callback(event, pos, node);
        node.setDirtyCanvas(true, true);
        return true;
      }
      this._isPressed = false;
    }
    return false;
  }
}

// ------------------------------------------------------------------
// Node registration
// ------------------------------------------------------------------

app.registerExtension({
  name: "Prompt808.LibrarySelect",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Prompt808 Library Select") return;

    // ---- onNodeCreated ----
    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      origOnNodeCreated?.apply(this, arguments);
      this.serialize_widgets = true;
      this._slotCounter = 0;
      this._buttonSpacer = null;

      _liveSelectNodes.add(this);
      fetchLibraryNames();

      // Add 2 default slots, then static UI (button at bottom)
      this._addSlot();
      this._addSlot();
      this._addNonSlotWidgets();

      const computed = this.computeSize();
      this.size = this.size || [0, 0];
      this.size[0] = Math.max(this.size[0], computed[0]);
      this.size[1] = Math.max(this.size[1], computed[1]);
      this.setDirtyCanvas(true, true);
    };

    // ---- onRemoved ----
    const origOnRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      _liveSelectNodes.delete(this);
      origOnRemoved?.apply(this, arguments);
    };

    // ---- configure (restore from saved workflow) ----
    const origConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      // 1. Clear ALL widgets (call onRemove if present, then clear array)
      if (this.widgets) {
        for (const w of this.widgets) {
          if (w.onRemove) w.onRemove();
        }
        this.widgets.length = 0;
      }
      this._buttonSpacer = null;
      this._slotCounter = 0;

      // 2. Call parent configure
      if (info.id != null) {
        origConfigure?.apply(this, arguments);
      }

      // 3. Save size before restoring
      const savedW = this.size[0];
      const savedH = this.size[1];

      // 4. Restore slot widgets from saved values
      for (const val of info.widgets_values || []) {
        if (val && typeof val === "object" && "on" in val && "name" in val) {
          const widget = this._addSlot();
          widget.value = { ...val };
        }
      }

      // Ensure at least one slot
      const slotCount = (this.widgets || []).filter(
        (w) => w instanceof LibrarySlotWidget,
      ).length;
      if (slotCount === 0) {
        this._addSlot();
      }

      // 5. Add static UI (button) LAST — always at bottom
      this._addNonSlotWidgets();

      // 6. Restore size
      this.size[0] = savedW;
      this.size[1] = Math.max(savedH, this.computeSize()[1]);

      fetchLibraryNames();
    };

    // ---- addSlot ----
    nodeType.prototype._addSlot = function () {
      this._slotCounter++;
      const name = "library_" + this._slotCounter;
      const widget = this.addCustomWidget(new LibrarySlotWidget(name));

      // Insert before the button spacer to maintain order
      if (this._buttonSpacer) {
        moveArrayItem(
          this.widgets,
          widget,
          this.widgets.indexOf(this._buttonSpacer),
        );
      }

      return widget;
    };

    // ---- addNonSlotWidgets (button always at bottom) ----
    nodeType.prototype._addNonSlotWidgets = function () {
      // Spacer (invisible divider that marks where button zone starts)
      this._buttonSpacer = this.addCustomWidget({
        name: "spacer",
        type: "custom",
        value: "",
        options: { serialize: false },
        y: 0,
        last_y: 0,
        computeSize() { return [0, 4]; },
        draw() {},
        mouse() { return false; },
        serializeValue() { return undefined; },
      });

      // Add Library button
      this.addCustomWidget(
        new AddButtonWidget("\u2795 Add Library", (_event, _pos, _node) => {
          this._addSlot();
          const computed = this.computeSize();
          this.size[1] = Math.max(this.size[1], computed[1]);
          this.setDirtyCanvas(true, true);
        }),
      );
    };

    // ---- removeSlot ----
    nodeType.prototype._removeSlot = function (widget) {
      const idx = this.widgets?.indexOf(widget);
      if (idx == null || idx < 0) return;
      this.widgets.splice(idx, 1);
      const inputIdx = this.inputs?.findIndex((inp) => inp.name === widget.name);
      if (inputIdx >= 0) {
        this.removeInput(inputIdx);
      }
      const computed = this.computeSize();
      this.size[1] = Math.max(computed[1], this.size[1]);
      this.setDirtyCanvas(true, true);
    };

    // ---- Right-click context menu ----
    const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
      origGetExtraMenuOptions?.apply(this, arguments);

      const slotWidgets = (this.widgets || []).filter(
        (w) => w instanceof LibrarySlotWidget,
      );
      if (slotWidgets.length <= 1) return;

      const mouseY = canvas.graph_mouse[1] - this.pos[1];
      for (const sw of slotWidgets) {
        if (mouseY >= sw.last_y && mouseY < sw.last_y + LiteGraph.NODE_WIDGET_HEIGHT) {
          options.unshift({
            content: `Remove "${sw.value.name || "empty slot"}"`,
            callback: () => this._removeSlot(sw),
          });
          break;
        }
      }
    };
  },
});

// Refresh library names and update live nodes when sidebar signals a change
document.addEventListener("prompt808:options-changed", async () => {
  await fetchLibraryNames();
  for (const node of _liveSelectNodes) {
    const slots = (node.widgets || []).filter(
      (w) => w instanceof LibrarySlotWidget,
    );
    for (const slot of slots) {
      if (slot._value.name && !_libraryNames.includes(slot._value.name)) {
        slot._value = { ...slot._value, name: "" };
      }
    }
    node.setDirtyCanvas(true, true);
  }
});
