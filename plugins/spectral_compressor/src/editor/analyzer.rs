// Spectral Compressor: an FFT based compressor
// Copyright (C) 2021-2024 Robbert van der Helm
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

use atomic_float::AtomicF32;
use nih_plug::nih_debug_assert;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::vg;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::{ChainMode, EditorEvent};
use crate::analyzer::AnalyzerData;
use crate::capture::{smooth_into, CaptureBlend};
use crate::compressor_bank::{LF_BYPASS_FADE_OCTAVES, LF_BYPASS_OFF_HZ, NUM_CHAINS};
use crate::curve::{Curve, CurveParams};
use crate::eq_curve::CompressorDirection;
use crate::eq_curve::{
    soloed_index, EqBankParams, EqCurve, EqCurveParams, EqNodeTarget, EqNodeType,
};
use crate::SpectralCompressorParams;
use nih_plug::prelude::{Enum, Param, ParamPtr};
use nih_plug_vizia::widgets::RawParamEvent;

// We'll show the bins from 30 Hz (to your chest) to 22 kHz, scaled logarithmically
#[allow(unused)]
const FREQ_RANGE_START_HZ: f32 = 30.0;
#[allow(unused)]
const FREQ_RANGE_END_HZ: f32 = 22_000.0;
const LN_FREQ_RANGE_START_HZ: f32 = 3.4011974; // 30.0f32.ln();
const LN_FREQ_RANGE_END_HZ: f32 = 9.998797; // 22_000.0f32.ln();
const LN_FREQ_RANGE: f32 = LN_FREQ_RANGE_END_HZ - LN_FREQ_RANGE_START_HZ;

/// The frequencies that get a labelled gridline underneath the analyzer.
pub(crate) const FREQUENCY_TICKS: &[f32] = &[
    50.0, 100.0, 200.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0, 20_000.0,
];

/// Unlabelled gridlines, drawn dimmer than the labelled ones. Ten to a decade below 1 kHz and
/// every kilohertz above it, which stays readable once the logarithmic scale squeezes the top end.
const MINOR_FREQUENCY_TICKS: &[f32] = &[
    30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0,
    800.0, 900.0, 1_000.0, 2_000.0, 3_000.0, 4_000.0, 5_000.0, 6_000.0, 7_000.0, 8_000.0, 9_000.0,
    10_000.0, 11_000.0, 12_000.0, 13_000.0, 14_000.0, 15_000.0, 16_000.0, 17_000.0, 18_000.0,
    19_000.0, 20_000.0,
];

/// The decibel values that get a horizontal gridline. These are thresholds, not signal levels.
const DB_TICKS: &[f32] = &[-60.0, -40.0, -20.0, 0.0];

/// Where a frequency sits horizontally, as a `[0, 1]` fraction of the analyzer's width.
///
/// The label row in the editor uses this same function, so the labels cannot drift out of
/// alignment with the gridlines drawn here.
pub(crate) fn frequency_to_t(frequency: f32) -> f32 {
    (frequency.ln() - LN_FREQ_RANGE_START_HZ) / LN_FREQ_RANGE
}

/// Render a tick frequency the way it's usually written on an analyzer.
pub(crate) fn format_frequency(frequency: f32) -> String {
    if frequency >= 1000.0 {
        format!("{:.0}k", frequency / 1000.0)
    } else {
        format!("{frequency:.0}")
    }
}

/// The color of the labelled gridlines. Kept dim so the spectrum stays the thing you look at.
const GRID_COLOR: vg::Color = vg::Color::rgbaf(0.35, 0.35, 0.35, 0.35);
/// The unlabelled gridlines are dimmer still, so they read as subdivisions rather than competing
/// with the frequencies that have a number underneath them.
const MINOR_GRID_COLOR: vg::Color = vg::Color::rgbaf(0.35, 0.35, 0.35, 0.15);

/// The radius of a node's handle, in logical pixels.
const NODE_RADIUS: f32 = 5.0;

/// The decibel span the analyzer covers vertically, matching `db_to_unclamped_t`.
const DB_RANGE: f32 = 100.0;

/// How far one scroll wheel notch moves a node's Q. Multiplicative so the steps feel even across
/// the parameter's skewed range.
const SCROLL_Q_FACTOR: f32 = 1.15;

/// How much of its opacity the curve that isn't being edited keeps. Dimming it makes it obvious
/// which curve a click will act on, since only the edited one responds to the mouse.
const UNSELECTED_CURVE_OPACITY: f32 = 0.3;

/// The color used for drawing the overlay. Currently not configurable using the style sheet (that
/// would be possible by moving this to a dedicated view and overlaying that).
///
/// # Notes
///
/// This is drawn using some blending options that make it interact differently with darker
/// backgrounds.
const GR_BAR_OVERLAY_COLOR: vg::Color = vg::Color::rgbaf(0.85, 0.95, 1.0, 0.8);

/// Downwards compression attenuates, so it gets the warmer of the two colors.
const DOWNWARDS_THRESHOLD_CURVE_COLOR: vg::Color = vg::Color::rgbaf(0.82, 0.34, 0.32, 0.9);
/// Upwards compression lifts, so it gets the cooler one.
const UPWARDS_THRESHOLD_CURVE_COLOR: vg::Color = vg::Color::rgbaf(0.25, 0.50, 0.82, 0.9);

/// The wash laid over the range the low frequency bypass leaves alone. Dark rather than tinted, so
/// it reads as "nothing happens here" instead of as another kind of processing.
const LF_BYPASS_WASH_COLOR: vg::Color = vg::Color::rgbaf(0.0, 0.0, 0.0, 0.45);

/// The captured curve, drawn as a reference rather than as something editable. Neutral so it does
/// not read as a third threshold curve.
const CAPTURE_CURVE_COLOR: vg::Color = vg::Color::rgbaf(0.72, 0.72, 0.60, 0.45);

/// Below this the other chain's spectrum isn't drawn at all. At a full detection link the two
/// chains detect on the same signal, so what's being skipped is a second copy of the line already
/// there.
const OTHER_CHAIN_ALPHA_FLOOR: f32 = 0.01;

/// How visible the other chain's spectrum is in [`ChainMode::Split`], where it is drawn as a
/// reference rather than to fill something in.
///
/// Constant rather than following the detection link, unlike in [`ChainMode::Linked`]: raising the
/// link is what pulls the two chains' detection signals together, so this line has to stay visible
/// for that to be watchable.
const OTHER_CHAIN_SPECTRUM_ALPHA: f32 = 0.3;

/// A very analyzer showing the envelope followers as a magnitude spectrum with an overlay for the
/// gain reduction.
pub struct Analyzer<L, LSelected, LMode> {
    analyzer_data: Arc<Mutex<triple_buffer::Output<AnalyzerData>>>,
    sample_rate: Arc<AtomicF32>,
    /// Which compressor's curve the mouse acts on. Read during both drawing and event handling,
    /// so it's kept as a live lens rather than a value copied at build time.
    edited_direction: L,
    /// The parameters, needed to read node values back and to write edits to them.
    params: Arc<SpectralCompressorParams>,
    /// Set for the duration of a drag.
    drag: Option<NodeDrag>,
    /// Which node the inspector below the graph is editing, so its handle can be marked.
    selected_node: LSelected,
    /// Which node is being listened to on its own, shared with the audio thread.
    ///
    /// A plain atomic rather than a lens because it is read from deep inside the drawing and hit
    /// testing, which run without a context to resolve a lens against. Toggling it always changes
    /// the editor's own copy as well, so a redraw is guaranteed to follow.
    node_solo: Arc<AtomicUsize>,
    /// The chain this graph draws, and the one every mouse event on it acts on.
    ///
    /// Baked in at build time rather than read from a lens: each graph is one chain, so clicking a
    /// graph is what picks the chain. The layout rebuilds nothing when the mode changes, it just
    /// shows or hides the second graph.
    chain_idx: usize,
    /// Whether the chains are linked, which decides what a new node is given and whether the other
    /// chain's spectrum is drawn behind this one. A lens, because it changes without this graph
    /// being rebuilt.
    chain_mode: LMode,
}

/// A node handle being dragged.
///
/// The node's values are recorded when the drag starts and the pointer's movement is applied as a
/// delta, so grabbing a handle slightly off-center moves it smoothly instead of snapping it under
/// the cursor.
struct NodeDrag {
    node_index: usize,
    start_frequency: f32,
    start_gain_db: f32,
    start_cursor: (f32, f32),
}

/// Where one node's handle goes on a graph, as `[0, 1]` fractions of the analyzer's bounds.
struct NodeHandle {
    index: usize,
    /// The line the handle sits on. A node applying to both gets one handle on each; they are the
    /// same node, so dragging either moves both.
    direction: CompressorDirection,
    /// Whether the node currently deforms the curve, which is what tells a filled handle from a
    /// hollow one. This already accounts for a solo elsewhere in the bank, not just the node's own
    /// bypass, because the curve it would be drawn against does too.
    active: bool,
    t_x: f32,
    t_y: f32,
}

impl<L, LSelected, LMode> Analyzer<L, LSelected, LMode>
where
    L: Lens<Target = CompressorDirection>,
    LSelected: Lens<Target = Option<usize>>,
    LMode: Lens<Target = ChainMode>,
{
    /// Creates a new [`Analyzer`].
    pub fn new<LAnalyzerData, LRate, LParams>(
        cx: &mut Context,
        analyzer_data: LAnalyzerData,
        sample_rate: LRate,
        params: LParams,
        edited_direction: L,
        selected_node: LSelected,
        node_solo: Arc<AtomicUsize>,
        chain_idx: usize,
        chain_mode: LMode,
    ) -> Handle<'_, Self>
    where
        LAnalyzerData: Lens<Target = Arc<Mutex<triple_buffer::Output<AnalyzerData>>>>,
        LRate: Lens<Target = Arc<AtomicF32>>,
        LParams: Lens<Target = Arc<SpectralCompressorParams>>,
        LMode: Lens<Target = ChainMode>,
    {
        Self {
            analyzer_data: analyzer_data.get(cx),
            sample_rate: sample_rate.get(cx),
            params: params.get(cx),
            edited_direction,
            drag: None,
            selected_node,
            node_solo,
            chain_idx,
            chain_mode,
        }
        .build(
            cx,
            // This is an otherwise empty element only used for custom drawing
            |_cx| (),
        )
    }

    /// The shared bank of nodes. Which curves a node deforms are its own `target` and `channel`
    /// parameters.
    fn nodes(&self) -> &EqBankParams {
        &self.params.threshold.eq
    }

    /// The node being listened to on its own, if any. Every curve the editor builds goes through
    /// this so that soloing shows the same thing it makes the compressors do.
    fn soloed_node(&self) -> Option<usize> {
        soloed_index(self.node_solo.load(Ordering::Relaxed))
    }

    /// Where each node's handle is drawn, as `[0, 1]` fractions of the analyzer's bounds.
    ///
    /// [`Self::draw_nodes()`] places the handles from this too, so hit testing and drawing cannot
    /// disagree about where a node is. Each line's composite curve is built once rather than once
    /// per node, which matters because this runs on every click and scroll.
    ///
    /// Bypassed nodes are included. A handle is how a node is found, so taking it away would leave
    /// no way to switch one back on, and its position does not depend on whether it is active.
    fn node_positions_t(&self, chain_idx: usize, chain_mode: ChainMode) -> Vec<NodeHandle> {
        let mut positions = Vec::new();

        // Both lines are collected, so a handle on the one that isn't focused can still be found.
        // A node applying to both gets a handle on each; they are the same node, so dragging
        // either moves both.
        // A handle sits on the curve it belongs to, so it has to be placed on the same curve that
        // gets drawn -- capture and all. Leaving the capture out left every handle floating off the
        // line, and since hit testing reads these positions too, clicking one missed by the same
        // distance.
        let capture_smoothed = self.smoothed_capture();

        for direction in [CompressorDirection::Upwards, CompressorDirection::Downwards] {
            let (curve_params, eq_params, offset_db) =
                curve_inputs(&self.params, self.soloed_node(), chain_idx, direction);
            let curve = Curve::new(&curve_params);
            let capture_blend = capture_smoothed.as_deref().map(|smoothed| {
                capture_blend_for(
                    &self.params,
                    &curve_params,
                    smoothed,
                    self.params.threshold.capture.amount.value(),
                )
            });

            for (index, node) in eq_params.nodes.iter().enumerate() {
                // Linked means one curve, and `merge_chains()` sees to it that every node really
                // does apply to both chains. One that slipped past that -- from a preset, from
                // automation, from the host writing the parameter directly -- would otherwise
                // vanish from the only graph there is, so nothing is filtered by channel here.
                let hidden_by_channel =
                    chain_mode == ChainMode::Split && !node.channel.applies_to(chain_idx);
                if node.node_type == EqNodeType::Off
                    || !node.target.applies_to(direction)
                    || hidden_by_channel
                {
                    continue;
                }

                let ln_freq = node.center_frequency.ln();
                let capture_delta_db = capture_blend
                    .as_ref()
                    .map_or(0.0, |blend| blend.delta_ln(ln_freq));
                let y_db = curve.evaluate_ln(ln_freq)
                    + capture_delta_db
                    + node.handle_offset_db()
                    + offset_db;

                positions.push(NodeHandle {
                    index,
                    direction,
                    active: node.enabled,
                    t_x: frequency_to_t(node.center_frequency),
                    t_y: 1.0 - db_to_unclamped_t(y_db),
                });
            }
        }

        positions
    }

    /// The parameters a drag moves: the node's frequency and its gain.
    fn drag_param_ptrs(&self, node_index: usize) -> (ParamPtr, ParamPtr) {
        let node = &self.nodes().nodes[node_index];

        (node.center_frequency.as_ptr(), node.gain_db.as_ptr())
    }

    /// The node whose handle is under `(x, y)`, and the line it was grabbed on.
    ///
    /// Handles on the line that isn't focused are grabbable too. Clicking one is how the focus
    /// moves onto it, which beats having to notice it belongs to the other line first.
    fn node_at(
        &self,
        cx: &EventContext,
        chain_idx: usize,
        chain_mode: ChainMode,
        x: f32,
        y: f32,
    ) -> Option<(usize, CompressorDirection)> {
        let bounds = cx.bounds();
        let radius = NODE_RADIUS * cx.scale_factor() * 2.0;

        let mut closest: Option<((usize, CompressorDirection), f32)> = None;
        for handle in self.node_positions_t(chain_idx, chain_mode) {
            let dx = (bounds.x + (bounds.w * handle.t_x)) - x;
            let dy = (bounds.y + (bounds.h * handle.t_y)) - y;
            let distance_squared = (dx * dx) + (dy * dy);
            if distance_squared > radius * radius {
                continue;
            }

            if closest.is_none_or(|(_, best)| distance_squared < best) {
                closest = Some(((handle.index, handle.direction), distance_squared));
            }
        }

        closest.map(|(hit, _)| hit)
    }

    /// The first switched-off node, which is the one a double click turns on.
    fn first_free_node(&self) -> Option<usize> {
        self.nodes()
            .nodes
            .iter()
            .position(|node| node.node_type.value() == EqNodeType::Off)
    }
}

/// Set a parameter as a complete one-off gesture.
///
/// Only for discrete edits. A drag must not use this: `BeginSetParameter` and `EndSetParameter`
/// tell the host an automation gesture started and finished, and hosts typically record an undo
/// entry for each one. Emitting a full gesture per mouse move floods them badly enough to stall
/// the editor. Use [`begin_gesture()`], [`set_param_value()`] and [`end_gesture()`] instead, which
/// is what a slider does.
fn set_param(cx: &mut EventContext, param_ptr: ParamPtr, plain_value: f32) {
    begin_gesture(cx, param_ptr);
    set_param_value(cx, param_ptr, plain_value);
    end_gesture(cx, param_ptr);
}

/// Tell the host an automation gesture is starting. Must be matched by [`end_gesture()`].
fn begin_gesture(cx: &mut EventContext, param_ptr: ParamPtr) {
    cx.emit(RawParamEvent::BeginSetParameter(param_ptr));
}

/// Set a parameter's value within an already-started gesture.
///
/// Unchanged values are dropped, so holding the mouse still during a drag costs nothing.
fn set_param_value(cx: &mut EventContext, param_ptr: ParamPtr, plain_value: f32) {
    // SAFETY: The pointer comes from the params object owned by the editor's data, which outlives
    //         this view.
    unsafe {
        let normalized = param_ptr.preview_normalized(plain_value);
        if param_ptr.preview_plain(normalized) == param_ptr.unmodulated_plain_value() {
            return;
        }

        cx.emit(RawParamEvent::SetParameterNormalized(param_ptr, normalized));
    }
}

/// Tell the host the gesture started by [`begin_gesture()`] has finished.
fn end_gesture(cx: &mut EventContext, param_ptr: ParamPtr) {
    cx.emit(RawParamEvent::EndSetParameter(param_ptr));
}

impl<L, LSelected, LMode> View for Analyzer<L, LSelected, LMode>
where
    L: 'static + Lens<Target = CompressorDirection>,
    LSelected: 'static + Lens<Target = Option<usize>>,
    LMode: 'static + Lens<Target = ChainMode>,
{
    fn element(&self) -> Option<&'static str> {
        Some("analyzer")
    }

    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        let direction = self.edited_direction.get(cx);
        let chain_idx = self.chain_idx;
        let chain_mode = self.chain_mode.get(cx);
        // Creating a node on one chain's graph should give it to that chain only
        let new_node_channel = chain_mode.new_node_channel(chain_idx);

        event.map(|window_event, meta| match window_event {
            WindowEvent::MouseDown(MouseButton::Left) => {
                let (x, y) = (cx.mouse().cursorx, cx.mouse().cursory);
                if let Some((node_index, node_direction)) =
                    self.node_at(cx, chain_idx, chain_mode, x, y)
                {
                    let node = self.nodes().nodes[node_index].snapshot();
                    // A node applying to both lines belongs to neither in particular, so the focus
                    // stays put; the buttons above the graph remain the way to change it.
                    if node.target != EqNodeTarget::Both && node_direction != direction {
                        cx.emit(EditorEvent::SelectDirection(node_direction));
                    }
                    let (frequency_ptr, gain_ptr) = self.drag_param_ptrs(node_index);

                    // One gesture spanning the whole drag, rather than one per mouse move
                    begin_gesture(cx, frequency_ptr);
                    begin_gesture(cx, gain_ptr);

                    self.drag = Some(NodeDrag {
                        node_index,
                        start_frequency: node.center_frequency,
                        start_gain_db: node.gain_db,
                        start_cursor: (x, y),
                    });

                    cx.emit(EditorEvent::SelectNode(Some(node_index)));
                    cx.capture();
                    cx.set_active(true);
                    meta.consume();
                } else {
                    cx.emit(EditorEvent::SelectNode(None));
                }
            }
            WindowEvent::MouseUp(MouseButton::Left) => {
                if let Some(drag) = self.drag.take() {
                    let (frequency_ptr, gain_ptr) = self.drag_param_ptrs(drag.node_index);
                    end_gesture(cx, frequency_ptr);
                    end_gesture(cx, gain_ptr);

                    cx.release();
                    cx.set_active(false);
                    meta.consume();
                }
            }
            WindowEvent::MouseMove(x, y) => {
                let Some(drag) = &self.drag else {
                    return;
                };

                let bounds = cx.bounds();
                if bounds.w <= 0.0 || bounds.h <= 0.0 {
                    return;
                }

                // Horizontal movement is a shift along the logarithmic frequency axis, vertical
                // movement a shift in decibels. Both are relative to where the drag started.
                let ln_frequency = drag.start_frequency.ln()
                    + (((x - drag.start_cursor.0) / bounds.w) * LN_FREQ_RANGE);
                let gain_db =
                    drag.start_gain_db - (((y - drag.start_cursor.1) / bounds.h) * DB_RANGE);

                let (frequency_ptr, gain_ptr) = self.drag_param_ptrs(drag.node_index);
                set_param_value(cx, frequency_ptr, ln_frequency.exp());
                set_param_value(cx, gain_ptr, gain_db);

                meta.consume();
            }
            WindowEvent::MouseScroll(_scroll_x, scroll_y) => {
                let (x, y) = (cx.mouse().cursorx, cx.mouse().cursory);
                let Some((node_index, _)) = self.node_at(cx, chain_idx, chain_mode, x, y) else {
                    return;
                };

                // Scrolling over a handle adjusts its Q. Multiplying rather than adding keeps the
                // steps feeling even across the parameter's skewed range.
                let node = &self.nodes().nodes[node_index];
                let q = node.q.value() * SCROLL_Q_FACTOR.powf(*scroll_y);
                set_param(cx, node.q.as_ptr(), q);

                meta.consume();
            }
            WindowEvent::MouseDoubleClick(MouseButton::Left) => {
                let (x, y) = (cx.mouse().cursorx, cx.mouse().cursory);

                // Double clicking a handle removes it. There is no ambiguity with creating one:
                // the hit test already tells a handle apart from empty space.
                if let Some((node_index, _)) = self.node_at(cx, chain_idx, chain_mode, x, y) {
                    let node = &self.nodes().nodes[node_index];
                    set_param(
                        cx,
                        node.node_type.as_ptr(),
                        EqNodeType::Off.to_index() as f32,
                    );

                    // The first click of this double click started a drag and selected the node
                    self.drag = None;
                    // A solo pointing at a node that no longer exists would leave every curve empty
                    if self.soloed_node() == Some(node_index) {
                        cx.emit(EditorEvent::SetNodeSolo(None));
                    }
                    cx.emit(EditorEvent::SelectNode(None));
                    cx.release();
                    cx.set_active(false);
                    meta.consume();
                    return;
                }

                // Otherwise it creates one at the pointer, if there's a spare
                let Some(node_index) = self.first_free_node() else {
                    return;
                };
                let bounds = cx.bounds();
                if bounds.w <= 0.0 {
                    return;
                }

                let t = ((x - bounds.x) / bounds.w).clamp(0.0, 1.0);
                let frequency = (LN_FREQ_RANGE_START_HZ + (LN_FREQ_RANGE * t)).exp();

                let node = &self.nodes().nodes[node_index];
                set_param(cx, node.center_frequency.as_ptr(), frequency);
                set_param(cx, node.gain_db.as_ptr(), 0.0);
                // Deleting a node only switches its type off, so a reused slot would otherwise
                // inherit whatever target it had before -- or worse, stay bypassed, which would
                // hand back a node that does nothing and give no hint as to why
                set_param(cx, node.enabled.as_ptr(), 1.0);
                set_param(
                    cx,
                    node.target.as_ptr(),
                    EqNodeTarget::Both.to_index() as f32,
                );
                set_param(
                    cx,
                    node.channel.as_ptr(),
                    new_node_channel.to_index() as f32,
                );
                set_param(
                    cx,
                    node.node_type.as_ptr(),
                    EqNodeType::Bell.to_index() as f32,
                );
                cx.emit(EditorEvent::SelectNode(Some(node_index)));

                meta.consume();
            }
            _ => {}
        });
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let bounds = cx.bounds();
        if bounds.w == 0.0 || bounds.h == 0.0 {
            return;
        }

        let edited_direction = self.edited_direction.get(cx);
        let selected_node = self.selected_node.get(cx);
        let chain_idx = self.chain_idx;
        let chain_mode = self.chain_mode.get(cx);

        // The analyzer data is pulled directly from the spectral `CompressorBank`
        let mut analyzer_data = self.analyzer_data.lock().unwrap();
        let analyzer_data = analyzer_data.read();
        let nyquist = self.sample_rate.load(Ordering::Relaxed) / 2.0;

        draw_grid(cx, canvas);

        // Both modes draw the other chain's spectrum behind this one, but for different reasons,
        // and so at different opacities.
        //
        // What makes this exact rather than approximate: the envelopes written for the analyzer
        // (`compressor_bank.rs`) are the ones the detection link has already mixed, so these two
        // lines are the actual detection signals the compressors act on.
        let other_chain_alpha = match chain_mode {
            // One graph, so without this the second chain's peaks would simply be missing from the
            // only graph there is. At a full detection link both chains detect on the same mixed
            // signal, so the two really do coincide and fading it out hides nothing.
            ChainMode::Linked => 1.0 - self.params.global.channel_link.value(),
            // Each chain already has its own graph, so this is a reference point instead: raising
            // the detection link pulls the two lines together, and that convergence is the thing
            // worth watching. Fading it out with the link would hide it at exactly the moment it
            // becomes visible, so it stays at a constant wash.
            ChainMode::Split => OTHER_CHAIN_SPECTRUM_ALPHA,
        };

        if other_chain_alpha > OTHER_CHAIN_ALPHA_FLOOR {
            for other_idx in (0..NUM_CHAINS).filter(|idx| *idx != chain_idx) {
                draw_spectrum(
                    cx,
                    canvas,
                    analyzer_data,
                    nyquist,
                    other_idx,
                    other_chain_alpha,
                );

                // A gain reduction bar says how much *that* chain is being compressed, which is
                // not this graph's subject once each chain has one of its own. Drawing it would
                // also blend on top of this chain's bars and read as a brighter reading here.
                if chain_mode == ChainMode::Linked {
                    draw_gain_reduction(
                        cx,
                        canvas,
                        analyzer_data,
                        nyquist,
                        other_idx,
                        other_chain_alpha,
                    );
                }
            }
        }

        draw_spectrum(cx, canvas, analyzer_data, nyquist, chain_idx, 1.0);
        // Read once per frame and shared by both directions, since it does not depend on which
        // compressor is being drawn
        let capture_smoothed = self.smoothed_capture();
        if let Some(smoothed) = capture_smoothed.as_deref() {
            self.draw_capture_curve(cx, canvas, chain_idx, smoothed);
        }

        self.draw_threshold_curves(
            cx,
            canvas,
            chain_idx,
            edited_direction,
            capture_smoothed.as_deref(),
        );
        self.draw_nodes(
            cx,
            canvas,
            chain_idx,
            chain_mode,
            edited_direction,
            selected_node,
        );
        draw_gain_reduction(cx, canvas, analyzer_data, nyquist, chain_idx, 1.0);

        // Last of the content, so it dims the spectrum, the curves, and the bars together. Any of
        // those still drawn at full strength inside the bypassed range would suggest something is
        // happening there.
        self.draw_lf_bypass(cx, canvas, chain_idx);

        // Draw the border last
        let border_width = cx.border_width();
        let border_color: vg::Color = cx.border_color().into();

        let mut path = vg::Path::new();
        {
            let x = bounds.x + border_width / 2.0;
            let y = bounds.y + border_width / 2.0;
            let w = bounds.w - border_width;
            let h = bounds.h - border_width;
            path.move_to(x, y);
            path.line_to(x, y + h);
            path.line_to(x + w, y + h);
            path.line_to(x + w, y);
            path.close();
        }

        let paint = vg::Paint::color(border_color).with_line_width(border_width);
        canvas.stroke_path(&path, &paint);
    }
}

impl<L, LSelected, LMode> Analyzer<L, LSelected, LMode>
where
    L: 'static + Lens<Target = CompressorDirection>,
    LSelected: 'static + Lens<Target = Option<usize>>,
    LMode: 'static + Lens<Target = ChainMode>,
{
    /// Overlays the threshold curves over the spectrum analyzer. The upwards and downwards curves
    /// can have different shapes as well as different offsets, so both are always drawn.
    /// Wash over the range this chain's low frequency bypass leaves uncompressed.
    ///
    /// The right edge is feathered across the same half octave the weights fade over, so what the
    /// eye reads as the edge of the shaded area is where compression actually comes back rather
    /// than a line drawn near it.
    fn draw_lf_bypass(&self, cx: &mut DrawContext, canvas: &mut Canvas, chain_idx: usize) {
        let corner_hz = self.params.threshold.chains[chain_idx]
            .bypass_below_hz
            .value();
        if corner_hz <= LF_BYPASS_OFF_HZ {
            return;
        }

        let bounds = cx.bounds();
        let full_hz = corner_hz * LF_BYPASS_FADE_OCTAVES.exp2();
        let corner_x = bounds.x + (bounds.w * frequency_to_t(corner_hz));
        let full_x = bounds.x + (bounds.w * frequency_to_t(full_hz));
        if full_x <= bounds.x {
            return;
        }

        let mut path = vg::Path::new();
        path.rect(bounds.x, bounds.y, full_x - bounds.x, bounds.h);

        // Solid up to the corner, then fading out across the transition
        let paint = vg::Paint::linear_gradient(
            corner_x,
            bounds.y,
            full_x,
            bounds.y,
            LF_BYPASS_WASH_COLOR,
            vg::Color::rgbaf(0.0, 0.0, 0.0, 0.0),
        );

        canvas.scissor(bounds.x, bounds.y, bounds.w, bounds.h);
        canvas.fill_path(&path, &paint);
        canvas.reset_scissor();
    }

    /// This chain's captured curve for the current stereo mode, smoothed exactly the way the
    /// compressor bank smooths it. `None` when nothing has been captured into that slot.
    ///
    /// Allocating per frame rather than keeping a scratch buffer around: `draw()` only has `&self`,
    /// this is a kilobyte, and it already takes a lock and reads a triple buffer right above.
    fn smoothed_capture(&self) -> Option<Vec<f32>> {
        let capture = &self.params.threshold.capture;
        let stereo_mode_idx = self.params.global.stereo_mode.value().to_index();

        capture.state.read(|state| {
            let slot = state.slot(stereo_mode_idx, self.chain_idx);
            if slot.is_empty() {
                return None;
            }

            let mut smoothed = vec![0.0; slot.curve_db.len()];
            smooth_into(
                &slot.curve_db,
                &mut smoothed,
                capture.smoothing_octaves.value(),
            );

            Some(smoothed)
        })
    }

    /// Draw the captured shape on its own, as it would sit at a full amount.
    ///
    /// Worth its own line rather than leaving it to show through the threshold curves: it stays
    /// visible while the amount is turned down for an A/B, it is what grows while a capture is
    /// running, and its absence is how an empty slot looks.
    fn draw_capture_curve(
        &self,
        cx: &mut DrawContext,
        canvas: &mut Canvas,
        chain_idx: usize,
        capture_smoothed: &[f32],
    ) {
        let bounds = cx.bounds();
        let num_points = 100.min(bounds.w.ceil() as usize);

        // Drawn at a full amount whatever the amount parameter says, so it answers "where would
        // this go if I pushed it all the way" even while the amount is turned down for an A/B
        let (curve_params, _, offset_db) = curve_inputs(
            &self.params,
            self.soloed_node(),
            chain_idx,
            CompressorDirection::Downwards,
        );
        let curve = Curve::new(&curve_params);
        let blend = capture_blend_for(&self.params, &curve_params, capture_smoothed, 1.0);

        let paint = vg::Paint::color(CAPTURE_CURVE_COLOR).with_line_width(cx.scale_factor() * 2.0);
        let mut path = vg::Path::new();
        for i in 0..num_points {
            let x_t = i as f32 / (num_points - 1) as f32;
            let ln_freq = LN_FREQ_RANGE_START_HZ + (LN_FREQ_RANGE * x_t);

            let y_db = curve.evaluate_ln(ln_freq) + blend.delta_ln(ln_freq) + offset_db;
            let y_t = db_to_unclamped_t(y_db);

            let physical_x_pos = bounds.x + (bounds.w * x_t);
            let physical_y_pos = bounds.y + (bounds.h * (1.0 - y_t));

            if i == 0 {
                path.move_to(physical_x_pos, physical_y_pos);
            } else {
                path.line_to(physical_x_pos, physical_y_pos);
            }
        }

        canvas.scissor(bounds.x, bounds.y, bounds.w, bounds.h);
        canvas.stroke_path(&path, &paint);
        canvas.reset_scissor();
    }

    fn draw_threshold_curves(
        &self,
        cx: &mut DrawContext,
        canvas: &mut Canvas,
        chain_idx: usize,
        edited_direction: CompressorDirection,
        capture_smoothed: Option<&[f32]>,
    ) {
        let bounds = cx.bounds();

        let line_width = cx.scale_factor() * 3.0;
        // This can be done slightly cleverer but for our purposes drawing line segments that are
        // either 1 pixel apart or that split the curve up into 100 segments (whichever results in
        // the least amount of line segments) should be sufficient
        let num_points = 100.min(bounds.w.ceil() as usize);

        for direction in [CompressorDirection::Upwards, CompressorDirection::Downwards] {
            let (curve_params, eq_params, offset_db) =
                curve_inputs(&self.params, self.soloed_node(), chain_idx, direction);
            let curve = Curve::new(&curve_params);
            let eq_curve = EqCurve::new(&eq_params, direction, chain_idx);
            // The compressor bank folds the capture into its thresholds, so leaving it out here
            // would draw a curve that no longer matches the one being heard
            let capture_blend = capture_smoothed.map(|smoothed| {
                capture_blend_for(
                    &self.params,
                    &curve_params,
                    smoothed,
                    self.params.threshold.capture.amount.value(),
                )
            });

            let color = match direction {
                CompressorDirection::Upwards => UPWARDS_THRESHOLD_CURVE_COLOR,
                CompressorDirection::Downwards => DOWNWARDS_THRESHOLD_CURVE_COLOR,
            };
            let paint = vg::Paint::color(curve_color(color, direction == edited_direction))
                .with_line_width(line_width);

            let mut path = vg::Path::new();
            for i in 0..num_points {
                let x_t = i as f32 / (num_points - 1) as f32;
                let ln_freq = LN_FREQ_RANGE_START_HZ + (LN_FREQ_RANGE * x_t);

                // Evaluating the curve results in a value in dB, which must then be mapped to the
                // same scale used in `draw_spectrum()`. The nodes are evaluated the same way the
                // compressor bank does it, so the drawn curve matches the audible one.
                let polynomial_db = curve.evaluate_ln(ln_freq);
                let capture_delta_db = capture_blend
                    .as_ref()
                    .map_or(0.0, |blend| blend.delta_ln(ln_freq));
                let y_db = polynomial_db
                    + capture_delta_db
                    + eq_curve.evaluate_db(ln_freq.exp())
                    + offset_db;
                let y_t = db_to_unclamped_t(y_db);

                let physical_x_pos = bounds.x + (bounds.w * x_t);
                // This value increases from bottom to top
                let physical_y_pos = bounds.y + (bounds.h * (1.0 - y_t));

                if i == 0 {
                    path.move_to(physical_x_pos, physical_y_pos);
                } else {
                    path.line_to(physical_x_pos, physical_y_pos);
                }
            }

            // This does a way better job at cutting off the tops and bottoms of the graph than we
            // could do by hand
            canvas.scissor(bounds.x, bounds.y, bounds.w, bounds.h);
            canvas.stroke_path(&path, &paint);
            canvas.reset_scissor();
        }
    }

    /// Draw a handle on each active EQ node, sitting on the curve it belongs to.
    ///
    /// The handles are placed by evaluating the same curve that gets drawn, so a node's handle
    /// always sits exactly on its own curve rather than near it.
    fn draw_nodes(
        &self,
        cx: &mut DrawContext,
        canvas: &mut Canvas,
        chain_idx: usize,
        chain_mode: ChainMode,
        edited_direction: CompressorDirection,
        selected_node: Option<usize>,
    ) {
        let bounds = cx.bounds();
        let scale_factor = cx.scale_factor();

        // Positions come from the same function hit testing uses, so the two cannot disagree about
        // where a node is. The focused line's handles go down last so they end up on top.
        let mut positions = self.node_positions_t(chain_idx, chain_mode);
        positions.sort_by_key(|handle| handle.direction == edited_direction);

        canvas.scissor(bounds.x, bounds.y, bounds.w, bounds.h);
        for handle in positions {
            if !(0.0..=1.0).contains(&handle.t_x) {
                continue;
            }

            let color = curve_color(
                match handle.direction {
                    CompressorDirection::Upwards => UPWARDS_THRESHOLD_CURVE_COLOR,
                    CompressorDirection::Downwards => DOWNWARDS_THRESHOLD_CURVE_COLOR,
                },
                handle.direction == edited_direction,
            );

            // The selected node is drawn larger so it's obvious which one the inspector below the
            // graph is editing
            let radius = if selected_node == Some(handle.index) {
                NODE_RADIUS * 1.6
            } else {
                NODE_RADIUS
            };

            let x = bounds.x + (bounds.w * handle.t_x);
            let y = bounds.y + (bounds.h * handle.t_y);
            let mut path = vg::Path::new();
            path.circle(x, y, radius * scale_factor);

            // A filled center with a ring around it stays legible against both the dark background
            // and the bright spectrum.
            //
            // A node that isn't deforming the curve is drawn as an outline instead. It has to stay
            // visible -- it is still there, still draggable, and it is the only way back to
            // switching it on -- but a filled handle would claim the curve passes through it, and
            // the curve has already forgotten about it.
            if handle.active {
                canvas.fill_path(&path, &vg::Paint::color(color));
                canvas.stroke_path(
                    &path,
                    &vg::Paint::color(vg::Color::rgbaf(0.05, 0.05, 0.05, 0.9))
                        .with_line_width(1.5 * scale_factor),
                );
            } else {
                canvas.fill_path(
                    &path,
                    &vg::Paint::color(vg::Color::rgbaf(0.05, 0.05, 0.05, 0.9)),
                );
                canvas.stroke_path(
                    &path,
                    &vg::Paint::color(color).with_line_width(1.5 * scale_factor),
                );
            }
        }
        canvas.reset_scissor();
    }
}

/// Compute an unclamped value based on a decibel value -80 and is mapped to 0, +20 is mapped to 1,
/// and all other values are linearly interpolated from there
#[inline]
fn db_to_unclamped_t(db_value: f32) -> f32 {
    (db_value + 80.0) / 100.0
}

/// Draw the spectrum analyzer part of the analyzer. These are drawn as vertical bars until the
/// spacing between the bars becomes less the line width, at which point it's drawn as a solid mesh
/// instead.
///
/// `alpha` scales the whole thing, so the chain that isn't being edited can be drawn behind the one
/// that is.
fn draw_spectrum(
    cx: &mut DrawContext,
    canvas: &mut Canvas,
    analyzer_data: &AnalyzerData,
    nyquist_hz: f32,
    chain_idx: usize,
    alpha: f32,
) {
    let bounds = cx.bounds();

    let line_width = cx.scale_factor() * 1.5;
    let mut text_color: vg::Color = cx.font_color().into();
    text_color.a *= alpha;
    // This is used to draw the individual bars
    let bars_paint = vg::Paint::color(text_color).with_line_width(line_width);
    // And this color is used to draw the mesh part of the spectrum. We'll create a gradient paint
    // that fades from this to `text_color` when we know the mesh's x-coordinates.
    let mut lighter_text_color = text_color;
    lighter_text_color.r = (lighter_text_color.r + 0.25) / 1.25;
    lighter_text_color.g = (lighter_text_color.g + 0.25) / 1.25;
    lighter_text_color.b = (lighter_text_color.b + 0.25) / 1.25;

    // The frequency belonging to a bin in Hz
    let bin_frequency = |bin_idx: f32| (bin_idx / analyzer_data.num_bins as f32) * nyquist_hz;
    // A `[0, 1]` value indicating at which relative x-coordinate a bin should be drawn at
    let bin_t =
        |bin_idx: f32| (bin_frequency(bin_idx).ln() - LN_FREQ_RANGE_START_HZ) / LN_FREQ_RANGE;
    // Converts a linear magnitude value in to a `[0, 1]` value where 0 is -80 dB or lower, and 1 is
    // +20 dB or higher.
    let magnitude_height = |magnitude: f32| {
        nih_debug_assert!(magnitude >= 0.0);
        let magnitude_db = nih_plug::util::gain_to_db(magnitude);
        db_to_unclamped_t(magnitude_db).clamp(0.0, 1.0)
    };

    // The first part of this drawing routing is simple. Individual bins are drawn as bars until the
    // distance between the bars approaches `mesh_start_delta_threshold`. After that the rest is
    // drawn as a solid mesh.
    let mesh_start_delta_threshold = line_width + 0.5;
    let mut mesh_bin_start_idx = analyzer_data.num_bins;
    let mut previous_physical_x_coord = bounds.x - 2.0;

    let mut bars_path = vg::Path::new();
    for (bin_idx, magnitude) in analyzer_data.envelope_followers[chain_idx]
        .iter()
        .enumerate()
        .take(analyzer_data.num_bins)
    {
        let t = bin_t(bin_idx as f32);
        if t <= 0.0 || t >= 1.0 {
            continue;
        }

        let physical_x_coord = bounds.x + (bounds.w * t);
        if physical_x_coord - previous_physical_x_coord < mesh_start_delta_threshold {
            // NOTE: We'll draw this one bar earlier because we're not stroking the solid mesh part,
            //       and otherwise there would be a weird looking gap at the left side
            mesh_bin_start_idx = bin_idx.saturating_sub(1);
            previous_physical_x_coord = physical_x_coord;
            break;
        }

        // Scale this so that 1.0/0 dBFS magnitude is at 80% of the height, the bars begin
        // at -80 dBFS, and that the scaling is linear. This is the same scaling used in
        // Diopser's spectrum analyzer.
        let height = magnitude_height(*magnitude);

        bars_path.move_to(physical_x_coord, bounds.y + (bounds.h * (1.0 - height)));
        bars_path.line_to(physical_x_coord, bounds.y + bounds.h);

        previous_physical_x_coord = physical_x_coord;
    }
    canvas.stroke_path(&bars_path, &bars_paint);

    // The mesh path starts at the bottom left, follows the top envelope of the spectrum analyzer,
    // and ends in the bottom right
    let mut mesh_path = vg::Path::new();
    let mesh_start_x_coordiante = bounds.x + (bounds.w * bin_t(mesh_bin_start_idx as f32));
    let mesh_start_y_coordinate = bounds.y + bounds.h;

    mesh_path.move_to(mesh_start_x_coordiante, mesh_start_y_coordinate);
    for (bin_idx, magnitude) in analyzer_data.envelope_followers[chain_idx]
        .iter()
        .enumerate()
        .take(analyzer_data.num_bins)
        .skip(mesh_bin_start_idx)
    {
        let t = bin_t(bin_idx as f32);
        if t <= 0.0 || t >= 1.0 {
            continue;
        }

        let physical_x_coord = bounds.x + (bounds.w * t);
        previous_physical_x_coord = physical_x_coord;
        let height = magnitude_height(*magnitude);
        if height > 0.0 {
            mesh_path.line_to(
                physical_x_coord,
                // This includes the line width, since this path is not stroked
                bounds.y + (bounds.h * (1.0 - height) - (line_width / 2.0)).max(0.0),
            );
        } else {
            mesh_path.line_to(physical_x_coord, mesh_start_y_coordinate);
        }
    }

    mesh_path.line_to(previous_physical_x_coord, mesh_start_y_coordinate);
    mesh_path.close();

    let mesh_paint = vg::Paint::linear_gradient_stops(
        mesh_start_x_coordiante,
        0.0,
        previous_physical_x_coord,
        0.0,
        [
            (0.0, lighter_text_color),
            (0.707, text_color),
            (1.0, text_color),
        ],
    )
    // NOTE:  This is very important, otherwise this looks all kinds of gnarly
    .with_anti_alias(false);
    canvas.fill_path(&mesh_path, &mesh_paint);
}

/// Fade a curve's color when it isn't the one being edited.
fn curve_color(color: vg::Color, is_edited: bool) -> vg::Color {
    if is_edited {
        color
    } else {
        let mut faded = color;
        faded.a *= UNSELECTED_CURVE_OPACITY;
        faded
    }
}

/// Draw the frequency and decibel gridlines. This goes underneath everything else so it reads as
/// background rather than as data.
fn draw_grid(cx: &mut DrawContext, canvas: &mut Canvas) {
    let bounds = cx.bounds();
    let line_width = cx.scale_factor();

    // The minor lines go down first so the labelled ones sit on top of them
    let mut minor_path = vg::Path::new();
    for frequency in MINOR_FREQUENCY_TICKS {
        let t = frequency_to_t(*frequency);
        if !(0.0..=1.0).contains(&t) {
            continue;
        }

        let x = bounds.x + (bounds.w * t);
        minor_path.move_to(x, bounds.y);
        minor_path.line_to(x, bounds.y + bounds.h);
    }
    canvas.stroke_path(
        &minor_path,
        &vg::Paint::color(MINOR_GRID_COLOR).with_line_width(line_width),
    );

    let paint = vg::Paint::color(GRID_COLOR).with_line_width(line_width);

    let mut path = vg::Path::new();
    for frequency in FREQUENCY_TICKS {
        let t = frequency_to_t(*frequency);
        if !(0.0..=1.0).contains(&t) {
            continue;
        }

        let x = bounds.x + (bounds.w * t);
        path.move_to(x, bounds.y);
        path.line_to(x, bounds.y + bounds.h);
    }

    for db in DB_TICKS {
        let t = db_to_unclamped_t(*db);
        if !(0.0..=1.0).contains(&t) {
            continue;
        }

        // This axis increases from bottom to top
        let y = bounds.y + (bounds.h * (1.0 - t));
        path.move_to(bounds.x, y);
        path.line_to(bounds.x + bounds.w, y);
    }

    canvas.stroke_path(&path, &paint);
}

/// The curve, node bank and offset for one compressor, read straight from the parameters.
///
/// These used to arrive through the analyzer's triple buffer, which meant they only refreshed
/// while `process()` was running. With the transport stopped a host may not call it at all, so
/// the curve froze at whatever it last saw, or at the default, whose zero center frequency makes
/// `ln(0)` and turns the whole curve into NaN. Reading the parameters directly keeps the curve
/// live whether or not audio is flowing.
/// Build the capture blend for one curve, matching what `CompressorBank::capture_blend()` does.
///
/// The two have to agree exactly: this one draws the curve and that one is what the audio hears.
fn capture_blend_for<'a>(
    params: &SpectralCompressorParams,
    curve_params: &CurveParams,
    capture_smoothed: &'a [f32],
    amount: f32,
) -> CaptureBlend<'a> {
    CaptureBlend::new(
        capture_smoothed,
        curve_params.center_frequency.ln(),
        params.threshold.baseline_slope(),
        amount,
        params.threshold.capture.low_frequency.value(),
        params.threshold.capture.high_frequency.value(),
    )
}

fn curve_inputs(
    params: &SpectralCompressorParams,
    soloed_node: Option<usize>,
    chain_idx: usize,
    direction: CompressorDirection,
) -> (CurveParams, EqCurveParams, f32) {
    let chain = &params.threshold.chains[chain_idx];
    let curve = match direction {
        CompressorDirection::Upwards => &chain.upwards,
        CompressorDirection::Downwards => &chain.downwards,
    };

    (
        params.threshold.curve_params(curve),
        params.threshold.eq.snapshot(soloed_node),
        curve.threshold_offset_db.value(),
    )
}

/// Overlays the gain reduction display over the spectrum analyzer.
///
/// `alpha` scales it the same way it scales the spectrum, so a faded chain fades as a whole.
fn draw_gain_reduction(
    cx: &mut DrawContext,
    canvas: &mut Canvas,
    analyzer_data: &AnalyzerData,
    nyquist_hz: f32,
    chain_idx: usize,
    alpha: f32,
) {
    let bounds = cx.bounds();

    // As with the above, anti aliasing only causes issues
    let mut color = GR_BAR_OVERLAY_COLOR;
    color.a *= alpha;
    let paint = vg::Paint::color(color).with_anti_alias(false);

    let bin_frequency = |bin_idx: f32| (bin_idx / analyzer_data.num_bins as f32) * nyquist_hz;

    let mut path = vg::Path::new();
    for (bin_idx, gain_difference_db) in analyzer_data.gain_difference_db[chain_idx]
        .iter()
        .enumerate()
        .take(analyzer_data.num_bins)
    {
        // Avoid drawing tiny slivers for low gain reduction values
        if gain_difference_db.abs() < 0.2 {
            continue;
        }

        // The gain reduction bars are drawn with the width of the bin, centered on the bin's center
        // frequency. The first and the last bin are extended to the edges of the graph because
        // otherwise it looks weird.
        let t_start = if bin_idx == 0 {
            0.0
        } else {
            let gr_start_ln_frequency = bin_frequency(bin_idx as f32 - 0.5).ln();
            (gr_start_ln_frequency - LN_FREQ_RANGE_START_HZ) / LN_FREQ_RANGE
        };
        let t_end = if bin_idx == analyzer_data.num_bins - 1 {
            1.0
        } else {
            let gr_end_ln_frequency = bin_frequency(bin_idx as f32 + 0.5).ln();
            (gr_end_ln_frequency - LN_FREQ_RANGE_START_HZ) / LN_FREQ_RANGE
        };
        if t_end < 0.0 || t_start > 1.0 {
            continue;
        }

        let (t_start, t_end) = (t_start.max(0.0), t_end.min(1.0));

        // For the bar's height we'll draw 0 dB of gain reduction as a flat line (except we
        // don't actually draw 0 dBs of GR because it looks glitchy, but that's besides the
        // point). 40 dB of gain reduction causes the bar to be drawn from the center all
        // the way to the bottom of the spectrum analyzer. 40 dB of additional gain causes
        // the bar to be drawn from the center all the way to the top of the graph.
        // NOTE: Y-coordinates go from top to bottom, hence the minus
        let t_y = ((-gain_difference_db + 40.0) / 80.0).clamp(0.0, 1.0);

        path.move_to(bounds.x + (bounds.w * t_start), bounds.y + (bounds.h * 0.5));
        path.line_to(bounds.x + (bounds.w * t_end), bounds.y + (bounds.h * 0.5));
        path.line_to(bounds.x + (bounds.w * t_end), bounds.y + (bounds.h * t_y));
        path.line_to(bounds.x + (bounds.w * t_start), bounds.y + (bounds.h * t_y));
        path.close();
    }

    canvas
        .global_composite_blend_func(vg::BlendFactor::DstAlpha, vg::BlendFactor::OneMinusDstColor);
    canvas.fill_path(&path, &paint);
    canvas.global_composite_blend_func(vg::BlendFactor::One, vg::BlendFactor::OneMinusSrcAlpha);
}
