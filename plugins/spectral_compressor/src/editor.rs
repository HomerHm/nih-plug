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
use nih_plug::prelude::*;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::*;
use nih_plug_vizia::{assets, create_vizia_editor, ViziaState, ViziaTheming};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use self::analyzer::{format_frequency, frequency_to_t, Analyzer, FREQUENCY_TICKS};
use self::param_link::{param_ptr_by_id, param_ptr_pairs, ParamLink, ParamLinkEvent};
use crate::analyzer::AnalyzerData;
use crate::compressor_bank::{ThresholdCurveParams, NUM_CHAINS};
use crate::eq_curve::{CompressorDirection, EqNodeChannel, EqNodeParams, EqNodeType, MAX_EQ_NODES};
use crate::{SoloMode, StereoMode};
use crate::{SpectralCompressor, SpectralCompressorParams};
use crossbeam::atomic::AtomicCell;

mod analyzer;
mod param_link;

/// The GUI's width, in logical pixels. Wide enough for the four control columns below the
/// analyzer to sit side by side.
const GUI_WIDTH: u32 = 1360;
/// The GUI's height with one analyzer, in logical pixels.
///
/// The controls take exactly as much vertical space as they need and the analyzer absorbs whatever
/// is left over, so this number only sets how much room the analyzer gets. Growing a control
/// column can therefore never truncate it -- it just eats into the analyzer.
const GUI_HEIGHT: u32 = 900;
/// The GUI's height in [`ChainMode::Split`], which stacks a second analyzer.
///
/// The extra room goes to the second graph rather than to the controls, so both end up roughly the
/// height a single one has in [`GUI_HEIGHT`].
const SPLIT_GUI_HEIGHT: u32 = 1230;
// I couldn't get `LayoutType::Grid` to work as expected, so we'll fake a 4x4 grid with
// hardcoded column widths
const COLUMN_WIDTH: Units = Pixels(330.0);

const DARKER_GRAY: Color = Color::rgb(0x69, 0x69, 0x69);

/// Where the "Mod by HomerHm" label links to. The plugin title itself still links to
/// [`SpectralCompressor::URL`], which points at the original upstream project.
const MOD_URL: &str = "https://github.com/HomerHm/nih-plug";

/// Open `url` in the user's browser, ignoring failures. Mirrors what the title label does.
fn open_url(url: &str) {
    // FIXME: On Windows this blocks, and while this is blocking a timer may proc which causes
    //        the window state to be mutably borrowed again, resulting in a panic. This needs to
    //        be fixed in baseview first.
    if cfg!(not(windows)) {
        let result = open::that(url);
        if cfg!(debug_assertions) && result.is_err() {
            nih_debug_assert_failure!("Failed to open web browser: {:?}", result);
        }
    }
}

// NOTE: This is written out rather than derived because the derive macro emits an unqualified
//       `impl Data for ...`, and `Data` in this module resolves to the editor's own struct.
impl nih_plug_vizia::vizia::prelude::Data for CompressorDirection {
    fn same(&self, other: &Self) -> bool {
        self == other
    }
}

/// Whether the two chains are kept identical or edited apart.
///
/// The two chains carry two *different signals* (left and right, or mid and side), so their spectra
/// and gain reduction genuinely differ. Drawing both on one graph would be a lie, which is why
/// `Split` stacks a second analyzer rather than adding more lines to the first one. Upwards and
/// downwards do share a graph, because those really do act on the same signal.
///
/// `Linked` is the default and the common case. Switching to it is destructive on purpose: it
/// merges the chains outright rather than leaving a difference that the single graph cannot show.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChainMode {
    // These serialization names are hardcoded so the variants can be renamed later without
    // breaking compatibility with saved projects
    #[default]
    #[serde(rename = "linked")]
    Linked,
    #[serde(rename = "split")]
    Split,
}

impl nih_plug_vizia::vizia::prelude::Data for ChainMode {
    fn same(&self, other: &Self) -> bool {
        self == other
    }
}

impl ChainMode {
    /// Whether edits should be mirrored onto the other chain.
    fn mirrors(self) -> bool {
        self == ChainMode::Linked
    }

    /// The channel a node created on `chain_idx`'s graph belongs to. In `Split` each graph is one
    /// chain, so a node drawn on it belongs to that chain alone.
    fn new_node_channel(self, chain_idx: usize) -> EqNodeChannel {
        match (self, chain_idx) {
            (ChainMode::Linked, _) => EqNodeChannel::Both,
            (ChainMode::Split, 0) => EqNodeChannel::First,
            (ChainMode::Split, _) => EqNodeChannel::Second,
        }
    }

    fn name(self) -> &'static str {
        match self {
            ChainMode::Linked => "Linked",
            ChainMode::Split => "Split",
        }
    }
}

/// One chain's name, which depends on how the stereo mode splits the signal up.
fn chain_name(chain_idx: usize, stereo_mode: &StereoMode) -> &'static str {
    match (chain_idx, stereo_mode) {
        (0, StereoMode::MidSide) => "Mid",
        (0, _) => "Left",
        (_, StereoMode::MidSide) => "Side",
        (_, _) => "Right",
    }
}

/// Events the editor handles itself, rather than passing on to the parameters.
pub enum EditorEvent {
    /// Switch between keeping the two chains identical and editing them apart.
    SetChainMode(ChainMode),
    /// Listen to one chain on its own. Momentary, and deliberately not a parameter.
    SetSolo(SoloMode),
    /// Switch which curve within that chain the analyzer edits.
    SelectDirection(CompressorDirection),
    /// Select a node for the inspector below the analyzer, or clear it.
    SelectNode(Option<usize>),
}

#[derive(Clone, Lens)]
pub struct Data {
    pub(crate) params: Arc<SpectralCompressorParams>,

    pub(crate) analyzer_data: Arc<Mutex<triple_buffer::Output<AnalyzerData>>>,
    /// Used by the analyzer to determine which FFT bins belong to which frequencies.
    pub(crate) sample_rate: Arc<AtomicF32>,

    /// Which curve the analyzer edits. Editor state rather than a parameter, since it changes
    /// nothing about the sound.
    pub(crate) edited_direction: CompressorDirection,
    /// Whether the chains are linked or edited apart.
    ///
    /// This is deliberately a plain copy of what [`Self::chain_mode_cell`] holds. Storing an
    /// `AtomicCell` alone would leave the `Arc` itself unchanged when the mode is switched, so
    /// nothing bound to it through a lens would ever be told to update.
    pub(crate) chain_mode: ChainMode,
    /// The same value as [`Self::chain_mode`], in the form the window's `size_fn` can read: that
    /// closure runs outside the editor, where the model is not reachable.
    pub(crate) chain_mode_cell: Arc<AtomicCell<ChainMode>>,
    /// Which chain is being listened to on its own, shared with the audio thread. Not a parameter,
    /// so it is never saved with a project: leaving a solo engaged should not outlive the session.
    pub(crate) solo: Arc<AtomicCell<SoloMode>>,
    /// The node the inspector below the analyzer is editing. There is one shared bank of nodes,
    /// each carrying which chain it belongs to, so an index identifies a node on its own.
    pub(crate) selected_node: Option<usize>,
}

impl Model for Data {
    fn event(&mut self, _cx: &mut EventContext, event: &mut Event) {
        event.map(|editor_event, _| match editor_event {
            EditorEvent::SetChainMode(mode) => {
                self.chain_mode = *mode;
                self.chain_mode_cell.store(*mode);
                self.selected_node = None;
            }
            EditorEvent::SetSolo(solo) => self.solo.store(*solo),
            EditorEvent::SelectDirection(direction) => self.edited_direction = *direction,
            EditorEvent::SelectNode(node_index) => {
                self.selected_node = *node_index;
            }
        });
    }
}

/// The editor's window size, which grows to fit the second analyzer in [`ChainMode::Split`].
///
/// This reads the mode through an `AtomicCell` because it is called from outside the editor, and
/// possibly while it is closed.
pub(crate) fn default_state(chain_mode: Arc<AtomicCell<ChainMode>>) -> Arc<ViziaState> {
    ViziaState::new(move || match chain_mode.load() {
        ChainMode::Linked => (GUI_WIDTH, GUI_HEIGHT),
        ChainMode::Split => (GUI_WIDTH, SPLIT_GUI_HEIGHT),
    })
}

pub(crate) fn create(editor_state: Arc<ViziaState>, editor_data: Data) -> Option<Box<dyn Editor>> {
    create_vizia_editor(editor_state, ViziaTheming::Custom, move |cx, _| {
        assets::register_noto_sans_light(cx);
        assets::register_noto_sans_thin(cx);

        if let Err(err) = cx.add_stylesheet(include_style!("src/editor/theme.css")) {
            nih_error!("Failed to load stylesheet: {err:?}")
        }

        editor_data.clone().build(cx);

        // Both links wrap the whole editor rather than just the controls they belong to, because
        // nodes can also be edited by dragging them on the analyzer. Anything narrower would
        // leave those edits unmirrored. Each view only ever acts on its own pairs, so nesting
        // them is harmless.
        param_links(cx, |cx| {
            VStack::new(cx, |cx| {
                title_bar(cx);
                // `Stretch` here is what keeps the controls below from ever being clipped: they
                // take their natural height first and the analyzer gets the remainder.
                analyzer(cx);
                controls(cx);
            })
            .row_between(Pixels(10.0));
        });

        ResizeHandle::new(cx);
    })
}

/// Wrap `content` in the view that keeps the two compressors' curve shapes in step.
///
/// This wraps the whole editor rather than just the columns it belongs to, because the curve can
/// also be reshaped by dragging on the analyzer, and anything narrower would leave those edits
/// unmirrored.
fn param_links(cx: &mut Context, content: impl FnOnce(&mut Context)) {
    let params = Data::params.get(cx);

    // Downwards leads, so switching the link on pulls the upwards curve onto the downwards one.
    // Every chain is paired, not just the one on screen, so switching chains can't reveal a pair
    // that quietly drifted apart while the link was on.
    let curve_pairs = params
        .threshold
        .chains
        .iter()
        .flat_map(|chain| param_ptr_pairs(&chain.downwards, &chain.upwards))
        .collect();

    let curve_link_ptr = param_ptr_by_id(&params.threshold, "thresh_link");
    let curve_linked = {
        let params = params.clone();
        move |_: &EventContext| params.threshold.slope_curve_link.value()
    };

    // Only the curve shapes are paired. The nodes are one shared bank now, each carrying which
    // chain it belongs to, so mirroring them would be duplicating a node onto itself.
    let chain_pairs = param_ptr_pairs(&params.threshold.chains[0], &params.threshold.chains[1]);
    // There is no link parameter here: which chains an edit reaches is editor state, so switching
    // to `Both` deliberately overwrites nothing. The two converge the moment something is touched.
    let chains_mirrored = |cx: &EventContext| {
        cx.data::<Data>()
            .is_some_and(|data| data.chain_mode.mirrors())
    };

    ParamLink::new(
        cx,
        curve_linked,
        Some(curve_link_ptr),
        curve_pairs,
        move |cx| {
            ParamLink::new(cx, chains_mirrored, None, chain_pairs, content)
                .width(Stretch(1.0))
                .height(Stretch(1.0));
        },
    )
    .width(Stretch(1.0))
    .height(Stretch(1.0));
}

fn title_bar(cx: &mut Context) {
    HStack::new(cx, |cx| {
        Label::new(cx, "Spectral Compressor")
            .font_family(vec![FamilyOwned::Name(String::from(assets::NOTO_SANS))])
            .font_weight(FontWeightKeyword::Thin)
            .font_size(30.0)
            // Clicking the title opens the original project's page
            .on_mouse_down(|_, _| open_url(SpectralCompressor::URL));
        Label::new(cx, SpectralCompressor::VERSION)
            .color(DARKER_GRAY)
            .top(Stretch(1.0))
            .bottom(Pixels(4.0))
            .left(Pixels(2.0));
        // GPL requires modified versions to be marked as such. This also doubles as the
        // link to this fork, so the title above can keep pointing at the original.
        Label::new(cx, "Mod by HomerHm")
            .color(DARKER_GRAY)
            .font_size(11.0)
            .top(Stretch(1.0))
            .bottom(Pixels(5.0))
            .left(Pixels(8.0))
            .on_mouse_down(|_, _| open_url(MOD_URL));
    })
    .height(Pixels(30.0))
    .left(Pixels(12.0))
    .top(Pixels(10.0));
}

fn analyzer(cx: &mut Context) {
    VStack::new(cx, |cx| {
        analyzer_toolbar(cx);

        // One signal per graph. The two chains carry different signals, so their spectra and gain
        // reduction genuinely differ and drawing both on one graph would be a lie; upwards and
        // downwards do share one, because those really do act on the same signal.
        //
        // Both graphs are always built and the second one hidden while linked, rather than
        // swapping them inside a `Binding`: a binding's entity is ignored by the layout, so its
        // children end up stacked on top of each other at the origin.
        for chain_idx in 0..NUM_CHAINS {
            analyzer_graph(cx, chain_idx);
        }

        node_inspector(cx);
    })
    .height(Stretch(1.0))
    .left(Pixels(12.0))
    .right(Pixels(12.0));
}

/// One chain's graph, with its own frequency scale underneath.
///
/// The chain is baked in rather than read from a lens, so every mouse event on this graph acts on
/// the chain it draws. That is what makes clicking either graph edit the chain under the pointer.
fn analyzer_graph(cx: &mut Context, chain_idx: usize) {
    VStack::new(cx, move |cx| {
        // Which chain a graph shows only needs saying when there are two of them
        Label::new(
            cx,
            Data::params.map(move |p| chain_name(chain_idx, &p.global.stereo_mode.value())),
        )
        .font_size(12.0)
        .color(DARKER_GRAY)
        .height(Pixels(16.0))
        .display(Data::chain_mode.map(|mode| *mode == ChainMode::Split));

        Analyzer::new(
            cx,
            Data::analyzer_data,
            Data::sample_rate,
            Data::params,
            Data::edited_direction,
            Data::selected_node,
            chain_idx,
            // Not baked in: switching modes changes what a new node is given, and whether the
            // other chain is drawn behind this one, without the graph being rebuilt
            Data::chain_mode,
        )
        // Soaks up all vertical space the controls below don't need
        .height(Stretch(1.0));

        frequency_scale(cx);
    })
    .height(Stretch(1.0))
    // The second graph only exists once the chains are edited apart
    .display(Data::chain_mode.map(move |mode| chain_idx == 0 || *mode == ChainMode::Split));
}

/// The buttons above the graphs: whether the chains are linked, which curve the mouse edits, and
/// soloing one chain to hear it on its own.
///
/// These act on every graph below them, so there is one toolbar rather than one per graph.
fn analyzer_toolbar(cx: &mut Context) {
    HStack::new(cx, |cx| {
        for mode in [ChainMode::Linked, ChainMode::Split] {
            Button::new(
                cx,
                move |cx| {
                    // Switching to linked merges the chains, so re-clicking the engaged button
                    // would throw away edits for no reason
                    if cx
                        .data::<Data>()
                        .is_some_and(|data| data.chain_mode == mode)
                    {
                        return;
                    }

                    if mode == ChainMode::Linked {
                        merge_chains(cx);
                    }
                    cx.emit(EditorEvent::SetChainMode(mode));
                    // `default_state()`'s size function reads the mode, so this is what actually
                    // resizes the window to fit the second graph
                    cx.emit(GuiContextEvent::Resize);
                },
                move |cx| Label::new(cx, mode.name()).font_size(12.0),
            )
            .checked(Data::chain_mode.map(move |current| *current == mode))
            .class("direction-button");
        }

        Element::new(cx).width(Pixels(12.0));
        solo_buttons(cx);
        Element::new(cx).width(Pixels(16.0));

        for direction in [CompressorDirection::Upwards, CompressorDirection::Downwards] {
            Button::new(
                cx,
                move |cx| cx.emit(EditorEvent::SelectDirection(direction)),
                move |cx| Label::new(cx, direction.name()).font_size(12.0),
            )
            .checked(Data::edited_direction.map(move |edited| *edited == direction))
            .class("direction-button");
        }
    })
    .height(Pixels(22.0))
    .col_between(Pixels(4.0))
    .bottom(Pixels(4.0));
}

/// The buttons for listening to one chain on its own.
///
/// Clicking the engaged one releases it. These are styled as a warning because a solo left engaged
/// is a silent way to make everything downstream sound wrong.
fn solo_buttons(cx: &mut Context) {
    for (solo, label) in [(SoloMode::First, "S1"), (SoloMode::Second, "S2")] {
        Button::new(
            cx,
            move |cx| {
                let engaged = cx
                    .data::<Data>()
                    .is_some_and(|data| data.solo.load() == solo);
                cx.emit(EditorEvent::SetSolo(if engaged {
                    SoloMode::Off
                } else {
                    solo
                }));
            },
            move |cx| Label::new(cx, label).font_size(12.0),
        )
        .checked(Data::solo.map(move |current| current.load() == solo))
        .class("solo-button");
    }
}

/// The frequency labels underneath the analyzer.
///
/// Text can't be drawn onto the canvas from outside vizia, so these are real labels positioned
/// over the analyzer's width. They share [`analyzer::frequency_to_t`] with the gridlines, so the
/// two cannot drift apart.
fn frequency_scale(cx: &mut Context) {
    // Wide enough for a label to stay centred on its gridline without being clipped
    const LABEL_WIDTH_PCT: f32 = 10.0;

    ZStack::new(cx, |cx| {
        for frequency in FREQUENCY_TICKS {
            let t = frequency_to_t(*frequency);
            if !(0.0..=1.0).contains(&t) {
                continue;
            }

            ZStack::new(cx, |cx| {
                Label::new(cx, &format_frequency(*frequency))
                    .font_size(11.0)
                    .color(DARKER_GRAY);
            })
            .left(Percentage((t * 100.0) - (LABEL_WIDTH_PCT / 2.0)))
            .width(Percentage(LABEL_WIDTH_PCT))
            .child_left(Stretch(1.0))
            .child_right(Stretch(1.0));
        }
    })
    .height(Pixels(16.0))
    .top(Pixels(2.0));
}

fn controls(cx: &mut Context) {
    HStack::new(cx, |cx| {
        make_column(cx, "Globals", |cx| {
            GenericUi::new(cx, Data::params.map(|p| p.global.clone()));
        });

        make_column(cx, "Threshold", |cx| {
            GenericUi::new(cx, Data::params.map(|p| p.threshold.clone()));
        });

        // The two compressor columns own the slope and curve of their own threshold curve, so the
        // toggle that links those sits directly above the pair rather than off in another column
        compressor_columns(cx);
    })
    // Auto means this row is exactly as tall as its tallest column, so adding parameters makes
    // the window's analyzer shrink rather than pushing controls out of view
    .height(Auto)
    .bottom(Pixels(12.0))
    .child_left(Stretch(1.0))
    .child_right(Stretch(1.0));
}

/// The Upwards and Downwards columns, with the toggle that links their curve shapes above them.
fn compressor_columns(cx: &mut Context) {
    VStack::new(cx, |cx| {
        ParamButton::new(cx, Data::params, |p| &p.threshold.slope_curve_link)
            .with_label("Thresh Curve Link")
            .left(Stretch(1.0))
            .right(Stretch(1.0))
            .bottom(Pixels(4.0));

        HStack::new(cx, |cx| {
            compressor_column(cx, CompressorDirection::Upwards);
            compressor_column(cx, CompressorDirection::Downwards);
        })
        .height(Auto);
    })
    .width(Pixels(660.0))
    .height(Auto);
}

/// One compressor's column. `direction` is both the heading and the parameter name prefix that
/// gets stripped from each row's label.
fn compressor_column(cx: &mut Context, direction: CompressorDirection) {
    make_column(cx, direction.name(), move |cx| {
        // Every chain's rows are built and the second set hidden while linked, rather than
        // rebuilding them inside a `Binding`: a binding's entity is ignored by the layout, so its
        // children end up stacked at the origin.
        for chain_idx in 0..NUM_CHAINS {
            let params = Data::params;

            VStack::new(cx, move |cx| {
                // Which chain a set belongs to only needs saying once there are two of them
                Label::new(
                    cx,
                    Data::params.map(move |p| chain_name(chain_idx, &p.global.stereo_mode.value())),
                )
                .font_size(11.0)
                .color(DARKER_GRAY)
                .left(Stretch(1.0))
                .right(Pixels(7.0))
                .display(Data::chain_mode.map(|mode| *mode == ChainMode::Split));

                labelled_row(cx, "Thresh Center", move |cx| {
                    ParamSlider::new(cx, params, move |p| {
                        &chain_curve(p, chain_idx, direction).center_frequency
                    });
                });
                labelled_row(cx, "Thresh Slope", move |cx| {
                    ParamSlider::new(cx, params, move |p| {
                        &chain_curve(p, chain_idx, direction).curve_slope
                    });
                });
                labelled_row(cx, "Thresh Curve", move |cx| {
                    ParamSlider::new(cx, params, move |p| {
                        &chain_curve(p, chain_idx, direction).curve_curve
                    });
                });
                labelled_row(cx, "Offset", move |cx| {
                    ParamSlider::new(cx, params, move |p| {
                        &chain_curve(p, chain_idx, direction).threshold_offset_db
                    });
                });
            })
            .height(Auto)
            // Both sets have to be reachable once the chains are edited apart. Unlike the nodes,
            // these can't be dragged on the graph, so hiding one would put it out of reach.
            .display(Data::chain_mode.map(move |mode| chain_idx == 0 || *mode == ChainMode::Split));
        }

        // We don't want to show the 'Upwards'/'Downwards' prefix here, but it should still be in
        // the parameter name so the parameter list makes sense
        let compressor_params = compressor_params_lens(direction);
        let strip_prefix = format!("{} ", direction.name());
        GenericUi::new_custom(cx, compressor_params, move |cx, param_ptr| {
            HStack::new(cx, |cx| {
                Label::new(
                    cx,
                    unsafe { param_ptr.name() }
                        .strip_prefix(&strip_prefix)
                        .expect("Expected parameter name prefix, this is a bug"),
                )
                .class("label");

                GenericUi::draw_widget(cx, compressor_params, param_ptr);
            })
            .class("row");
        });
    });
}

fn make_column(cx: &mut Context, title: &str, contents: impl FnOnce(&mut Context)) {
    VStack::new(cx, |cx| {
        Label::new(cx, title)
            .font_family(vec![FamilyOwned::Name(String::from(assets::NOTO_SANS))])
            .font_weight(FontWeightKeyword::Thin)
            .font_size(23.0)
            .left(Stretch(1.0))
            // This should align nicely with the right edge of the slider
            .right(Pixels(7.0))
            .bottom(Pixels(-10.0));

        contents(cx);
    })
    .width(COLUMN_WIDTH)
    .height(Auto);
}

/// One chain's threshold curve for one compressor.
fn chain_curve(
    params: &SpectralCompressorParams,
    chain_idx: usize,
    direction: CompressorDirection,
) -> &ThresholdCurveParams {
    let chain = &params.threshold.chains[chain_idx];
    match direction {
        CompressorDirection::Upwards => &chain.upwards,
        CompressorDirection::Downwards => &chain.downwards,
    }
}

/// One row of the generic-UI-styled parameter list, with a label on the left.
fn labelled_row(cx: &mut Context, label: &'static str, widget: impl FnOnce(&mut Context)) {
    HStack::new(cx, |cx| {
        Label::new(cx, label).class("label");
        widget(cx);
    })
    // Not `row`: that is only styled inside a generic UI, and this is not in one
    .class("param-row");
}

/// A lens to one compressor's parameters, picked by the same name used for its heading.
fn compressor_params_lens(
    direction: CompressorDirection,
) -> impl Lens<Target = Arc<crate::compressor_bank::CompressorParams>> {
    Data::params.map(move |p| match direction {
        CompressorDirection::Upwards => p.compressors.upwards.clone(),
        CompressorDirection::Downwards => p.compressors.downwards.clone(),
    })
}

/// The controls for the selected node, shown underneath the analyzer.
///
/// Everything here is something a drag can't express: the node's shape, which compressors it
/// applies to, and removing it. It's a fixed row rather than a context menu because vizia's popups
/// need a good deal of the default stylesheet that `ViziaTheming::Custom` doesn't load, and this
/// costs far less to get working. It also puts typed entry within reach, since every slider here
/// already takes a keyboard value on a double click.
fn node_inspector(cx: &mut Context) {
    HStack::new(cx, |cx| {
        Label::new(
            cx,
            "Double click the graph to add a node, or click one to edit it",
        )
        .color(DARKER_GRAY)
        .font_size(12.0)
        .top(Stretch(1.0))
        .bottom(Stretch(1.0))
        .display(Data::selected_node.map(|selected| selected.is_none()));

        // As with the compressor columns, every row is built and all but the selected one hidden.
        // A `Binding` marks its entity as ignored by the layout, so using one as a container
        // leaves its children stacked on top of each other at the origin.
        for index in 0..MAX_EQ_NODES {
            HStack::new(cx, move |cx| {
                Label::new(cx, &format!("Node {}", index + 1))
                    .width(Pixels(60.0))
                    .top(Stretch(1.0))
                    .bottom(Stretch(1.0));

                let params = Data::params;
                ParamSlider::new(cx, params, move |p| &p.threshold.eq.nodes[index].node_type)
                    .set_style(ParamSliderStyle::FromLeft)
                    .width(Pixels(150.0));
                ParamSlider::new(cx, params, move |p| &p.threshold.eq.nodes[index].target)
                    .set_style(ParamSliderStyle::FromLeft)
                    .width(Pixels(110.0));
                ParamSlider::new(cx, params, move |p| &p.threshold.eq.nodes[index].channel)
                    .set_style(ParamSliderStyle::FromLeft)
                    .width(Pixels(110.0));
                ParamSlider::new(cx, params, move |p| {
                    &p.threshold.eq.nodes[index].center_frequency
                })
                .width(Pixels(110.0));
                ParamSlider::new(cx, params, move |p| &p.threshold.eq.nodes[index].gain_db)
                    .width(Pixels(110.0));
                ParamSlider::new(cx, params, move |p| &p.threshold.eq.nodes[index].q)
                    .width(Pixels(90.0));

                Button::new(
                    cx,
                    move |cx| {
                        // Switching a node off is what removing it means. Its other parameters
                        // stay put, so the host's automation lanes don't shift around
                        // underneath the user.
                        let off_index = EqNodeType::variants()
                            .iter()
                            .position(|name| *name == "Off")
                            .expect("EqNodeType has no Off variant, this is a bug");
                        set_node_variant(
                            cx,
                            index,
                            |node| node.node_type.as_ptr(),
                            off_index,
                            EqNodeType::variants().len(),
                        );
                        cx.emit(EditorEvent::SelectNode(None));
                    },
                    |cx| Label::new(cx, "Delete").font_size(12.0),
                )
                .class("direction-button");
            })
            .height(Auto)
            .col_between(Pixels(4.0))
            .display(Data::selected_node.map(move |selected| *selected == Some(index)));
        }
    })
    .height(Pixels(24.0))
    .child_left(Stretch(1.0))
    .child_right(Stretch(1.0))
    .top(Pixels(4.0));
}

/// Fold the second chain onto the first, so that switching to [`ChainMode::Linked`] really does
/// leave one curve behind.
///
/// This throws away whatever the second chain was given while the two were apart, and that is the
/// point. A single graph cannot show a difference between the chains, so leaving one in place would
/// mean a curve that keeps acting on the audio while the editor claims it isn't there. Linked means
/// linked.
///
/// The cost is that this is not undoable from inside the plugin -- a misclick means putting the
/// nodes back by hand. See the undo/redo entry in CLAUDE.md's future work.
fn merge_chains(cx: &mut EventContext) {
    // The curve shapes are already paired up by `param_links()`, which knows how to copy one side
    // onto the other. Editor state has no parameter for that view to watch, hence the explicit nudge.
    cx.emit(ParamLinkEvent::AdoptLeaderValues);

    let Some(data) = cx.data::<Data>() else {
        return;
    };
    // Cloned so the loop below can borrow the parameters while still handing out `cx` mutably
    let params = data.params.clone();

    // A node given to one chain would otherwise stay on that chain, invisibly, since one graph has
    // no way to say "this one only applies to the left"
    let both_index = EqNodeChannel::Both.to_index();
    for index in 0..MAX_EQ_NODES {
        let node = &params.threshold.eq.nodes[index];
        if node.node_type.value() == EqNodeType::Off || node.channel.value() == EqNodeChannel::Both
        {
            continue;
        }

        set_node_variant(
            cx,
            index,
            |node| node.channel.as_ptr(),
            both_index,
            EqNodeChannel::variants().len(),
        );
    }
}

/// Set one of a node's enum parameters to the variant at `index`.
fn set_node_variant(
    cx: &mut EventContext,
    node_index: usize,
    param_ptr: fn(&EqNodeParams) -> ParamPtr,
    index: usize,
    variant_count: usize,
) {
    let Some(data) = cx.data::<Data>() else {
        return;
    };
    let ptr = param_ptr(&data.params.threshold.eq.nodes[node_index]);

    // An enum parameter's normalized range is divided evenly between its variants
    let normalized = index as f32 / (variant_count.saturating_sub(1).max(1)) as f32;

    cx.emit(RawParamEvent::BeginSetParameter(ptr));
    cx.emit(RawParamEvent::SetParameterNormalized(ptr, normalized));
    cx.emit(RawParamEvent::EndSetParameter(ptr));
}
