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
use std::sync::{Arc, Mutex};

use self::analyzer::{format_frequency, frequency_to_t, Analyzer, FREQUENCY_TICKS};
use self::param_link::{param_ptr_by_id, param_ptr_pairs, ParamLink};
use crate::analyzer::AnalyzerData;
use crate::compressor_bank::{ThresholdCurveParams, NUM_CHAINS};
use crate::eq_curve::{CompressorDirection, EqNodeParams, EqNodeType, MAX_EQ_NODES};
use crate::StereoMode;
use crate::{SpectralCompressor, SpectralCompressorParams};

mod analyzer;
mod param_link;

/// The GUI's width, in logical pixels. Wide enough for the four control columns below the
/// analyzer to sit side by side.
const GUI_WIDTH: u32 = 1360;
/// The GUI's height, in logical pixels.
///
/// The controls take exactly as much vertical space as they need and the analyzer absorbs whatever
/// is left over, so this number only sets how much room the analyzer gets. Growing a control
/// column can therefore never truncate it -- it just eats into the analyzer.
const GUI_HEIGHT: u32 = 900;
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

/// Which compressor's threshold curve the analyzer is currently editing.
///
/// Both curves are always drawn, but only this one responds to the mouse and is drawn at full
/// opacity. Stage 4 will add a second axis for the processing chain; this is deliberately a
/// selection rather than a pile of always-visible controls so that adding that axis doesn't
/// multiply the number of on-screen buttons.
pub type EditedDirection = CompressorDirection;

// NOTE: This is written out rather than derived because the derive macro emits an unqualified
//       `impl Data for ...`, and `Data` in this module resolves to the editor's own struct.
impl nih_plug_vizia::vizia::prelude::Data for CompressorDirection {
    fn same(&self, other: &Self) -> bool {
        self == other
    }
}

/// Events the editor handles itself, rather than passing on to the parameters.
pub enum EditorEvent {
    /// Switch which chain the analyzer edits.
    SelectChain(usize),
    /// Switch which curve within that chain the analyzer edits.
    SelectDirection(CompressorDirection),
    /// Select a node in the current chain for the inspector below the analyzer, or clear it.
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
    /// Which chain the analyzer edits. What it means depends on the stereo mode.
    pub(crate) edited_chain: usize,
    /// The node the inspector below the analyzer is editing, as `(chain, node)`. The chain is part
    /// of it because a node index only means something within one chain's bank.
    pub(crate) selected_node: Option<(usize, usize)>,
}

impl Model for Data {
    fn event(&mut self, _cx: &mut EventContext, event: &mut Event) {
        event.map(|editor_event, _| match editor_event {
            EditorEvent::SelectChain(chain_idx) => {
                self.edited_chain = *chain_idx;
                self.selected_node = None;
            }
            EditorEvent::SelectDirection(direction) => self.edited_direction = *direction,
            EditorEvent::SelectNode(node_index) => {
                self.selected_node = node_index.map(|index| (self.edited_chain, index));
            }
        });
    }
}

// Makes sense to also define this here, makes it a bit easier to keep track of
pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::new(|| (GUI_WIDTH, GUI_HEIGHT))
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
        move || params.threshold.slope_curve_link.value()
    };

    ParamLink::new(cx, curve_linked, curve_link_ptr, curve_pairs, content)
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
        direction_selector(cx);

        Analyzer::new(
            cx,
            Data::analyzer_data,
            Data::sample_rate,
            Data::params,
            Data::edited_direction,
            Data::selected_node,
            Data::edited_chain,
        )
        // Soaks up all vertical space the controls below don't need
        .height(Stretch(1.0));

        frequency_scale(cx);
        node_inspector(cx);
    })
    .height(Stretch(1.0))
    .left(Pixels(12.0))
    .right(Pixels(12.0));
}

/// The buttons picking which chain and which curve the analyzer edits.
///
/// These sit directly above the graph they act on. Keeping this a selection is what stops the
/// chains from doubling the number of controls on screen: everything below shows the chain that's
/// picked here.
fn direction_selector(cx: &mut Context) {
    HStack::new(cx, |cx| {
        for chain_idx in 0..NUM_CHAINS {
            Button::new(
                cx,
                move |cx| cx.emit(EditorEvent::SelectChain(chain_idx)),
                move |cx| {
                    // The chains are only left/right or mid/side depending on the stereo mode, so
                    // the label follows it rather than being fixed
                    Label::new(
                        cx,
                        Data::params.map(move |p| {
                            match (p.global.stereo_mode.value(), chain_idx) {
                                (StereoMode::MidSide, 0) => "Mid",
                                (StereoMode::MidSide, _) => "Side",
                                (_, 0) => "Left",
                                (_, _) => "Right",
                            }
                        }),
                    )
                    .font_size(12.0)
                },
            )
            .checked(Data::edited_chain.map(move |edited| *edited == chain_idx))
            .class("direction-button");
        }

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
        // The threshold curve belongs to the selected chain. Every chain's rows are built and all
        // but the selected one hidden, rather than rebuilding them inside a `Binding`: a binding's
        // entity is ignored by the layout, so its children end up stacked at the origin.
        for chain_idx in 0..NUM_CHAINS {
            let params = Data::params;

            VStack::new(cx, move |cx| {
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
            .display(Data::edited_chain.map(move |edited| *edited == chain_idx));
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
        for chain_idx in 0..NUM_CHAINS {
            for index in 0..MAX_EQ_NODES {
                HStack::new(cx, move |cx| {
                    Label::new(cx, &format!("Node {}", index + 1))
                        .width(Pixels(60.0))
                        .top(Stretch(1.0))
                        .bottom(Stretch(1.0));

                    let params = Data::params;
                    ParamSlider::new(cx, params, move |p| {
                        &p.threshold.chains[chain_idx].eq.nodes[index].node_type
                    })
                    .set_style(ParamSliderStyle::FromLeft)
                    .width(Pixels(150.0));
                    ParamSlider::new(cx, params, move |p| {
                        &p.threshold.chains[chain_idx].eq.nodes[index].target
                    })
                    .set_style(ParamSliderStyle::FromLeft)
                    .width(Pixels(120.0));
                    ParamSlider::new(cx, params, move |p| {
                        &p.threshold.chains[chain_idx].eq.nodes[index].center_frequency
                    })
                    .width(Pixels(110.0));
                    ParamSlider::new(cx, params, move |p| {
                        &p.threshold.chains[chain_idx].eq.nodes[index].gain_db
                    })
                    .width(Pixels(110.0));
                    ParamSlider::new(cx, params, move |p| {
                        &p.threshold.chains[chain_idx].eq.nodes[index].q
                    })
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
                                chain_idx,
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
                .display(
                    Data::selected_node.map(move |selected| *selected == Some((chain_idx, index))),
                );
            }
        }
    })
    .height(Pixels(24.0))
    .child_left(Stretch(1.0))
    .child_right(Stretch(1.0))
    .top(Pixels(4.0));
}

/// Set one of a node's enum parameters to the variant at `index`.
fn set_node_variant(
    cx: &mut EventContext,
    chain_idx: usize,
    node_index: usize,
    param_ptr: fn(&EqNodeParams) -> ParamPtr,
    index: usize,
    variant_count: usize,
) {
    let Some(data) = cx.data::<Data>() else {
        return;
    };
    let ptr = param_ptr(&data.params.threshold.chains[chain_idx].eq.nodes[node_index]);

    // An enum parameter's normalized range is divided evenly between its variants
    let normalized = index as f32 / (variant_count.saturating_sub(1).max(1)) as f32;

    cx.emit(RawParamEvent::BeginSetParameter(ptr));
    cx.emit(RawParamEvent::SetParameterNormalized(ptr, normalized));
    cx.emit(RawParamEvent::EndSetParameter(ptr));
}
