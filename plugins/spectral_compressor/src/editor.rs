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

use self::analyzer::Analyzer;
use self::param_link::{param_ptr_by_id, param_ptr_pairs, ParamLink};
use crate::analyzer::AnalyzerData;
use crate::eq_curve::MAX_EQ_NODES;
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

#[derive(Clone, Lens)]
pub struct Data {
    pub(crate) params: Arc<SpectralCompressorParams>,

    pub(crate) analyzer_data: Arc<Mutex<triple_buffer::Output<AnalyzerData>>>,
    /// Used by the analyzer to determine which FFT bins belong to which frequencies.
    pub(crate) sample_rate: Arc<AtomicF32>,
}

impl Model for Data {}

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

        VStack::new(cx, |cx| {
            title_bar(cx);
            // `Stretch` here is what keeps the controls below from ever being clipped: they take
            // their natural height first and the analyzer gets the remainder.
            analyzer(cx);
            controls(cx);
            threshold_eq(cx);
        })
        .row_between(Pixels(10.0));

        ResizeHandle::new(cx);
    })
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
    Analyzer::new(cx, Data::analyzer_data, Data::sample_rate)
        // Soaks up all vertical space the controls below don't need
        .height(Stretch(1.0))
        .left(Pixels(12.0))
        .right(Pixels(12.0));
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

/// The Upwards and Downwards columns, wrapped in the view that keeps their threshold curve shapes
/// in lockstep while the link is enabled.
fn compressor_columns(cx: &mut Context) {
    let params = Data::params.get(cx);

    // These IDs are the ones declared on `CompressorParams`; the `upwards`/`downwards` prefixes
    // are added a level up and are not part of them. Downwards is the leader, so switching the
    // link on pulls the upwards curve onto the downwards one rather than the other way around.
    let pairs = ["curve_center", "curve_slope", "curve_curve"]
        .iter()
        .map(|id| {
            (
                param_ptr_by_id(&params.compressors.downwards, id),
                param_ptr_by_id(&params.compressors.upwards, id),
            )
        })
        .collect();

    let link_ptr = param_ptr_by_id(&params.threshold, "thresh_link");
    let is_linked = {
        let params = params.clone();
        move || params.threshold.slope_curve_link.value()
    };

    // NOTE: The link button lives inside `ParamLink` so that toggling it is visible to the view,
    //       which needs to sync the two sides the moment the link is switched on.
    ParamLink::new(cx, is_linked, link_ptr, pairs, |cx| {
        VStack::new(cx, |cx| {
            ParamButton::new(cx, Data::params, |p| &p.threshold.slope_curve_link)
                .with_label("Thresh Curve Link")
                .left(Stretch(1.0))
                .right(Stretch(1.0))
                .bottom(Pixels(4.0));

            HStack::new(cx, |cx| {
                compressor_column(cx, "Upwards");
                compressor_column(cx, "Downwards");
            })
            .height(Auto);
        })
        .height(Auto);
    })
    .width(Pixels(660.0))
    .height(Auto);
}

/// One compressor's column. `direction` is both the heading and the parameter name prefix that
/// gets stripped from each row's label.
fn compressor_column(cx: &mut Context, direction: &'static str) {
    make_column(cx, direction, move |cx| {
        // We don't want to show the 'Upwards'/'Downwards' prefix here, but it should still be in
        // the parameter name so the parameter list makes sense
        let compressor_params = compressor_params_lens(direction);
        let strip_prefix = format!("{direction} ");
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

/// The threshold curve's EQ nodes, one section per compressor.
///
/// These sit below the four control columns and span the full width, because a node needs four
/// controls and cramming those into a 330 pixel column would leave them unusably narrow. One row
/// per node also keeps this section six rows tall instead of twenty-four.
///
/// This whole section is temporary scaffolding: it exists so the node maths can be driven and
/// verified before the nodes become draggable on the analyzer itself, and should go away once
/// they are.
fn threshold_eq(cx: &mut Context) {
    let params = Data::params.get(cx);

    // Pairing by ID rather than listing the parameters means adding one to `EqNodeParams` later
    // cannot silently leave it out of the link
    let pairs = param_ptr_pairs(
        &params.compressors.downwards.eq,
        &params.compressors.upwards.eq,
    );
    let link_ptr = param_ptr_by_id(&params.threshold, "eq_link");
    let is_linked = {
        let params = params.clone();
        move || params.threshold.eq_link.value()
    };

    // NOTE: As with the curve link, the button has to be inside the view that acts on it.
    ParamLink::new(cx, is_linked, link_ptr, pairs, |cx| {
        VStack::new(cx, |cx| {
            ParamButton::new(cx, Data::params, |p| &p.threshold.eq_link)
                .with_label("Thresh EQ Link")
                .left(Stretch(1.0))
                .right(Stretch(1.0));

            HStack::new(cx, |cx| {
                threshold_eq_section(cx, "Upwards");
                threshold_eq_section(cx, "Downwards");
            })
            .height(Auto)
            .child_left(Stretch(1.0))
            .child_right(Stretch(1.0));
        })
        .height(Auto);
    })
    .height(Auto)
    .bottom(Pixels(12.0));
}

fn threshold_eq_section(cx: &mut Context, direction: &'static str) {
    let params = compressor_params_lens(direction);

    VStack::new(cx, |cx| {
        Label::new(cx, &format!("{direction} Threshold EQ"))
            .font_family(vec![FamilyOwned::Name(String::from(assets::NOTO_SANS))])
            .font_weight(FontWeightKeyword::Thin)
            .font_size(23.0)
            .left(Stretch(1.0))
            .right(Pixels(7.0))
            .bottom(Pixels(-4.0));

        for index in 0..MAX_EQ_NODES {
            HStack::new(cx, move |cx| {
                Label::new(cx, &format!("{}", index + 1))
                    .width(Pixels(16.0))
                    .top(Stretch(1.0))
                    .bottom(Stretch(1.0));

                // A node is inert until its type is set to something other than Off, so the type
                // comes first and the three values that shape it follow. `CurrentStepLabeled`
                // would overlay all eleven type names on top of each other, so this uses the same
                // style the generic UI picks for parameters with this many steps.
                ParamSlider::new(cx, params, move |p| &p.eq.nodes[index].node_type)
                    .set_style(ParamSliderStyle::FromLeft)
                    .width(Stretch(1.3));
                ParamSlider::new(cx, params, move |p| &p.eq.nodes[index].center_frequency)
                    .width(Stretch(1.0));
                ParamSlider::new(cx, params, move |p| &p.eq.nodes[index].gain_db)
                    .width(Stretch(1.0));
                ParamSlider::new(cx, params, move |p| &p.eq.nodes[index].q).width(Stretch(0.8));
            })
            .height(Pixels(26.0))
            .col_between(Pixels(2.0))
            .class("row");
        }
    })
    .width(Pixels(660.0))
    .height(Auto);
}

/// A lens to one compressor's parameters, picked by the same name used for its heading.
fn compressor_params_lens(
    direction: &'static str,
) -> impl Lens<Target = Arc<crate::compressor_bank::CompressorParams>> {
    Data::params.map(move |p| match direction {
        "Upwards" => p.compressors.upwards.clone(),
        _ => p.compressors.downwards.clone(),
    })
}
