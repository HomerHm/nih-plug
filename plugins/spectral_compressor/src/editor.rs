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
use crate::analyzer::AnalyzerData;
use crate::{SpectralCompressor, SpectralCompressorParams};

mod analyzer;

/// The GUI's width, in logical pixels. Wide enough for the four control columns below the
/// analyzer to sit side by side.
const GUI_WIDTH: u32 = 1360;
/// The GUI's height, in logical pixels.
///
/// The controls take exactly as much vertical space as they need and the analyzer absorbs whatever
/// is left over, so this number only sets how much room the analyzer gets. Growing a control
/// column can therefore never truncate it -- it just eats into the analyzer.
const GUI_HEIGHT: u32 = 760;
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

        make_column(cx, "Upwards", |cx| {
            // We don't want to show the 'Upwards' prefix here, but it should still be in
            // the parameter name so the parameter list makes sense
            let upwards_compressor_params = Data::params.map(|p| p.compressors.upwards.clone());
            GenericUi::new_custom(cx, upwards_compressor_params, |cx, param_ptr| {
                HStack::new(cx, |cx| {
                    Label::new(
                        cx,
                        unsafe { param_ptr.name() }
                            .strip_prefix("Upwards ")
                            .expect("Expected parameter name prefix, this is a bug"),
                    )
                    .class("label");

                    GenericUi::draw_widget(cx, upwards_compressor_params, param_ptr);
                })
                .class("row");
            });
        });

        make_column(cx, "Downwards", |cx| {
            let downwards_compressor_params = Data::params.map(|p| p.compressors.downwards.clone());
            GenericUi::new_custom(cx, downwards_compressor_params, |cx, param_ptr| {
                HStack::new(cx, |cx| {
                    Label::new(
                        cx,
                        unsafe { param_ptr.name() }
                            .strip_prefix("Downwards ")
                            .expect("Expected parameter name prefix, this is a bug"),
                    )
                    .class("label");

                    GenericUi::draw_widget(cx, downwards_compressor_params, param_ptr);
                })
                .class("row");
            });
        });
    })
    // Auto means this row is exactly as tall as its tallest column, so adding parameters makes
    // the window's analyzer shrink rather than pushing controls out of view
    .height(Auto)
    .bottom(Pixels(12.0))
    .child_left(Stretch(1.0))
    .child_right(Stretch(1.0));
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
