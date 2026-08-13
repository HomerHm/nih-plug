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

use analyzer::AnalyzerData;
use atomic_float::AtomicF32;
use crossbeam::atomic::AtomicCell;
use editor::ChainMode;
use nih_plug::prelude::*;
use nih_plug_vizia::ViziaState;
use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use triple_buffer::TripleBuffer;

mod analyzer;
mod capture;
mod compressor_bank;
mod curve;
mod dry_wet_mixer;
mod editor;
mod eq_curve;

const MIN_WINDOW_ORDER: usize = 6;
#[allow(dead_code)]
const MIN_WINDOW_SIZE: usize = 1 << MIN_WINDOW_ORDER; // 64
const DEFAULT_WINDOW_ORDER: usize = 11;
#[allow(dead_code)]
const DEFAULT_WINDOW_SIZE: usize = 1 << DEFAULT_WINDOW_ORDER; // 2048
const MAX_WINDOW_ORDER: usize = 15;
const MAX_WINDOW_SIZE: usize = 1 << MAX_WINDOW_ORDER; // 32768

const MIN_OVERLAP_ORDER: usize = 2;
#[allow(dead_code)]
const MIN_OVERLAP_TIMES: usize = 1 << MIN_OVERLAP_ORDER; // 4
const DEFAULT_OVERLAP_ORDER: usize = 4;
#[allow(dead_code)]
const DEFAULT_OVERLAP_TIMES: usize = 1 << DEFAULT_OVERLAP_ORDER; // 16
const MAX_OVERLAP_ORDER: usize = 5;
#[allow(dead_code)]
const MAX_OVERLAP_TIMES: usize = 1 << MAX_OVERLAP_ORDER; // 32

/// How long the bypass takes to fade in or out, in milliseconds.
///
/// Short enough to feel immediate, long enough that switching does not click. Bypassing swaps
/// between two signals that are aligned in time but not identical, and a hard swap between those
/// is a step.
const BYPASS_FADE_MS: f32 = 15.0;

/// This is a port of <https://github.com/robbert-vdh/spectral-compressor/>.
pub struct SpectralCompressor {
    params: Arc<SpectralCompressorParams>,

    /// The current buffer config, used for updating the compressors.
    buffer_config: BufferConfig,
    /// The current sample rate. Stores the same information as in `BufferConfig`, but this can be
    /// shared with the editor where it's used to compute frequencies for the spectrum analyzer.
    sample_rate: Arc<AtomicF32>,

    /// An adapter that performs most of the overlap-add algorithm for us.
    stft: util::StftHelper<1>,
    /// Contains a Hann window function of the current window length, passed to the overlap-add
    /// helper. Allocated with a `MAX_WINDOW_SIZE` initial capacity.
    window_function: Vec<f32>,
    /// A mixer to mix the dry signal back into the processed signal with latency compensation.
    dry_wet_mixer: dry_wet_mixer::DryWetMixer,
    /// Spectral per-bin upwards and downwards compressors with soft-knee settings. This is where
    /// the magic happens.
    compressor_bank: compressor_bank::CompressorBank,

    /// The algorithms for the FFT and IFFT operations, for each supported order so we can switch
    /// between them without replanning or allocations. Initialized during `initialize()`.
    plan_for_order: Option<[Plan; MAX_WINDOW_ORDER - MIN_WINDOW_ORDER + 1]>,
    /// The output of our real->complex FFT.
    complex_fft_buffer: Vec<Complex32>,

    /// The output for the analyzer data computed in `CompressorBank` while the editor is open. This
    /// can be cloned and moved into the editor.
    analyzer_output_data: Arc<Mutex<triple_buffer::Output<AnalyzerData>>>,

    /// Which chain is being soloed, set from the editor. Not part of the parameters, so it is
    /// never persisted.
    solo: Arc<AtomicCell<SoloMode>>,

    /// Raised by the editor while the user holds capture, and to throw the current stereo mode's
    /// curves away. Like `solo`, these are actions rather than settings, so they are deliberately
    /// not parameters: a project that reopened mid-capture would be a nasty surprise.
    capture_active: Arc<AtomicBool>,
    capture_clear: Arc<AtomicBool>,

    /// How far the bypass has faded in, from zero to one.
    ///
    /// Bypassing is done by pushing the dry/wet mix all the way dry rather than by skipping the
    /// processing, so this rides on top of that parameter. See `process()` for why.
    ///
    /// Ramped by hand rather than with a [`Smoother`], which restarts its ramp on every
    /// `set_target` call and so would approach its target without ever arriving. "Not quite
    /// bypassed" is not bypassed: the mix has to reach exactly zero for the dry path to be exact.
    bypass_amount: f32,
}

/// An FFT plan for a specific window size, all of which will be precomputed during initilaization.
struct Plan {
    /// The algorithm for the FFT operation.
    r2c_plan: Arc<dyn RealToComplex<f32>>,
    /// The algorithm for the IFFT operation.
    c2r_plan: Arc<dyn ComplexToReal<f32>>,
}

#[derive(Params)]
pub struct SpectralCompressorParams {
    /// The editor state, saved together with the parameter state so the custom scaling can be
    /// restored.
    #[persist = "editor-state"]
    pub editor_state: Arc<ViziaState>,
    /// Whether the editor keeps the two chains linked or edits them apart. Saved with the project
    /// so reopening it doesn't throw away which view was being used.
    #[persist = "chain-mode"]
    pub chain_mode: Arc<AtomicCell<ChainMode>>,

    // NOTE: These `Arc`s are only here temporarily to work around Vizia's Lens requirements so we
    // can use the generic UIs
    /// Global parameters. These could just live in this struct but I wanted a separate generic UI
    /// just for these.
    #[nested(group = "global")]
    pub global: Arc<GlobalParams>,

    /// Parameters controlling the compressor thresholds and curves.
    #[nested(group = "threshold")]
    pub threshold: Arc<compressor_bank::ThresholdParams>,
    /// Parameters for the upwards and downwards compressors.
    #[nested(group = "compressors")]
    pub compressors: compressor_bank::CompressorBankParams,
}

/// Which chain, if any, is being listened to on its own.
///
/// This is deliberately not a parameter. Soloing is a way to hear what you are adjusting, not a
/// setting, and a parameter would be saved with the project: reopening it to a soloed channel with
/// no memory of why would be worse than not having the feature.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SoloMode {
    #[default]
    Off,
    First,
    Second,
}

/// How the two channels relate to each other.
#[derive(Enum, Debug, PartialEq, Eq)]
pub enum StereoMode {
    /// Compress the left and right channels as they are.
    #[id = "stereo"]
    #[name = "Left/Right"]
    LeftRight,
    /// Convert to mid and side before compressing and back again afterwards, so the centre and
    /// the sides are compressed independently.
    #[id = "mid_side"]
    #[name = "Mid/Side"]
    MidSide,
}

/// Global parameters controlling the output stage and all compressors.
#[derive(Params)]
pub struct GlobalParams {
    /// Whether the plugin passes its input through untouched.
    ///
    /// Hidden from the generic UI because it has its own button in the title bar, where a plugin
    /// wide switch belongs. The host still sees it, and drives it from its own bypass control.
    #[id = "bypass"]
    pub bypass: BoolParam,
    /// Whether the two channels are compressed as left/right or as mid/side.
    #[id = "stereo_mode"]
    pub stereo_mode: EnumParam<StereoMode>,
    /// How much of the other channel's level to fold into each channel's detection. At 0% the
    /// channels are detected entirely independently, which is how Spectral Compressor has always
    /// behaved. At 100% they share one detector, so a loud transient in one channel pulls both
    /// down by the same amount and the stereo image doesn't shift.
    #[id = "channel_link"]
    pub channel_link: FloatParam,
    /// Makeup gain applied after the IDFT in the STFT process. If automatic makeup gain is enabled,
    /// then this acts as an offset on top of that. This is stored as linear gain.
    #[id = "output"]
    pub output_gain: FloatParam,
    // TODO: Bring this back, and with values that make more sense
    // /// Try to automatically compensate for gain differences with different input gain, threshold, and ratio values.
    // #[id = "auto_makeup"]
    // auto_makeup_gain: BoolParam,
    /// How much of the dry signal to mix in with the processed signal. The mixing is done after
    /// applying the output gain. In other words, the dry signal is not gained in any way.
    #[id = "dry_wet"]
    pub dry_wet_ratio: FloatParam,

    /// The size of the FFT window as a power of two (to prevent invalid inputs).
    #[id = "stft_window"]
    pub window_size_order: IntParam,
    /// The amount of overlap to use in the overlap-add algorithm as a power of two (again to
    /// prevent invalid inputs).
    #[id = "stft_overlap"]
    pub overlap_times_order: IntParam,

    /// The compressor's attack time in milliseconds. Controls both upwards and downwards
    /// compression.
    #[id = "attack"]
    pub compressor_attack_ms: FloatParam,
    /// The compressor's release time in milliseconds. Controls both upwards and downwards
    /// compression.
    #[id = "release"]
    pub compressor_release_ms: FloatParam,
}

impl Default for SpectralCompressor {
    fn default() -> Self {
        // The spectrum analyzer and gain reduction data is computed directly in the spectral
        // compression routine in `compressor_bank`. `analyzer_output_data` can then be used in the
        // editor to draw the data.
        let (analyzer_input_data, analyzer_output_data) = TripleBuffer::default().split();

        // Changing any of the compressor threshold or ratio parameters will set an atomic flag in
        // this object that causes the compressor thresholds and ratios to be recalcualted
        let compressor_bank =
            compressor_bank::CompressorBank::new(analyzer_input_data, 2, MAX_WINDOW_SIZE);
        // Taken before the bank is moved into the struct below; the editor drives these
        let compressor_bank_capture_active = compressor_bank.capture.active_flag();
        let compressor_bank_capture_clear = compressor_bank.capture.clear_flag();

        SpectralCompressor {
            params: Arc::new(SpectralCompressorParams::new(&compressor_bank)),

            buffer_config: BufferConfig {
                sample_rate: 1.0,
                min_buffer_size: None,
                max_buffer_size: 0,
                process_mode: ProcessMode::Realtime,
            },
            sample_rate: Arc::new(AtomicF32::new(1.0)),

            // These three will be set to the correct values in the initialize function
            stft: util::StftHelper::new(2, MAX_WINDOW_SIZE, 0),
            window_function: Vec::with_capacity(MAX_WINDOW_SIZE),
            dry_wet_mixer: dry_wet_mixer::DryWetMixer::new(0, 0, 0),
            compressor_bank,

            // This is initialized later since we don't want to do non-trivial computations before
            // the plugin is initialized
            plan_for_order: None,
            complex_fft_buffer: Vec::with_capacity(MAX_WINDOW_SIZE / 2 + 1),

            analyzer_output_data: Arc::new(Mutex::new(analyzer_output_data)),
            solo: Arc::new(AtomicCell::new(SoloMode::Off)),

            capture_active: compressor_bank_capture_active,
            capture_clear: compressor_bank_capture_clear,

            bypass_amount: 0.0,
        }
    }
}

impl Default for GlobalParams {
    fn default() -> Self {
        GlobalParams {
            // Marking it as the bypass parameter is what hooks it up to the host's own bypass
            // control. NIH-plug only passes the flag along; the bypassing itself is ours to do,
            // and it has to be, because this plugin reports latency.
            bypass: BoolParam::new("Bypass", false)
                .make_bypass()
                .hide_in_generic_ui(),
            stereo_mode: EnumParam::new("Stereo Mode", StereoMode::LeftRight),
            // Defaults to fully independent, which is the behaviour this plugin has always had
            channel_link: FloatParam::new(
                "Detection Link",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_unit("%")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),

            // We don't need any smoothing for these parameters as the overlap-add process will
            // already act as a form of smoothing
            output_gain: FloatParam::new(
                "Output Gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-50.0),
                    max: util::db_to_gain(50.0),
                    factor: FloatRange::gain_skew_factor(-50.0, 50.0),
                },
            )
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
            // auto_makeup_gain: BoolParam::new("Auto Makeup Gain", true),
            dry_wet_ratio: FloatParam::new("Mix", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_unit("%")
                .with_smoother(SmoothingStyle::Linear(15.0))
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),

            window_size_order: IntParam::new(
                "Window Size",
                DEFAULT_WINDOW_ORDER as i32,
                IntRange::Linear {
                    min: MIN_WINDOW_ORDER as i32,
                    max: MAX_WINDOW_ORDER as i32,
                },
            )
            .with_value_to_string(formatters::v2s_i32_power_of_two())
            .with_string_to_value(formatters::s2v_i32_power_of_two()),
            overlap_times_order: IntParam::new(
                "Window Overlap",
                DEFAULT_OVERLAP_ORDER as i32,
                IntRange::Linear {
                    min: MIN_OVERLAP_ORDER as i32,
                    max: MAX_OVERLAP_ORDER as i32,
                },
            )
            .with_value_to_string(formatters::v2s_i32_power_of_two())
            .with_string_to_value(formatters::s2v_i32_power_of_two()),

            compressor_attack_ms: FloatParam::new(
                "Attack",
                150.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 10_000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" ms")
            .with_step_size(0.1),
            compressor_release_ms: FloatParam::new(
                "Release",
                300.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 10_000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" ms")
            .with_step_size(0.1),
        }
    }
}

impl SpectralCompressorParams {
    /// Create a new [`SpectralCompressorParams`] object. Changing any of the compressor threshold
    /// or ratio parameters causes the passed compressor bank's parameters to be updated.
    pub fn new(compressor_bank: &compressor_bank::CompressorBank) -> Self {
        SpectralCompressorParams {
            editor_state: editor::default_state(),
            chain_mode: Arc::default(),

            // TODO: Do still enable per-block smoothing for these settings, because why not. This
            //       will require updating the compressor bank.
            global: Arc::new(GlobalParams::default()),

            threshold: Arc::new(compressor_bank::ThresholdParams::new(compressor_bank)),
            compressors: compressor_bank::CompressorBankParams::new(compressor_bank),
        }
    }
}

impl Plugin for SpectralCompressor {
    const NAME: &'static str = "Spectral Compressor HM";
    // The original author and project are deliberately kept here. This fork's own attribution
    // lives in the GUI as a "Mod by HomerHm" link, see `editor.rs`.
    const VENDOR: &'static str = "Robbert van der Helm";
    const URL: &'static str = env!("CARGO_PKG_HOMEPAGE");
    const EMAIL: &'static str = "mail@robbertvanderhelm.nl";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),

            aux_input_ports: &[new_nonzero_u32(2)],

            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),

            aux_input_ports: &[new_nonzero_u32(1)],

            ..AudioIOLayout::const_default()
        },
    ];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        editor::create(
            self.params.editor_state.clone(),
            editor::Data {
                params: self.params.clone(),

                analyzer_data: self.analyzer_output_data.clone(),
                sample_rate: self.sample_rate.clone(),

                edited_direction: eq_curve::CompressorDirection::Downwards,
                // The mode is persisted, so a reopened editor picks up where it left off. The
                // model's copy has to start from the same value the window was sized with.
                chain_mode: self.params.chain_mode.load(),
                chain_mode_cell: self.params.chain_mode.clone(),
                solo: self.solo.clone(),
                capture_active: self.capture_active.clone(),
                capture_clear: self.capture_clear.clone(),
                capture_state: self.params.threshold.capture.state.clone(),
                capturing: false,
                selected_node: None,
                node_solo: self.compressor_bank.node_solo.clone(),
                soloed_node: None,
            },
        )
    }

    fn initialize(
        &mut self,
        audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        // Needed to update the compressors later
        self.buffer_config = *buffer_config;

        // And this is used in the editor to draw the analyzer
        self.sample_rate
            .store(buffer_config.sample_rate, Ordering::Relaxed);

        // This plugin can accept a variable number of audio channels, so we need to resize
        // channel-dependent data structures accordingly
        let num_output_channels = audio_io_layout
            .main_output_channels
            .expect("Plugin does not have a main output")
            .get() as usize;
        if self.stft.num_channels() != num_output_channels {
            self.stft = util::StftHelper::new(self.stft.num_channels(), MAX_WINDOW_SIZE, 0);
        }
        self.dry_wet_mixer.resize(
            num_output_channels,
            buffer_config.max_buffer_size as usize,
            MAX_WINDOW_SIZE,
        );
        self.compressor_bank
            .update_capacity(num_output_channels, MAX_WINDOW_SIZE);

        // Planning with RustFFT is very fast, but it will still allocate we we'll plan all of the
        // FFTs we might need in advance
        if self.plan_for_order.is_none() {
            let mut planner = RealFftPlanner::new();
            let plan_for_order: Vec<Plan> = (MIN_WINDOW_ORDER..=MAX_WINDOW_ORDER)
                .map(|order| Plan {
                    r2c_plan: planner.plan_fft_forward(1 << order),
                    c2r_plan: planner.plan_fft_inverse(1 << order),
                })
                .collect();
            self.plan_for_order = Some(
                plan_for_order
                    .try_into()
                    .unwrap_or_else(|_| panic!("Mismatched plan orders")),
            );
        }

        let window_size = self.window_size();
        self.resize_for_window(window_size);
        context.set_latency_samples(self.stft.latency_samples());

        true
    }

    fn reset(&mut self) {
        self.dry_wet_mixer.reset();
        self.compressor_bank.reset();
        // Starting a bypassed plugin part way through the fade would let a moment of the processed
        // signal through
        self.bypass_amount = if self.params.global.bypass.value() {
            1.0
        } else {
            0.0
        };
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // If the window size has changed since the last process call, reset the buffers and chance
        // our latency. All of these buffers already have enough capacity so this won't allocate.
        let window_size = self.window_size();
        let overlap_times = self.overlap_times();
        if self.window_function.len() != window_size {
            self.resize_for_window(window_size);
            context.set_latency_samples(self.stft.latency_samples());
        }

        // These plans have already been made during initialization we can switch between versions
        // without reallocating
        let fft_plan = &mut self.plan_for_order.as_mut().unwrap()
            [self.params.global.window_size_order.value() as usize - MIN_WINDOW_ORDER];
        let num_bins = self.complex_fft_buffer.len();
        // The Hann window function spreads the DC signal out slightly, so we'll clear all 0-20 Hz
        // bins for this. With small window sizes you probably don't want this as it would result in
        // a significant low-pass filter. When it's disabled, the DC bin will also be compressed.
        let first_non_dc_bin_idx =
            (20.0 / ((self.buffer_config.sample_rate / 2.0) / num_bins as f32)).floor() as usize
                + 1;

        // The overlap gain compensation is based on a squared Hann window, which will sum perfectly
        // at four times overlap or higher. We'll apply a regular Hann window before the analysis
        // and after the synthesis.
        let gain_compensation: f32 =
            ((overlap_times as f32 / 4.0) * 1.5).recip() / window_size as f32;

        // We'll apply the square root of the total gain compensation at the DFT and the IDFT
        // stages. That way the compressor threshold values make much more sense. This version of
        // Spectral Compressor does not have in input gain option and instead has the curve
        // threshold option. When sidechaining is enabled this is used to gain up the sidechain
        // signal instead.
        let input_gain = gain_compensation.sqrt();
        let output_gain = self.params.global.output_gain.value() * gain_compensation.sqrt();
        // TODO: Auto makeup gain

        // This is mixed in later with latency compensation applied. It has to happen before the
        // mid/side conversion below so the dry signal stays in left/right.
        self.dry_wet_mixer.write_dry(buffer);

        // The conversion is its own inverse up to a factor of two, and the STFT delays both
        // channels equally, so encoding what goes in and decoding what comes out lines up even
        // though those are different samples.
        let mid_side = self.params.global.stereo_mode.value() == StereoMode::MidSide;
        if mid_side {
            convert_mid_side(buffer);
        }

        // The sidechain STFT normally only runs for the modes that need it. Capturing from the
        // sidechain needs it too, and without this the capture would quietly record silence --
        // which is the workflow that makes matching another track practical in the first place.
        let capturing_sidechain = self.compressor_bank.capture.is_active()
            && self.params.threshold.capture.source.value() == capture::CaptureSource::Sidechain;
        let needs_sidechain = capturing_sidechain
            || !matches!(
                self.params.threshold.mode.value(),
                compressor_bank::ThresholdMode::Internal
            );

        // The main buffer was converted above but the sidechain never was, so in mid/side mode the
        // second chain was detecting on the sidechain's *right* channel rather than its side
        // channel. A mono sidechain made that plain: side should be silent, and instead it carried
        // the whole signal. Both sidechain threshold modes had the same mismatch.
        //
        // Converting in place rather than into a copy is sanctioned: nih-plug documents auxiliary
        // input buffers as safe to overwrite (`audio_setup.rs`). It is skipped when the sidechain
        // STFT is not going to run, so nothing is touched that isn't about to be read.
        if mid_side && needs_sidechain {
            convert_mid_side(&mut aux.inputs[0]);
        }

        match self.params.threshold.mode.value() {
            compressor_bank::ThresholdMode::Internal if !needs_sidechain => self
                .stft
                .process_overlap_add(buffer, overlap_times, |channel_idx, real_fft_buffer| {
                    process_stft_main(
                        channel_idx,
                        real_fft_buffer,
                        &mut self.complex_fft_buffer,
                        fft_plan,
                        &self.window_function,
                        &self.params,
                        &mut self.compressor_bank,
                        input_gain,
                        output_gain,
                        overlap_times,
                        first_non_dc_bin_idx,
                    )
                }),
            _ => self.stft.process_overlap_add_sidechain(
                buffer,
                [&aux.inputs[0]],
                overlap_times,
                |channel_idx, sidechain_buffer_idx, real_fft_buffer| {
                    if sidechain_buffer_idx.is_some() {
                        process_stft_sidechain(
                            channel_idx,
                            real_fft_buffer,
                            &mut self.complex_fft_buffer,
                            fft_plan,
                            &self.window_function,
                            &self.params,
                            &mut self.compressor_bank,
                            input_gain,
                        );
                    } else {
                        process_stft_main(
                            channel_idx,
                            real_fft_buffer,
                            &mut self.complex_fft_buffer,
                            fft_plan,
                            &self.window_function,
                            &self.params,
                            &mut self.compressor_bank,
                            input_gain,
                            output_gain,
                            overlap_times,
                            first_non_dc_bin_idx,
                        )
                    }
                },
            ),
        }

        if mid_side {
            convert_mid_side(buffer);
        }

        // Bypassing is done by pushing the mix all the way dry rather than by skipping the work
        // above, for two reasons. The dry signal is already delayed to match the plugin's reported
        // latency, so a fully dry output lines up with the host's delay compensation exactly --
        // `mix_in_dry` copies it verbatim at a ratio of zero, so what comes out is the input, to
        // the bit. And leaving the STFT running keeps its ring buffers current, so unbypassing does
        // not spill up to a window of stale audio; it also leaves the analyzer and the capture
        // working, which is what makes "bypass, capture a reference, unbypass" possible at all.
        //
        // The cost is that a bypassed plugin still does all the work. Skipping it would save that,
        // but only by reintroducing the discontinuity above.
        let bypass_target = if self.params.global.bypass.value() {
            1.0
        } else {
            0.0
        };
        let bypass_start = self.bypass_amount;
        let bypass_step = buffer.samples() as f32
            / (BYPASS_FADE_MS / 1000.0 * self.buffer_config.sample_rate).max(1.0);
        self.bypass_amount = if bypass_target > bypass_start {
            (bypass_start + bypass_step).min(bypass_target)
        } else {
            (bypass_start - bypass_step).max(bypass_target)
        };

        // The mix is handed both ends of the block so the bypass fade happens per sample. The dry
        // and wet halves are the same signal at the same moment in time but not the same numbers,
        // and stepping between them once per block is a switch, not a fade.
        let dry_wet_ratio = self
            .params
            .global
            .dry_wet_ratio
            .smoothed
            .next_step(buffer.samples() as u32);
        self.dry_wet_mixer.mix_in_dry(
            buffer,
            dry_wet_ratio * (1.0 - bypass_start),
            dry_wet_ratio * (1.0 - self.bypass_amount),
            // The dry and wet signals are in phase, so we can do a linear mix
            dry_wet_mixer::MixingStyle::Linear,
            self.stft.latency_samples() as usize,
        );

        // Soloing is applied to what actually leaves the plugin, dry signal and all: hearing the
        // dry half of the other channel through the mix would defeat the point. It fades out with
        // the bypass, since a bypassed plugin has no business muting anything.
        apply_solo(buffer, self.solo.load(), mid_side, self.bypass_amount);

        ProcessStatus::Normal
    }
}

impl SpectralCompressor {
    fn window_size(&self) -> usize {
        1 << self.params.global.window_size_order.value() as usize
    }

    fn overlap_times(&self) -> usize {
        1 << self.params.global.overlap_times_order.value() as usize
    }

    /// `window_size` should not exceed `MAX_WINDOW_SIZE` or this will allocate.
    fn resize_for_window(&mut self, window_size: usize) {
        // The FFT algorithms for this window size have already been planned in
        // `self.plan_for_order`, and all of these data structures already have enough capacity, so
        // we just need to change some sizes.
        self.stft.set_block_size(window_size);
        self.window_function.resize(window_size, 0.0);
        util::window::hann_in_place(&mut self.window_function);
        self.complex_fft_buffer
            .resize(window_size / 2 + 1, Complex32::default());

        // This also causes the thresholds and ratios to be updated on the next STFT process cycle.
        self.compressor_bank
            .resize(&self.buffer_config, window_size);
        self.compressor_bank.reset();
    }
}

/// Mute everything but one chain, so it can be listened to on its own.
///
/// In mid/side the chains are not channels, so the buffer is converted, the unwanted half zeroed,
/// and converted back. Soloing the mid then plays it from both speakers and soloing the side gives
/// the usual out of phase pair, which is what those are supposed to sound like.
///
/// `muted_gain` is how much of the unwanted chain survives, which is the bypass fade. A bypassed
/// plugin has to pass its input through untouched, and stepping the muted chain back to full at the
/// end of that fade would put a click in exactly the signal that is supposed to be clean.
fn apply_solo(buffer: &mut Buffer, solo: SoloMode, mid_side: bool, muted_gain: f32) {
    let muted_channel = match solo {
        SoloMode::Off => return,
        SoloMode::First => 1,
        SoloMode::Second => 0,
    };
    if muted_gain >= 1.0 {
        return;
    }

    if mid_side {
        convert_mid_side(buffer);
    }

    let channels = buffer.as_slice();
    if let Some(channel) = channels.get_mut(muted_channel) {
        for sample in channel.iter_mut() {
            *sample *= muted_gain;
        }
    }

    if mid_side {
        convert_mid_side(buffer);
    }
}

/// Convert a stereo buffer between left/right and mid/side, in place.
///
/// `mid = (left + right) / 2` and `side = (left - right) / 2` inverts to `left = mid + side` and
/// `right = mid - side`, so running this twice scales by a half. Both directions therefore use the
/// same expression with a gain of one, keeping a mid/side round trip unity.
fn convert_mid_side(buffer: &mut Buffer) {
    let channels = buffer.as_slice();
    if channels.len() != 2 {
        return;
    }

    let (first, second) = channels.split_at_mut(1);
    for (left, right) in first[0].iter_mut().zip(second[0].iter_mut()) {
        let mid = (*left + *right) * std::f32::consts::FRAC_1_SQRT_2;
        let side = (*left - *right) * std::f32::consts::FRAC_1_SQRT_2;
        (*left, *right) = (mid, side);
    }
}

// These separate functions are needed to avoid having to either duplicate the main process function
// or always do the sidechain STFT. You can't do partial borrows and call `&mut self` methods at the
// same time.

/// The main process function inside of the STFT callback. If the sidechaining option is
/// enabled, another callback will run just before this to set up the siddechain frequency
/// spectrum magnitudes.
#[allow(clippy::too_many_arguments)]
fn process_stft_main(
    channel_idx: usize,
    real_fft_buffer: &mut [f32],
    complex_fft_buffer: &mut [Complex32],
    fft_plan: &Plan,
    window_function: &[f32],
    params: &SpectralCompressorParams,
    compressor_bank: &mut compressor_bank::CompressorBank,
    input_gain: f32,
    output_gain: f32,
    overlap_times: usize,
    first_non_dc_bin_idx: usize,
) {
    // We'll window the input with a Hann function to avoid spectral leakage. The input gain
    // here also contains a compensation factor for the forward FFT to make the compressor
    // thresholds make more sense.
    for (sample, window_sample) in real_fft_buffer.iter_mut().zip(window_function) {
        *sample *= window_sample * input_gain;
    }

    // RustFFT doesn't actually need a scratch buffer here, so we'll pass an empty buffer
    // instead
    fft_plan
        .r2c_plan
        .process_with_scratch(real_fft_buffer, complex_fft_buffer, &mut [])
        .unwrap();

    // This is where the magic happens
    compressor_bank.process(
        complex_fft_buffer,
        channel_idx,
        params,
        overlap_times,
        first_non_dc_bin_idx,
    );

    // Inverse FFT back into the scratch buffer. This will be added to a ring buffer
    // which gets written back to the host at a one block delay.
    fft_plan
        .c2r_plan
        .process_with_scratch(complex_fft_buffer, real_fft_buffer, &mut [])
        .unwrap();

    // Apply the window function once more to reduce time domain aliasing. The gain
    // compensation compensates for the squared Hann window that would be applied if we
    // didn't do any processing at all as well as the FFT+IFFT itself.
    for (sample, window_sample) in real_fft_buffer.iter_mut().zip(window_function) {
        *sample *= window_sample * output_gain;
    }
}

/// The analysis process function inside of the STFT callback used to compute the frequency
/// spectrum magnitudes from the sidechain input if the sidechaining option is enabled. All
/// sidechain channels will be processed before processing the main input
#[allow(clippy::too_many_arguments)]
fn process_stft_sidechain(
    channel_idx: usize,
    real_fft_buffer: &mut [f32],
    complex_fft_buffer: &mut [Complex32],
    fft_plan: &Plan,
    window_function: &[f32],
    params: &SpectralCompressorParams,
    compressor_bank: &mut compressor_bank::CompressorBank,
    input_gain: f32,
) {
    // The sidechain input should be gained, scaled, and windowed the exact same was as the
    // main input as it's used for analysis
    for (sample, window_sample) in real_fft_buffer.iter_mut().zip(window_function) {
        *sample *= window_sample * input_gain;
    }

    fft_plan
        .r2c_plan
        .process_with_scratch(real_fft_buffer, complex_fft_buffer, &mut [])
        .unwrap();
    compressor_bank.process_sidechain(complex_fft_buffer, channel_idx, params);
}

impl ClapPlugin for SpectralCompressor {
    const CLAP_ID: &'static str = "com.homerhm.spectral-compressor-hm";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Turn things into pink noise on demand (HomerHm fork)");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::PhaseVocoder,
        ClapFeature::Compressor,
        ClapFeature::Custom("nih:spectral"),
        ClapFeature::Custom("nih:sosig"),
    ];
}

impl Vst3Plugin for SpectralCompressor {
    // NOTE: This must be exactly 16 bytes, and it must differ from the upstream
    //       `SpectrlComprRvdH` so this fork and the original can coexist in a host.
    const VST3_CLASS_ID: [u8; 16] = *b"SpectralComprsHM";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Dynamics,
        Vst3SubCategory::Custom("Spectral"),
    ];
}

nih_export_clap!(SpectralCompressor);
nih_export_vst3!(SpectralCompressor);
