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

use nih_plug::prelude::*;
use realfft::num_complex::Complex32;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::analyzer::AnalyzerData;
use crate::capture::{
    smooth_into, CaptureBank, CaptureBlend, CaptureParams, CaptureSource, CAPTURE_GRID_LEN,
};
use crate::curve::{smoothstep, Curve, CurveParams};
use crate::eq_curve::{power_to_db, CompressorDirection, EqBankParams, EqCurve, EqCurveParams};
use crate::SpectralCompressorParams;

// These are the parameter name prefixes used for the downwards and upwards compression parameters.
// The ID prefixes a re set in the `CompressorBankParams` struct.
const DOWNWARDS_NAME_PREFIX: &str = "Downwards";
const UPWARDS_NAME_PREFIX: &str = "Upwards";

/// How much the threshold curve tilts down by default, as a slope in the log/log domain.
///
/// This is what makes the default settings compress a typical mix evenly: music falls off at
/// roughly this rate, so a curve that follows it sits at a constant distance above the audio. It is
/// an assumption about what music looks like, which is exactly what a sound signature capture
/// replaces -- see [`ThresholdParams::baseline_slope()`].
const PINK_NOISE_SLOPE: f32 = -3.0;

/// At or below this the low frequency bypass is switched off entirely.
///
/// This is what makes the parameter's leftmost position mean "off" without needing a separate
/// switch. Nothing below twenty hertz is audible, and the FFT can barely resolve it either, so
/// "bypass below twenty hertz" and "no bypass" are the same setting in practice.
const LF_BYPASS_OFF_HZ: f32 = 20.0;

/// How far above its corner the low frequency bypass fades back to full compression, in octaves.
///
/// Fixed rather than exposed. Switching compression off between one bin and the next is a step in
/// the frequency response, and a step there is a long tail in the time domain.
const LF_BYPASS_FADE_OCTAVES: f32 = 0.5;

/// The envelopes are initialized to the RMS value of a -24 dB sine wave to make sure extreme upwards
/// compression doesn't cause pops when switching between window sizes and when deactivating and
/// reactivating the plugin.
const ENVELOPE_INIT_VALUE: f32 = std::f32::consts::FRAC_1_SQRT_2 / 8.0;

/// The target frequency for the high frequency ratio rolloff. This is fixed to prevent Spectral
/// Compressor from getting brighter as the sample rate increases.
#[allow(unused)]
const HIGH_FREQ_RATIO_ROLLOFF_FREQUENCY: f32 = 22_050.0;
const HIGH_FREQ_RATIO_ROLLOFF_FREQUENCY_LN: f32 = 10.001068; // 22_050.0f32.ln()

/// The length of time over which the envelope followers fade back from being instant to using the
/// configured timingsafter the compressor bank has been reset.
const ENVELOPE_FOLLOWER_TIMING_FADE_MS: f32 = 150.0;

/// A bank of compressors so each FFT bin can be compressed individually. The vectors in this struct
/// will have a capacity of `MAX_WINDOW_SIZE / 2 + 1` and a size that matches the current complex
/// FFT buffer size. This is stored as a struct of arrays to make SIMD-ing easier in the future.
pub struct CompressorBank {
    /// If set, then the downwards thresholds should be updated on the next processing cycle. Can be
    /// set from a parameter value change listener, and is also set when calling `.reset_for_size`.
    pub should_update_downwards_thresholds: Arc<AtomicBool>,
    /// The same as `should_update_downwards_thresholds`, but for upwards thresholds.
    pub should_update_upwards_thresholds: Arc<AtomicBool>,
    /// If set, then the downwards ratios should be updated on the next processing cycle. Can be set
    /// from a parameter value change listener, and is also set when calling `.reset_for_size`.
    pub should_update_downwards_ratios: Arc<AtomicBool>,
    /// The same as `should_update_downwards_ratios`, but for upwards ratios.
    pub should_update_upwards_ratios: Arc<AtomicBool>,
    /// If set, then the parameters for the downwards compression soft knee parabola should be
    /// updated on the next processing cycle. Can be set from a parameter value change listener, and
    /// is also set when calling `.reset_for_size`.
    pub should_update_downwards_knee_parabolas: Arc<AtomicBool>,
    /// The same as `should_update_downwards_knee_parabolas`, but for upwards compression.
    pub should_update_upwards_knee_parabolas: Arc<AtomicBool>,
    /// If set, then the low frequency bypass weights should be recomputed on the next cycle.
    pub should_update_process_weights: Arc<AtomicBool>,

    /// For each compressor bin, `ln(freq)` where `freq` is the frequency associated with that
    /// compressor. This is precomputed since all update functions need it.
    ln_freqs: Vec<f32>,
    /// The plain frequency in Hertz for each bin. The EQ nodes need this rather than its logarithm,
    /// and caching it turns their `freq / center_freq` into a single multiplication per bin.
    freqs: Vec<f32>,
    /// Scratch space for accumulating the EQ nodes' squared magnitude responses. Kept here so
    /// updating the thresholds never has to allocate on the audio thread.
    eq_power: Vec<f32>,

    /// Downwards compressor thresholds, in decibels.
    downwards_thresholds_db: [Vec<f32>; NUM_CHAINS],
    /// The ratios for the the downwards compressors. At 1.0 the cmopressor won't do anything. If
    /// [`CompressorBankParams::high_freq_ratio_rolloff`] is set to 1.0, then this will be the same
    /// for each compressor.
    downwards_ratios: Vec<f32>,
    /// The knee is modelled as a parabola using the formula `x + a * (x + b)^2`. This is `a` in
    /// that equation. The formula is taken from the Digital Dynamic Range Compressor Design paper
    /// by Dimitrios Giannoulis et. al.
    downwards_knee_parabola_scale: [Vec<f32>; NUM_CHAINS],
    /// `b` in the equation from `downwards_knee_parabola_scale`.
    downwards_knee_parabola_intercept: [Vec<f32>; NUM_CHAINS],

    /// Upwards compressor thresholds, in decibels.
    upwards_thresholds_db: [Vec<f32>; NUM_CHAINS],
    /// The same as `downwards_ratios`, but for the upwards compression.
    upwards_ratios: Vec<f32>,
    /// `downwards_knee_parabola_scale`, but for the upwards compressors.
    upwards_knee_parabola_scale: [Vec<f32>; NUM_CHAINS],
    /// `downwards_knee_parabola_intercept`, but for the upwards compressors.
    upwards_knee_parabola_intercept: [Vec<f32>; NUM_CHAINS],

    /// How much of the computed gain change each bin actually receives, from zero to one. This is
    /// what the low frequency bypass acts through: at zero a bin comes out exactly as it went in.
    ///
    /// Applied to the gain rather than to the threshold because no threshold can express "leave
    /// this alone" -- raising it stops downwards compression but makes upwards compression lift
    /// *more*, which is the opposite of what is wanted.
    process_weights: [Vec<f32>; NUM_CHAINS],

    /// The current envelope value for this bin, in linear space. Indexed by
    /// `[channel_idx][compressor_idx]`.
    envelopes: Vec<Vec<f32>>,
    /// A scaling factor for the envelope follower timings. This is set to 0 and then slowly brought
    /// back up to 1 after after [`CompressorBank::reset()`] has been called to allow the envelope
    /// followers to settle back in.
    envelope_followers_timing_scale: f32,
    /// The current block's bin magnitudes for each channel, used to let the channels share their
    /// detection. Indexed by `[channel_idx][bin_idx]`.
    spectrum_magnitudes: Vec<Vec<f32>>,
    /// When sidechaining is enabled, this contains the per-channel frqeuency spectrum magnitudes
    /// for the current block. The compressor thresholds and knee values are multiplied by these
    /// values to get the effective thresholds.
    sidechain_spectrum_magnitudes: Vec<Vec<f32>>,
    /// The window size this compressor bank was configured for. This is used to compute the
    /// coefficients for the envelope followers in the process function.
    window_size: usize,
    /// The sample rate this compressor bank was configured for. This is used to compute the
    /// coefficients for the envelope followers in the process function.
    sample_rate: f32,

    /// The captured sound signature, and the accumulator that fills it. Lives here because it
    /// needs the same bin magnitudes and frequencies everything else in this file works from.
    pub capture: CaptureBank,
    /// One chain's captured curve after smoothing, rebuilt just before the threshold arrays that
    /// use it. Kept here so smoothing never allocates on the audio thread.
    capture_smoothed: Vec<f32>,

    /// The input data for the spectrum analyzer. Stores both the spectrum analyzer values and the
    /// current gain reduction. Used to draw the spectrum analyzer and gain reduction display in the
    /// editor.
    analyzer_input_data: triple_buffer::Input<AnalyzerData>,
}

#[derive(Params)]
pub struct ThresholdParams {
    /// The compressor threshold at the center frequency. When sidechaining is enabled, the input
    /// signal is gained by the inverse of this value. This replaces the input gain in the original
    /// Spectral Compressor. In the polynomial below, this is the intercept.
    #[id = "tresh_global"]
    pub threshold_db: FloatParam,
    /// Mirrors the threshold curve shape between the upwards and downwards compressors while
    /// enabled.
    ///
    /// The mirroring deliberately happens in the editor rather than in the DSP: each compressor
    /// always reads its own slope and curve, and the editor copies one side's gestures onto the
    /// other. That keeps this file free of any link special-casing, and it means both sliders
    /// visibly show the same number instead of one silently shadowing the other.
    #[id = "thresh_link"]
    pub slope_curve_link: BoolParam,

    /// Controls the type of threshold that should be used. Check [`ThresholdMode`] for more
    /// information.
    #[id = "thresh_mode"]
    pub mode: EnumParam<ThresholdMode>,
    /// A `[0, 1]` parameter that controls how much of the other channels should be mixed in when
    /// computing the channel gain value that is then multiplied with he thresholds and knee values
    /// to the the compression parameters when using the sidechain modes.
    #[id = "thresh_sc_link"]
    pub sc_channel_link: FloatParam,

    /// The two processing chains. Which audio ends up in which depends on the stereo mode.
    #[nested(array, group = "Chain")]
    pub chains: [ChainParams; NUM_CHAINS],

    /// EQ-shaped nodes deforming the threshold curves. One bank is shared by both compressors and
    /// both chains, with each node carrying which of them it applies to. Giving each chain its own
    /// bank instead meant a node meant for both was really two nodes that started out identical,
    /// so editing one silently left the other behind.
    #[nested(group = "Threshold EQ")]
    pub eq: EqBankParams,

    /// The captured sound signature, and how much of it stands in for the polynomial's shape.
    #[nested(group = "Capture")]
    pub capture: CaptureParams,
}

/// The type of threshold to use.
#[derive(Enum, Debug, PartialEq, Eq)]
pub enum ThresholdMode {
    /// Configure the thresholds to offset pink noise. This means that the slope will receive an
    /// additional -3 dB/octave slope.
    #[id = "internal"]
    #[name = "Pink Noise"]
    Internal,
    /// Dynamically reconfigure the thresholds based on a sidechain input. The -3 dB/octave slope
    /// offset is not applied here so the curve stays true to the sidechain input at the default
    /// settings. This works by simply multiplying the sidechain gain levels with the precomputed
    /// threshold, knee start, and knee end values. The sidechain channel linking option determines
    /// how how much of the other channel values to mix in before multiplying the sidechain gain
    /// values with the thresholds.
    #[id = "sidechain"]
    #[name = "Sidechain Matching"]
    SidechainMatch,
    /// Compress the input signal based on the sidechain signal's activity. Can be used to
    /// spectrally duck the input, or to amplify parts of the input based on holes in the sidechain
    /// signal.
    #[id = "sidechain_compress"]
    #[name = "Sidechain Compression"]
    SidechainCompress,
}

/// Contains the compressor parameters for both the upwards and downwards compressor banks.
#[derive(Params)]
pub struct CompressorBankParams {
    #[nested(id_prefix = "upwards", group = "upwards")]
    pub upwards: Arc<CompressorParams>,
    #[nested(id_prefix = "downwards", group = "downwards")]
    pub downwards: Arc<CompressorParams>,
}

/// This struct contains the parameters for either the upward or downward compressors. The `Params`
/// trait is implemented manually to avoid copy-pasting parameters for both types of compressor.
/// Both versions will have a parameter ID and a parameter name prefix to distinguish them.
/// The number of independent processing chains.
///
/// Which audio ends up in which chain depends on [`crate::StereoMode`]: left and right, or mid and
/// side. Everything downstream just sees two chains.
pub const NUM_CHAINS: usize = 2;

/// One compressor's threshold curve within one chain.
///
/// Only the threshold is per-chain. Ratio, knee and the high frequency rolloff stay shared, since
/// wanting a different threshold for the sides than the centre is common while wanting a different
/// ratio for them is not, and splitting those too would double the parameter count again.
#[derive(Params)]
pub struct ThresholdCurveParams {
    /// The center frequency this curve pivots around. The curve is a polynomial
    /// `threshold_db + curve_slope*x + curve_curve*(x^2)` that evaluates to a decibel value, where
    /// `x = ln(center_frequency) - ln(bin_frequency)`. In other words, this is evaluated in the
    /// log/log domain for decibels and octaves.
    #[id = "curve_center"]
    pub center_frequency: FloatParam,
    /// The slope for the curve, in the log/log domain. See the polynomial above.
    #[id = "curve_slope"]
    pub curve_slope: FloatParam,
    /// The, uh, 'curve' for the curve. This is the third coefficient in the quadratic polynomial
    /// and controls the parabolic behavior. Positive values turn the curve into a v-shaped curve,
    /// while negative values attenuate everything outside of the center frequency.
    #[id = "curve_curve"]
    pub curve_curve: FloatParam,
    /// This compressor's threshold relative to the curve above.
    #[id = "threshold_offset"]
    pub threshold_offset_db: FloatParam,
}

/// Everything that can differ between the two processing chains.
#[derive(Params)]
pub struct ChainParams {
    #[nested(id_prefix = "upwards", group = "Upwards")]
    pub upwards: ThresholdCurveParams,
    #[nested(id_prefix = "downwards", group = "Downwards")]
    pub downwards: ThresholdCurveParams,

    /// Below this frequency the chain is left alone entirely, in both directions.
    ///
    /// Per chain because the two carry different signals: the low end of a side channel is a very
    /// different thing from the low end of a mid channel, and wanting to protect one but not the
    /// other is the normal case rather than an exotic one.
    #[id = "lfbypass"]
    pub bypass_below_hz: FloatParam,
}

/// This struct contains the parameters for either the upward or downward compressors. The `Params`
/// trait is implemented manually to avoid copy-pasting parameters for both types of compressor.
/// Both versions will have a parameter ID and a parameter name prefix to distinguish them.
#[derive(Params)]
pub struct CompressorParams {
    /// The compression ratio. At 1.0 the compressor is disengaged.
    #[id = "ratio"]
    pub ratio: FloatParam,
    /// A `[0, 1]` scaling factor that causes the compressors for the higher registers to have lower
    /// ratios than the compressors for the lower registers. The scaling is applied logarithmically
    /// rather than linearly over the compressors. If this is set to 1.0, then the ratios will be
    /// the same for every compressor. A value of 0.5 means that at
    /// `HIGH_FREQ_RATIO_ROLLOFF_FREQUENCY` Hz, the compression ratio will be 0.5 times that as the
    /// one at 0 Hz.
    #[id = "high_freq_rolloff"]
    pub high_freq_ratio_rolloff: FloatParam,
    /// The compression knee width, in decibels.
    #[id = "knee"]
    pub knee_width_db: FloatParam,
}

impl ThresholdCurveParams {
    /// Create the threshold curve parameters for one compressor of one chain. `name_prefix`
    /// identifies both.
    pub fn new(name_prefix: &str, set_update_thresholds: Arc<dyn Fn(f32) + Send + Sync>) -> Self {
        ThresholdCurveParams {
            // These three shape the curve and share a "Thresh" prefix so they read as one group.
            // They are polynomial coefficients evaluated in the log/log domain
            // (octaves/decibels), with `ThresholdParams::threshold_db` as the intercept.
            center_frequency: FloatParam::new(
                format!("{name_prefix} Thresh Center"),
                420.0,
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20_000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_callback(set_update_thresholds.clone())
            // This includes the unit
            .with_value_to_string(formatters::v2s_f32_hz_then_khz(0))
            .with_string_to_value(formatters::s2v_f32_hz_then_khz())
            .hide_in_generic_ui(),
            curve_slope: FloatParam::new(
                format!("{name_prefix} Thresh Slope"),
                0.0,
                FloatRange::SymmetricalSkewed {
                    min: -36.0,
                    max: 36.0,
                    factor: FloatRange::skew_factor(-2.0),
                    center: 0.0,
                },
            )
            .with_callback(set_update_thresholds.clone())
            .with_unit(" dB/oct")
            .with_step_size(0.01)
            .hide_in_generic_ui(),
            curve_curve: FloatParam::new(
                format!("{name_prefix} Thresh Curve"),
                0.0,
                FloatRange::SymmetricalSkewed {
                    min: -24.0,
                    max: 24.0,
                    factor: FloatRange::skew_factor(-2.0),
                    center: 0.0,
                },
            )
            .with_callback(set_update_thresholds.clone())
            .with_unit(" dB/oct²")
            .with_step_size(0.01)
            .hide_in_generic_ui(),
            threshold_offset_db: FloatParam::new(
                format!("{name_prefix} Offset"),
                0.0,
                FloatRange::Linear {
                    min: -50.0,
                    max: 50.0,
                },
            )
            .with_callback(set_update_thresholds)
            .with_unit(" dB")
            .with_step_size(0.1)
            .hide_in_generic_ui(),
        }
    }
}

impl ChainParams {
    /// Create one chain's parameters. `chain_idx` is zero based and only used for naming.
    pub fn new(
        chain_idx: usize,
        set_update_downwards_thresholds: Arc<dyn Fn(f32) + Send + Sync>,
        set_update_upwards_thresholds: Arc<dyn Fn(f32) + Send + Sync>,
        set_update_process_weights: Arc<dyn Fn(f32) + Send + Sync>,
    ) -> Self {
        // The chains are named neutrally because what they mean depends on the stereo mode; the
        // editor labels them Left/Right or Mid/Side to match
        let name_prefix = if chain_idx == 0 { "A" } else { "B" };

        ChainParams {
            upwards: ThresholdCurveParams::new(
                &format!("{name_prefix} {UPWARDS_NAME_PREFIX}"),
                set_update_upwards_thresholds,
            ),
            downwards: ThresholdCurveParams::new(
                &format!("{name_prefix} {DOWNWARDS_NAME_PREFIX}"),
                set_update_downwards_thresholds,
            ),

            bypass_below_hz: FloatParam::new(
                format!("{name_prefix} LF Bypass"),
                LF_BYPASS_OFF_HZ,
                FloatRange::Skewed {
                    min: LF_BYPASS_OFF_HZ,
                    max: 1000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_callback(set_update_process_weights)
            // The leftmost position is off rather than a twenty hertz corner, so it says so
            .with_value_to_string(Arc::new(|value| {
                if value <= LF_BYPASS_OFF_HZ {
                    String::from("Off")
                } else if value < 1000.0 {
                    format!("{value:.0} Hz")
                } else {
                    format!("{:.2} kHz", value / 1000.0)
                }
            }))
            .with_string_to_value(Arc::new(|string| {
                let string = string.trim();
                if string.eq_ignore_ascii_case("off") {
                    return Some(LF_BYPASS_OFF_HZ);
                }
                formatters::s2v_f32_hz_then_khz()(string)
            }))
            .hide_in_generic_ui(),
        }
    }
}

impl ThresholdParams {
    /// Create a new [`ThresholdParams`] object. Changing any of the threshold parameters causes the
    /// passed compressor bank's thresholds and knee parabolas to be updated.
    pub fn new(compressor_bank: &CompressorBank) -> Self {
        let should_update_downwards_thresholds =
            compressor_bank.should_update_downwards_thresholds.clone();
        let should_update_upwards_thresholds =
            compressor_bank.should_update_upwards_thresholds.clone();
        let should_update_downwards_knee_parabolas = compressor_bank
            .should_update_downwards_knee_parabolas
            .clone();
        let should_update_upwards_knee_parabolas =
            compressor_bank.should_update_upwards_knee_parabolas.clone();
        let set_update_both_thresholds = Arc::new(move |_| {
            should_update_downwards_thresholds.store(true, Ordering::SeqCst);
            should_update_upwards_thresholds.store(true, Ordering::SeqCst);
            should_update_downwards_knee_parabolas.store(true, Ordering::SeqCst);
            should_update_upwards_knee_parabolas.store(true, Ordering::SeqCst);
        });

        let set_update_process_weights: Arc<dyn Fn(f32) + Send + Sync> = Arc::new({
            let should_update_process_weights =
                compressor_bank.should_update_process_weights.clone();
            move |_| should_update_process_weights.store(true, Ordering::SeqCst)
        });

        let set_update_both_thresholds_for_chains = set_update_both_thresholds.clone();
        let set_update_both_thresholds_for_capture = set_update_both_thresholds.clone();
        // A chain's own curve only affects that compressor, so it doesn't need to dirty the other
        let set_update_downwards_thresholds: Arc<dyn Fn(f32) + Send + Sync> = Arc::new({
            let should_update_downwards_thresholds =
                compressor_bank.should_update_downwards_thresholds.clone();
            let should_update_downwards_knee_parabolas = compressor_bank
                .should_update_downwards_knee_parabolas
                .clone();
            move |_| {
                should_update_downwards_thresholds.store(true, Ordering::SeqCst);
                should_update_downwards_knee_parabolas.store(true, Ordering::SeqCst);
            }
        });
        let set_update_upwards_thresholds: Arc<dyn Fn(f32) + Send + Sync> = Arc::new({
            let should_update_upwards_thresholds =
                compressor_bank.should_update_upwards_thresholds.clone();
            let should_update_upwards_knee_parabolas =
                compressor_bank.should_update_upwards_knee_parabolas.clone();
            move |_| {
                should_update_upwards_thresholds.store(true, Ordering::SeqCst);
                should_update_upwards_knee_parabolas.store(true, Ordering::SeqCst);
            }
        });

        ThresholdParams {
            threshold_db: FloatParam::new(
                "Global Threshold",
                -12.0,
                FloatRange::Linear {
                    min: -100.0,
                    max: 20.0,
                },
            )
            .with_callback(set_update_both_thresholds.clone())
            .with_unit(" dB")
            .with_step_size(0.1),
            // Purely an editor concern, so it needs no callback: the DSP always reads each
            // compressor's own slope and curve regardless of this value. The editor draws this
            // above the two compressor columns instead of in the threshold column, so it's hidden
            // from the generic UI to avoid showing up twice.
            slope_curve_link: BoolParam::new("Thresh Curve Link", true).hide_in_generic_ui(),

            mode: EnumParam::new("Mode", ThresholdMode::Internal)
                // Not the most efficient way to do this, but it's a bit cleaner than the
                // alternative
                .with_callback(Arc::new(move |_| set_update_both_thresholds(0.0))),
            sc_channel_link: FloatParam::new(
                "SC Detection Link",
                0.8,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_unit("%")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),

            chains: std::array::from_fn(|chain_idx| {
                ChainParams::new(
                    chain_idx,
                    set_update_downwards_thresholds.clone(),
                    set_update_upwards_thresholds.clone(),
                    set_update_process_weights.clone(),
                )
            }),
            eq: EqBankParams::new("", set_update_both_thresholds_for_chains),
            capture: CaptureParams::new(
                compressor_bank.capture.shared(),
                set_update_both_thresholds_for_capture,
            ),
        }
    }

    /// Build the [`CurveParams`] for one compressor direction. The intercept and center frequency
    /// are shared between the two directions, while the shape comes from that direction's own
    /// slope and curve parameters.
    pub fn curve_params(&self, curve: &ThresholdCurveParams) -> CurveParams {
        CurveParams {
            intercept: self.threshold_db.value(),
            center_frequency: curve.center_frequency.value(),
            // The cheeky additional attenuation is to match pink noise with the default settings.
            // When using sidechaining we explicitly don't want this because the curve should be a
            // flat offset to the sidechain input at the default settings.
            slope: curve.curve_slope.value() + self.baseline_slope(),
            curve: curve.curve_curve.value(),
        }
    }

    /// The slope of the shape the threshold curve assumes the audio has, which is what a capture
    /// stands in for.
    ///
    /// Broken out from [`Self::curve_params()`] rather than folded into it because the capture has
    /// to replace *only* this part. Crossfading against the whole polynomial would take the user's
    /// own slope and curve down with it, leaving two controls that silently did nothing.
    pub fn baseline_slope(&self) -> f32 {
        match self.mode.value() {
            ThresholdMode::Internal => PINK_NOISE_SLOPE,
            ThresholdMode::SidechainMatch | ThresholdMode::SidechainCompress => 0.0,
        }
    }
}

impl CompressorBankParams {
    /// Create compressor bank parameter objects for both the downwards and upwards compressors of
    /// `compressor`. Changing the ratio, threshold, and knee parameters will cause the compressor
    /// to recompute its values on the next processing cycle.
    pub fn new(compressor: &CompressorBank) -> Self {
        CompressorBankParams {
            downwards: Arc::new(CompressorParams::new(
                DOWNWARDS_NAME_PREFIX,
                compressor.should_update_downwards_ratios.clone(),
                compressor.should_update_downwards_knee_parabolas.clone(),
            )),
            upwards: Arc::new(CompressorParams::new(
                UPWARDS_NAME_PREFIX,
                compressor.should_update_upwards_ratios.clone(),
                compressor.should_update_upwards_knee_parabolas.clone(),
            )),
        }
    }
}

impl CompressorParams {
    /// Create a new [`CompressorParams`] object with a prefix for all parameter names. Changing
    /// any of the ratio or knee parameters causes the passed atomics to be updated. These should
    /// be taken from a [`CompressorBank`] so the parameters are linked to it.
    pub fn new(
        name_prefix: &str,
        should_update_ratios: Arc<AtomicBool>,
        should_update_knee_parabolas: Arc<AtomicBool>,
    ) -> Self {
        let set_update_ratios = Arc::new({
            let should_update_knee_parabolas = should_update_knee_parabolas.clone();
            move |_| {
                should_update_ratios.store(true, Ordering::SeqCst);
                should_update_knee_parabolas.store(true, Ordering::SeqCst);
            }
        });
        let set_update_knee_parabolas = Arc::new(move |_| {
            should_update_knee_parabolas.store(true, Ordering::SeqCst);
        });

        CompressorParams {
            ratio: FloatParam::new(
                format!("{name_prefix} Ratio"),
                1.0,
                FloatRange::Skewed {
                    min: 1.0,
                    max: 500.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_callback(set_update_ratios.clone())
            .with_step_size(0.01)
            .with_value_to_string(formatters::v2s_compression_ratio(2))
            .with_string_to_value(formatters::s2v_compression_ratio()),
            high_freq_ratio_rolloff: FloatParam::new(
                format!("{name_prefix} Hi-Freq Rolloff"),
                // TODO: Bit of a hacky way to set the default values differently for upwards and
                //       downwards compressors
                if name_prefix == UPWARDS_NAME_PREFIX {
                    0.75
                } else {
                    // When used subtly, no rolloff is usually better for downwards compression
                    0.0
                },
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_callback(set_update_ratios)
            .with_unit("%")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            knee_width_db: FloatParam::new(
                format!("{name_prefix} Knee"),
                6.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 36.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_callback(set_update_knee_parabolas)
            .with_unit(" dB")
            .with_step_size(0.1),
        }
    }
}

impl CompressorBank {
    /// Set up the compressor for the given channel count and maximum FFT window size. The
    /// compressors won't be initialized yet.
    pub fn new(
        analyzer_input_data: triple_buffer::Input<AnalyzerData>,
        num_channels: usize,
        max_window_size: usize,
    ) -> Self {
        let complex_buffer_len = max_window_size / 2 + 1;

        CompressorBank {
            should_update_downwards_thresholds: Arc::new(AtomicBool::new(true)),
            should_update_upwards_thresholds: Arc::new(AtomicBool::new(true)),
            should_update_downwards_ratios: Arc::new(AtomicBool::new(true)),
            should_update_upwards_ratios: Arc::new(AtomicBool::new(true)),
            should_update_downwards_knee_parabolas: Arc::new(AtomicBool::new(true)),
            should_update_upwards_knee_parabolas: Arc::new(AtomicBool::new(true)),
            should_update_process_weights: Arc::new(AtomicBool::new(true)),

            ln_freqs: Vec::with_capacity(complex_buffer_len),
            freqs: Vec::with_capacity(complex_buffer_len),
            eq_power: Vec::with_capacity(complex_buffer_len),

            downwards_thresholds_db: std::array::from_fn(|_| {
                Vec::with_capacity(complex_buffer_len)
            }),
            downwards_ratios: Vec::with_capacity(complex_buffer_len),
            downwards_knee_parabola_scale: std::array::from_fn(|_| {
                Vec::with_capacity(complex_buffer_len)
            }),
            downwards_knee_parabola_intercept: std::array::from_fn(|_| {
                Vec::with_capacity(complex_buffer_len)
            }),

            upwards_thresholds_db: std::array::from_fn(|_| Vec::with_capacity(complex_buffer_len)),
            upwards_ratios: Vec::with_capacity(complex_buffer_len),
            upwards_knee_parabola_scale: std::array::from_fn(|_| {
                Vec::with_capacity(complex_buffer_len)
            }),
            upwards_knee_parabola_intercept: std::array::from_fn(|_| {
                Vec::with_capacity(complex_buffer_len)
            }),

            process_weights: std::array::from_fn(|_| Vec::with_capacity(complex_buffer_len)),

            envelopes: vec![Vec::with_capacity(complex_buffer_len); num_channels],
            spectrum_magnitudes: vec![Vec::with_capacity(complex_buffer_len); num_channels],
            envelope_followers_timing_scale: 0.0,
            sidechain_spectrum_magnitudes: vec![
                Vec::with_capacity(complex_buffer_len);
                num_channels
            ],
            window_size: 0,
            sample_rate: 1.0,

            capture: CaptureBank::new(),
            capture_smoothed: vec![0.0; CAPTURE_GRID_LEN],

            analyzer_input_data,
        }
    }

    /// Change the capacities of the internal buffers to fit new parameters. Use the
    /// `.reset_for_size()` method to clear the buffers and set the current window size.
    pub fn update_capacity(&mut self, num_channels: usize, max_window_size: usize) {
        let complex_buffer_len = max_window_size / 2 + 1;

        self.ln_freqs
            .reserve_exact(complex_buffer_len.saturating_sub(self.ln_freqs.len()));
        self.freqs
            .reserve_exact(complex_buffer_len.saturating_sub(self.freqs.len()));
        self.eq_power
            .reserve_exact(complex_buffer_len.saturating_sub(self.eq_power.len()));

        for buffer in self.process_weights.iter_mut() {
            buffer.reserve_exact(complex_buffer_len.saturating_sub(buffer.len()));
        }
        for buffer in self.downwards_thresholds_db.iter_mut() {
            buffer.reserve_exact(complex_buffer_len.saturating_sub(buffer.len()));
        }
        self.downwards_ratios
            .reserve_exact(complex_buffer_len.saturating_sub(self.downwards_ratios.len()));
        for buffer in self.downwards_knee_parabola_scale.iter_mut() {
            buffer.reserve_exact(complex_buffer_len.saturating_sub(buffer.len()));
        }
        for buffer in self.downwards_knee_parabola_intercept.iter_mut() {
            buffer.reserve_exact(complex_buffer_len.saturating_sub(buffer.len()));
        }

        for buffer in self.upwards_thresholds_db.iter_mut() {
            buffer.reserve_exact(complex_buffer_len.saturating_sub(buffer.len()));
        }
        self.upwards_ratios
            .reserve_exact(complex_buffer_len.saturating_sub(self.upwards_ratios.len()));
        for buffer in self.upwards_knee_parabola_scale.iter_mut() {
            buffer.reserve_exact(complex_buffer_len.saturating_sub(buffer.len()));
        }
        for buffer in self.upwards_knee_parabola_intercept.iter_mut() {
            buffer.reserve_exact(complex_buffer_len.saturating_sub(buffer.len()));
        }

        self.envelopes.resize_with(num_channels, Vec::new);
        for envelopes in self.envelopes.iter_mut() {
            envelopes.reserve_exact(complex_buffer_len.saturating_sub(envelopes.len()));
        }

        self.spectrum_magnitudes.resize_with(num_channels, Vec::new);
        for magnitudes in self.spectrum_magnitudes.iter_mut() {
            magnitudes.reserve_exact(complex_buffer_len.saturating_sub(magnitudes.len()));
        }

        self.sidechain_spectrum_magnitudes
            .resize_with(num_channels, Vec::new);
        for magnitudes in self.sidechain_spectrum_magnitudes.iter_mut() {
            magnitudes.reserve_exact(complex_buffer_len.saturating_sub(magnitudes.len()));
        }
    }

    /// Resize the number of compressors to match the current window size. Also precomputes the
    /// 2-log frequencies for each bin.
    ///
    /// If the window size is larger than the maximum window size, then this will allocate.
    pub fn resize(&mut self, buffer_config: &BufferConfig, window_size: usize) {
        let complex_buffer_len = window_size / 2 + 1;

        // These 2-log frequencies are needed when updating the compressor parameters, so we'll just
        // precompute them to avoid having to repeat the same expensive computations all the time
        self.ln_freqs.resize(complex_buffer_len, 0.0);
        self.freqs.resize(complex_buffer_len, 0.0);
        self.eq_power.resize(complex_buffer_len, 1.0);
        // The first one should always stay at zero, `0.0f32.ln() == NaN`.
        for (i, (ln_freq, freq_hz)) in self
            .ln_freqs
            .iter_mut()
            .zip(self.freqs.iter_mut())
            .enumerate()
            .skip(1)
        {
            let freq = (i as f32 / window_size as f32) * buffer_config.sample_rate;
            *ln_freq = freq.ln();
            *freq_hz = freq;
        }

        for buffer in self.process_weights.iter_mut() {
            buffer.resize(complex_buffer_len, 1.0);
        }
        for buffer in self.downwards_thresholds_db.iter_mut() {
            buffer.resize(complex_buffer_len, 1.0);
        }
        self.downwards_ratios.resize(complex_buffer_len, 1.0);
        for buffer in self.downwards_knee_parabola_scale.iter_mut() {
            buffer.resize(complex_buffer_len, 1.0);
        }
        for buffer in self.downwards_knee_parabola_intercept.iter_mut() {
            buffer.resize(complex_buffer_len, 1.0);
        }

        for buffer in self.upwards_thresholds_db.iter_mut() {
            buffer.resize(complex_buffer_len, 1.0);
        }
        self.upwards_ratios.resize(complex_buffer_len, 1.0);
        for buffer in self.upwards_knee_parabola_scale.iter_mut() {
            buffer.resize(complex_buffer_len, 1.0);
        }
        for buffer in self.upwards_knee_parabola_intercept.iter_mut() {
            buffer.resize(complex_buffer_len, 1.0);
        }

        for envelopes in self.envelopes.iter_mut() {
            envelopes.resize(complex_buffer_len, ENVELOPE_INIT_VALUE);
        }

        for magnitudes in self.spectrum_magnitudes.iter_mut() {
            magnitudes.resize(complex_buffer_len, 0.0);
        }

        for magnitudes in self.sidechain_spectrum_magnitudes.iter_mut() {
            magnitudes.resize(complex_buffer_len, 0.0);
        }

        self.window_size = window_size;
        self.sample_rate = buffer_config.sample_rate;

        // The compressors need to be updated on the next processing cycle
        self.should_update_downwards_thresholds
            .store(true, Ordering::SeqCst);
        self.should_update_upwards_thresholds
            .store(true, Ordering::SeqCst);
        self.should_update_downwards_ratios
            .store(true, Ordering::SeqCst);
        self.should_update_upwards_ratios
            .store(true, Ordering::SeqCst);
        self.should_update_downwards_knee_parabolas
            .store(true, Ordering::SeqCst);
        self.should_update_upwards_knee_parabolas
            .store(true, Ordering::SeqCst);
        self.should_update_process_weights
            .store(true, Ordering::SeqCst);
    }

    /// Clear out the envelope followers.
    pub fn reset(&mut self) {
        // This will make the timings instant for the first iteration after a reset and then slowly
        // fade the timings back to their intended values so the envelope followers can settle in.
        // Otherwise suspending and resetting the plugin, or changing the window size, may result in
        // some huge spikes.
        self.envelope_followers_timing_scale = 0.0;

        // Sidechain data doesn't need to be reset as it will be overwritten immediately before use
    }

    /// Apply the magnitude compression to a buffer of FFT bins. The compressors are first updated
    /// if needed. The overlap amount is needed to compute the effective sample rate. The
    /// `first_non_dc_bin` argument is used to avoid upwards compression on the DC bins, or the
    /// neighbouring bins the DC signal may have been convolved into because of the Hann window
    /// function.
    pub fn process(
        &mut self,
        buffer: &mut [Complex32],
        channel_idx: usize,
        params: &SpectralCompressorParams,
        overlap_times: usize,
        first_non_dc_bin: usize,
    ) {
        nih_debug_assert_eq!(buffer.len(), self.ln_freqs.len());

        // The gain difference/reduction amounts are accumulated in `self.analyzer_input_data`. When
        // processing the last channel, this data is divided by the channel count, the envelope
        // follower data is added, and the data is then sent to the editor so it can be displayed.
        // `analyzer_input_data` contains excess capacity so it can handle any supported window
        // size, so all operations on it are limited to the actual number of used bins.
        let num_bins = buffer.len();
        let num_channels = self.sidechain_spectrum_magnitudes.len();
        let should_update_analyzer_data = params.editor_state.is_open();
        if should_update_analyzer_data && channel_idx == 0 {
            // A mono layout only feeds the first chain, so the second would otherwise keep showing
            // whatever it last held
            let analyzer_input_data = self.analyzer_input_data.input_buffer();
            for chain in analyzer_input_data.gain_difference_db.iter_mut() {
                chain[..num_bins].fill(0.0);
            }
            for chain in analyzer_input_data.envelope_followers.iter_mut() {
                chain[..num_bins].fill(0.0);
            }
        }

        // Housekeeping first, so that stopping a capture is reflected in the curves this same block
        let stereo_mode_idx = params.global.stereo_mode.value().to_index();
        self.capture.poll(stereo_mode_idx);

        self.update_if_needed(params);

        // Both chains are always captured, so switching between left/right and mid/side later on
        // never throws a curve away. This runs before the match below because that is where the
        // bins get scaled: what belongs in a capture is the input, not the compressed output.
        if self.capture.is_active() {
            let chain_idx = chain_for_channel(channel_idx);
            match params.threshold.capture.source.value() {
                CaptureSource::Main => {
                    self.capture
                        .push_bins(buffer, &self.ln_freqs, stereo_mode_idx, chain_idx)
                }
                CaptureSource::Sidechain => self.capture.push_frame(
                    &self.sidechain_spectrum_magnitudes[channel_idx],
                    &self.ln_freqs,
                    stereo_mode_idx,
                    chain_idx,
                ),
            }
        }

        match params.threshold.mode.value() {
            ThresholdMode::Internal => {
                self.update_envelopes(buffer, channel_idx, params, overlap_times);
                self.compress(buffer, channel_idx, params, first_non_dc_bin)
            }
            ThresholdMode::SidechainMatch => {
                self.update_envelopes(buffer, channel_idx, params, overlap_times);
                self.compress_sidechain_match(buffer, channel_idx, params, first_non_dc_bin)
            }
            ThresholdMode::SidechainCompress => {
                // This mode uses regular compression, but the envelopes are computed from the
                // sidechain input magnitudes. These are already set in `process_sidechain`. This
                // separate envelope updating function is needed for the channel linking.
                self.update_envelopes_sidechain(channel_idx, params, overlap_times);
                self.compress(buffer, channel_idx, params, first_non_dc_bin)
            }
        };

        // When processing the last channel we can finalize the spectrum analyzer data and send it
        // to the editor for display
        if should_update_analyzer_data && channel_idx == num_channels - 1 {
            let analyzer_input_data = self.analyzer_input_data.input_buffer();

            analyzer_input_data.num_bins = num_bins;

            // After filling the object with data it can be sent to the editor. This happens
            // automatically when using the `.write()` interface, but since `AnalyzerData` contains
            // a lot of padding and we only use the first `num_bins` of the arrays that would be a
            // bit wasteful.
            self.analyzer_input_data.publish();
        }
    }

    /// Set the sidechain frequency spectrum magnitudes just before a [`process()`][Self::process()]
    /// call. These will be multiplied with the existing compressor thresholds and knee values to
    /// get the effective values for use with sidechaining.
    pub fn process_sidechain(&mut self, sc_buffer: &[Complex32], channel_idx: usize) {
        nih_debug_assert_eq!(sc_buffer.len(), self.ln_freqs.len());

        self.update_sidechain_spectra(sc_buffer, channel_idx);
    }

    /// Update the envelope followers based on the bin magnitudes.
    fn update_envelopes(
        &mut self,
        buffer: &[Complex32],
        channel_idx: usize,
        params: &SpectralCompressorParams,
        overlap_times: usize,
    ) {
        let effective_sample_rate =
            self.sample_rate / (self.window_size as f32 / overlap_times as f32);

        // The timings are scaled by `self.envelope_followers_timing_scale` to allow the envelope
        // followers to settle in quicker after a reset
        let attack_ms =
            params.global.compressor_attack_ms.value() * self.envelope_followers_timing_scale;
        let release_ms =
            params.global.compressor_release_ms.value() * self.envelope_followers_timing_scale;

        // This needs to gradually fade from 0.0 back to 1.0 after a reset
        if self.envelope_followers_timing_scale < 1.0 && channel_idx == self.envelopes.len() - 1 {
            let delta =
                ((ENVELOPE_FOLLOWER_TIMING_FADE_MS / 1000.0) * effective_sample_rate).recip();
            self.envelope_followers_timing_scale =
                (self.envelope_followers_timing_scale + delta).min(1.0);
        }

        // The coefficient the old envelope value is multiplied by when the current rectified sample
        // value is above the envelope's value. The 0 to 1 step response retains 36.8% of the old
        // value after the attack time has elapsed, and current value is 63.2% of the way towards 1.
        // The effective sample rate needs to compensate for the periodic nature of the STFT
        // operation. Since with a 2048 sample window and 4x overlap, you'd run this function once
        // for every 512 samples.
        let attack_old_t = if attack_ms == 0.0 {
            0.0
        } else {
            (-1.0 / (attack_ms / 1000.0 * effective_sample_rate)).exp()
        };
        let attack_new_t = 1.0 - attack_old_t;
        // The same as `attack_old_t`, but for the release phase of the envelope follower
        let release_old_t = if release_ms == 0.0 {
            0.0
        } else {
            (-1.0 / (release_ms / 1000.0 * effective_sample_rate)).exp()
        };
        let release_new_t = 1.0 - release_old_t;

        // Record this channel's magnitudes so the other channels can fold them into their own
        // detection below
        for (bin, magnitude) in buffer
            .iter()
            .zip(self.spectrum_magnitudes[channel_idx].iter_mut())
        {
            *magnitude = bin.norm();
        }

        let num_channels = self.spectrum_magnitudes.len() as f32;
        let other_channels_t = params.global.channel_link.value() / num_channels;
        let this_channel_t = 1.0 - (other_channels_t * (num_channels - 1.0));

        // The common case is fully independent channels, which doesn't need to look at the others
        // at all
        if other_channels_t == 0.0 {
            for (bin, envelope) in buffer.iter().zip(self.envelopes[channel_idx].iter_mut()) {
                let magnitude = bin.norm();
                if *envelope > magnitude {
                    // Release stage
                    *envelope = (release_old_t * *envelope) + (release_new_t * magnitude);
                } else {
                    // Attack stage
                    *envelope = (attack_old_t * *envelope) + (attack_new_t * magnitude);
                }
            }

            return;
        }

        // NOTE: The channels are processed one after another within a block, so the channels that
        //       haven't run yet still hold the previous block's magnitudes here. At one STFT hop
        //       that lag is orders of magnitude shorter than the envelope timings, and avoiding it
        //       would mean a separate analysis pass over every channel.
        for (bin_idx, envelope) in self.envelopes[channel_idx].iter_mut().enumerate() {
            let magnitude = linked_magnitude(
                &self.spectrum_magnitudes,
                channel_idx,
                bin_idx,
                this_channel_t,
                other_channels_t,
            );

            if *envelope > magnitude {
                *envelope = (release_old_t * *envelope) + (release_new_t * magnitude);
            } else {
                *envelope = (attack_old_t * *envelope) + (attack_new_t * magnitude);
            }
        }
    }

    /// The same as [`update_envelopes()`][Self::update_envelopes()], but based on the previously
    /// set sidechain bin magnitudes. This allows for channel linking.
    /// [`process_sidechain()`][Self::process_sidechain()] needs to be called for all channels
    /// before this function can be used to set the magnitude spectra.
    fn update_envelopes_sidechain(
        &mut self,
        channel_idx: usize,
        params: &SpectralCompressorParams,
        overlap_times: usize,
    ) {
        let effective_sample_rate =
            self.sample_rate / (self.window_size as f32 / overlap_times as f32);

        // The timings are scaled by `self.envelope_followers_timing_scale` to allow the envelope
        // followers to settle in quicker after a reset
        let attack_ms =
            params.global.compressor_attack_ms.value() * self.envelope_followers_timing_scale;
        let release_ms =
            params.global.compressor_release_ms.value() * self.envelope_followers_timing_scale;

        // This needs to gradually fade from 0.0 back to 1.0 after a reset
        if self.envelope_followers_timing_scale < 1.0 && channel_idx == self.envelopes.len() - 1 {
            let delta =
                ((ENVELOPE_FOLLOWER_TIMING_FADE_MS / 1000.0) * effective_sample_rate).recip();
            self.envelope_followers_timing_scale =
                (self.envelope_followers_timing_scale + delta).min(1.0);
        }

        // See `update_envelopes()`
        let attack_old_t = if attack_ms == 0.0 {
            0.0
        } else {
            (-1.0 / (attack_ms / 1000.0 * effective_sample_rate)).exp()
        };
        let attack_new_t = 1.0 - attack_old_t;
        let release_old_t = if release_ms == 0.0 {
            0.0
        } else {
            (-1.0 / (release_ms / 1000.0 * effective_sample_rate)).exp()
        };
        let release_new_t = 1.0 - release_old_t;

        // For the channel linking
        let num_channels = self.sidechain_spectrum_magnitudes.len() as f32;
        let other_channels_t = params.threshold.sc_channel_link.value() / num_channels;
        let this_channel_t = 1.0 - (other_channels_t * (num_channels - 1.0));

        for (bin_idx, envelope) in self.envelopes[channel_idx].iter_mut().enumerate() {
            // In this mode the envelopes are set based on the sidechain signal, taking channel
            // linking into account
            let sidechain_magnitude: f32 = self
                .sidechain_spectrum_magnitudes
                .iter()
                .enumerate()
                .map(|(sidechain_channel_idx, magnitudes)| {
                    let t = if sidechain_channel_idx == channel_idx {
                        this_channel_t
                    } else {
                        other_channels_t
                    };

                    unsafe { magnitudes.get_unchecked(bin_idx) * t }
                })
                .sum::<f32>();

            if *envelope > sidechain_magnitude {
                // Release stage
                *envelope = (release_old_t * *envelope) + (release_new_t * sidechain_magnitude);
            } else {
                // Attack stage
                *envelope = (attack_old_t * *envelope) + (attack_new_t * sidechain_magnitude);
            }
        }
    }

    /// Update the spectral data using the sidechain input
    fn update_sidechain_spectra(&mut self, sc_buffer: &[Complex32], channel_idx: usize) {
        nih_debug_assert!(channel_idx < self.sidechain_spectrum_magnitudes.len());

        for (bin, magnitude) in sc_buffer
            .iter()
            .zip(self.sidechain_spectrum_magnitudes[channel_idx].iter_mut())
        {
            *magnitude = bin.norm();
        }
    }

    /// Actually do the thing. [`Self::update_envelopes()`] or
    /// [`Self::update_envelopes_sidechain()`] must have been called before calling this.
    ///
    /// # Panics
    ///
    /// Panics if the buffer does not have the same length as the one that was passed to the last
    /// `resize()` call.
    fn compress(
        &mut self,
        buffer: &mut [Complex32],
        channel_idx: usize,
        params: &SpectralCompressorParams,
        first_non_dc_bin: usize,
    ) {
        // The gain reduction values are always added to the arrays stored in this object. This
        // makes it possible to visualize the gain reduction without a lot of conditionals.
        let analyzer_input_data = self.analyzer_input_data.input_buffer();

        let downwards_knee_width_db = params.compressors.downwards.knee_width_db.value();
        let upwards_knee_width_db = params.compressors.upwards.knee_width_db.value();
        let chain_idx = chain_for_channel(channel_idx);

        assert!(analyzer_input_data.gain_difference_db[chain_idx].len() >= buffer.len());
        assert!(analyzer_input_data.envelope_followers[chain_idx].len() >= buffer.len());
        assert!(self.process_weights[chain_idx].len() == buffer.len());
        assert!(self.downwards_thresholds_db[chain_idx].len() == buffer.len());
        assert!(self.downwards_ratios.len() == buffer.len());
        assert!(self.downwards_knee_parabola_scale[chain_idx].len() == buffer.len());
        assert!(self.downwards_knee_parabola_intercept[chain_idx].len() == buffer.len());
        assert!(self.upwards_thresholds_db[chain_idx].len() == buffer.len());
        assert!(self.upwards_ratios.len() == buffer.len());
        assert!(self.upwards_knee_parabola_scale[chain_idx].len() == buffer.len());
        assert!(self.upwards_knee_parabola_intercept[chain_idx].len() == buffer.len());
        // NOTE: In the sidechain compression mode these envelopes are computed from the sidechain
        //       signal instead of the main input
        for (bin_idx, (bin, envelope)) in buffer
            .iter_mut()
            .zip(self.envelopes[channel_idx].iter())
            .enumerate()
        {
            // We'll apply the transfer curve to the envelope signal, and then scale the complex
            // `bin` by the gain difference
            let envelope_db = util::gain_to_db_fast_epsilon(*envelope);

            // SAFETY: These sizes were asserted above
            let downwards_threshold_db =
                unsafe { self.downwards_thresholds_db[chain_idx].get_unchecked(bin_idx) };
            let downwards_ratio = unsafe { self.downwards_ratios.get_unchecked(bin_idx) };
            let downwards_knee_parabola_scale =
                unsafe { self.downwards_knee_parabola_scale[chain_idx].get_unchecked(bin_idx) };
            let downwards_knee_parabola_intercept =
                unsafe { self.downwards_knee_parabola_intercept[chain_idx].get_unchecked(bin_idx) };
            let downwards_compressed = compress_downwards(
                envelope_db,
                *downwards_threshold_db,
                *downwards_ratio,
                downwards_knee_width_db,
                *downwards_knee_parabola_scale,
                *downwards_knee_parabola_intercept,
            );

            // Upwards compression should not happen when the signal is _too_ quiet as we'd only be
            // amplifying noise. We also don't want to amplify DC noise and super low frequencies.
            let upwards_threshold_db =
                unsafe { self.upwards_thresholds_db[chain_idx].get_unchecked(bin_idx) };
            let upwards_ratio = unsafe { self.upwards_ratios.get_unchecked(bin_idx) };
            let upwards_knee_parabola_scale =
                unsafe { self.upwards_knee_parabola_scale[chain_idx].get_unchecked(bin_idx) };
            let upwards_knee_parabola_intercept =
                unsafe { self.upwards_knee_parabola_intercept[chain_idx].get_unchecked(bin_idx) };
            let upwards_compressed = if bin_idx >= first_non_dc_bin
                && *upwards_ratio != 1.0
                && envelope_db > util::MINUS_INFINITY_DB
            {
                compress_upwards(
                    envelope_db,
                    *upwards_threshold_db,
                    *upwards_ratio,
                    upwards_knee_width_db,
                    *upwards_knee_parabola_scale,
                    *upwards_knee_parabola_intercept,
                )
            } else {
                envelope_db
            };

            // If the comprssed output is -10 dBFS and the envelope follower was at -6 dBFS, then we
            // want to apply -4 dB of gain to the bin. The weight is what the low frequency bypass
            // acts through, and it is applied here rather than to the thresholds so that both
            // directions stop together.
            let gain_difference_db = (downwards_compressed + upwards_compressed
                - (envelope_db * 2.0))
                * unsafe {
                    *self
                        .process_weights
                        .get_unchecked(chain_idx)
                        .get_unchecked(bin_idx)
                };
            unsafe {
                *analyzer_input_data
                    .gain_difference_db
                    .get_unchecked_mut(chain_idx)
                    .get_unchecked_mut(bin_idx) = gain_difference_db;
                *analyzer_input_data
                    .envelope_followers
                    .get_unchecked_mut(chain_idx)
                    .get_unchecked_mut(bin_idx) = *envelope;
            }

            *bin *= util::db_to_gain_fast(gain_difference_db);
        }
    }

    /// The same as [`compress()`][Self::compress()], but multiplying the threshold and knee values
    /// with the sidechain gains.
    ///
    /// # Panics
    ///
    /// Panics if the buffer does not have the same length as the one that was passed to the last
    /// `resize()` call.
    fn compress_sidechain_match(
        &mut self,
        buffer: &mut [Complex32],
        channel_idx: usize,
        params: &SpectralCompressorParams,
        first_non_dc_bin: usize,
    ) {
        // See `compress()`
        let analyzer_input_data = self.analyzer_input_data.input_buffer();

        let downwards_knee_width_db = params.compressors.downwards.knee_width_db.value();
        let upwards_knee_width_db = params.compressors.upwards.knee_width_db.value();
        let chain_idx = chain_for_channel(channel_idx);

        // For the channel linking
        let num_channels = self.sidechain_spectrum_magnitudes.len() as f32;
        let other_channels_t = params.threshold.sc_channel_link.value() / num_channels;
        let this_channel_t = 1.0 - (other_channels_t * (num_channels - 1.0));

        assert!(analyzer_input_data.gain_difference_db[chain_idx].len() >= buffer.len());
        assert!(analyzer_input_data.envelope_followers[chain_idx].len() >= buffer.len());
        assert!(self.sidechain_spectrum_magnitudes[channel_idx].len() == buffer.len());
        assert!(self.process_weights[chain_idx].len() == buffer.len());
        assert!(self.downwards_thresholds_db[chain_idx].len() == buffer.len());
        assert!(self.downwards_ratios.len() == buffer.len());
        assert!(self.upwards_thresholds_db[chain_idx].len() == buffer.len());
        assert!(self.upwards_ratios.len() == buffer.len());
        for (bin_idx, (bin, envelope)) in buffer
            .iter_mut()
            .zip(self.envelopes[channel_idx].iter())
            .enumerate()
        {
            let envelope_db = util::gain_to_db_fast_epsilon(*envelope);

            // The idea here is that we scale the compressor thresholds/knee values by the sidechain
            // signal, thus sort of creating a dynamic multiband compressor
            let sidechain_scale: f32 = self
                .sidechain_spectrum_magnitudes
                .iter()
                .enumerate()
                .map(|(sidechain_channel_idx, magnitudes)| {
                    let t = if sidechain_channel_idx == channel_idx {
                        this_channel_t
                    } else {
                        other_channels_t
                    };

                    unsafe { magnitudes.get_unchecked(bin_idx) * t }
                })
                .sum::<f32>()
                // The thresholds may never reach zero as they are used in divisions
                .max(f32::EPSILON);
            let sidechain_scale_db = util::gain_to_db_fast_epsilon(sidechain_scale);

            // Notice how the threshold and knee values are scaled here
            let downwards_threshold_db = unsafe {
                self.downwards_thresholds_db[chain_idx].get_unchecked(bin_idx) + sidechain_scale_db
            }
            .max(util::MINUS_INFINITY_DB);
            let downwards_ratio = unsafe { self.downwards_ratios.get_unchecked(bin_idx) };
            // Because the thresholds are scaled based on the sidechain input, we also need to
            // recompute the knee coefficients
            let (downwards_knee_parabola_scale, downwards_knee_parabola_intercept) =
                downwards_soft_knee_coefficients(
                    downwards_threshold_db,
                    downwards_knee_width_db,
                    *downwards_ratio,
                );
            let downwards_compressed = compress_downwards(
                envelope_db,
                downwards_threshold_db,
                *downwards_ratio,
                downwards_knee_width_db,
                downwards_knee_parabola_scale,
                downwards_knee_parabola_intercept,
            );

            let upwards_threshold_db = unsafe {
                self.upwards_thresholds_db[chain_idx].get_unchecked(bin_idx) + sidechain_scale_db
            }
            .max(util::MINUS_INFINITY_DB);
            let upwards_ratio = unsafe { self.upwards_ratios.get_unchecked(bin_idx) };
            let upwards_compressed = if bin_idx >= first_non_dc_bin
                && *upwards_ratio != 1.0
                && envelope_db > util::MINUS_INFINITY_DB
            {
                let (upwards_knee_parabola_scale, upwards_knee_parabola_intercept) =
                    upwards_soft_knee_coefficients(
                        upwards_threshold_db,
                        upwards_knee_width_db,
                        *upwards_ratio,
                    );
                compress_upwards(
                    envelope_db,
                    upwards_threshold_db,
                    *upwards_ratio,
                    upwards_knee_width_db,
                    upwards_knee_parabola_scale,
                    upwards_knee_parabola_intercept,
                )
            } else {
                envelope_db
            };

            // If the comprssed output is -10 dBFS and the envelope follower was at -6 dBFS, then we
            // want to apply -4 dB of gain to the bin. The weight is what the low frequency bypass
            // acts through, and it is applied here rather than to the thresholds so that both
            // directions stop together.
            let gain_difference_db = (downwards_compressed + upwards_compressed
                - (envelope_db * 2.0))
                * unsafe {
                    *self
                        .process_weights
                        .get_unchecked(chain_idx)
                        .get_unchecked(bin_idx)
                };
            unsafe {
                *analyzer_input_data
                    .gain_difference_db
                    .get_unchecked_mut(chain_idx)
                    .get_unchecked_mut(bin_idx) = gain_difference_db;
                *analyzer_input_data
                    .envelope_followers
                    .get_unchecked_mut(chain_idx)
                    .get_unchecked_mut(bin_idx) = *envelope;
            }

            *bin *= util::db_to_gain_fast(gain_difference_db);
        }
    }

    /// Fill `thresholds_db` with one compressor's threshold curve: the quadratic polynomial plus
    /// the contribution of every active EQ node, offset by that compressor's threshold offset.
    ///
    /// The EQ nodes are accumulated as squared magnitudes and converted to decibels only once per
    /// bin at the end. Summing each node's decibels instead would cost a logarithm per node per
    /// bin, which is where practically all of the time would go.
    fn recompute_thresholds(
        freqs: &[f32],
        ln_freqs: &[f32],
        eq_power: &mut [f32],
        thresholds_db: &mut [f32],
        curve_params: &CurveParams,
        eq_params: &EqCurveParams,
        capture: Option<&CaptureBlend>,
        direction: CompressorDirection,
        chain_idx: usize,
        intercept_db: f32,
    ) {
        let curve = Curve::new(curve_params);
        let eq_curve = EqCurve::new(eq_params, direction, chain_idx);

        // A capture replaces the baseline tilt the curve assumes, leaving the slope, the curve, and
        // every node to apply on top as they always did. Adding a zero is exact, so a curve with no
        // capture in play is bit for bit what it was before any of this existed.
        let capture_delta = |ln_freq: f32| match capture {
            Some(capture) => capture.delta_ln(ln_freq),
            None => 0.0,
        };

        // Skipping the scratch buffer entirely when every node is off keeps the common case as
        // cheap as it was before the nodes existed
        if eq_curve.is_empty() {
            for (ln_freq, threshold_db) in ln_freqs.iter().zip(thresholds_db.iter_mut()) {
                let polynomial_db = curve.evaluate_ln(*ln_freq);
                *threshold_db = (polynomial_db + capture_delta(*ln_freq) + intercept_db)
                    .max(util::MINUS_INFINITY_DB);
            }
            return;
        }

        eq_power.fill(1.0);
        eq_curve.accumulate_power(freqs, eq_power);
        for ((ln_freq, eq_power), threshold_db) in ln_freqs
            .iter()
            .zip(eq_power.iter())
            .zip(thresholds_db.iter_mut())
        {
            let polynomial_db = curve.evaluate_ln(*ln_freq);
            *threshold_db =
                (polynomial_db + capture_delta(*ln_freq) + power_to_db(*eq_power) + intercept_db)
                    .max(util::MINUS_INFINITY_DB);
        }
    }

    /// Fill `weights` with how much of the computed gain change each bin should receive.
    ///
    /// Everything below the corner comes out at zero, which is a true bypass rather than a very
    /// high threshold: a threshold that stops downwards compression makes upwards compression lift
    /// *harder*, so there is no threshold that means "leave this alone".
    ///
    /// The fade above the corner matters more than it looks. Switching compression off between one
    /// bin and the next is a step in the frequency response, and a step in frequency is a long tail
    /// in time.
    fn recompute_process_weights(ln_freqs: &[f32], weights: &mut [f32], bypass_below_hz: f32) {
        if bypass_below_hz <= LF_BYPASS_OFF_HZ {
            weights.fill(1.0);
            return;
        }

        let ln_corner = bypass_below_hz.ln();
        let ln_full = ln_corner + (LF_BYPASS_FADE_OCTAVES * std::f32::consts::LN_2);
        for (ln_freq, weight) in ln_freqs.iter().zip(weights.iter_mut()) {
            *weight = smoothstep(ln_corner, ln_full, *ln_freq);
        }
    }

    /// Smooth the captured curve for `chain_idx` into [`Self::capture_smoothed`], and report
    /// whether there is one to blend in at all.
    ///
    /// Smoothing happens here rather than at capture time so that changing it does not mean
    /// capturing all over again.
    fn prepare_capture_curve(
        &mut self,
        params: &SpectralCompressorParams,
        chain_idx: usize,
    ) -> bool {
        if params.threshold.capture.amount.value() <= 0.0 {
            return false;
        }

        let stereo_mode_idx = params.global.stereo_mode.value().to_index();
        match self.capture.curve(stereo_mode_idx, chain_idx) {
            Some(raw) => {
                smooth_into(
                    raw,
                    &mut self.capture_smoothed,
                    params.threshold.capture.smoothing_octaves.value(),
                );
                true
            }
            None => false,
        }
    }

    /// Build the blend for one chain's threshold curve. [`Self::prepare_capture_curve()`] must
    /// have been called for the same chain first, and must have returned `true`.
    ///
    /// Takes the smoothed buffer rather than `&self` so that the threshold array it feeds can be
    /// borrowed mutably at the same time.
    fn capture_blend<'a>(
        capture_smoothed: &'a [f32],
        params: &SpectralCompressorParams,
        curve_params: &CurveParams,
    ) -> CaptureBlend<'a> {
        CaptureBlend::new(
            capture_smoothed,
            curve_params.center_frequency.ln(),
            params.threshold.baseline_slope(),
            params.threshold.capture.amount.value(),
            params.threshold.capture.low_frequency.value(),
            params.threshold.capture.high_frequency.value(),
        )
    }

    /// Update the compressors if needed. This is called just before processing, and the compressors
    /// are updated in accordance to the atomic flags set on this struct.
    fn update_if_needed(&mut self, params: &SpectralCompressorParams) {
        // A capture that grew, got cleared, or arrived with a preset changes the base shape of
        // both curves, and the knee parabolas are built from the thresholds so they follow
        if self.capture.take_dirty() {
            self.should_update_downwards_thresholds
                .store(true, Ordering::SeqCst);
            self.should_update_upwards_thresholds
                .store(true, Ordering::SeqCst);
            self.should_update_downwards_knee_parabolas
                .store(true, Ordering::SeqCst);
            self.should_update_upwards_knee_parabolas
                .store(true, Ordering::SeqCst);
        }

        // NOTE: The threshold curves are polynomials in log-log (decibels-octaves) space. They're
        //       built inside of the branches below rather than up here because `Curve::new()`
        //       takes a logarithm, and in the common case neither array needs recomputing at all.
        if self
            .should_update_downwards_thresholds
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            for (chain_idx, chain) in params.threshold.chains.iter().enumerate() {
                let has_capture = self.prepare_capture_curve(params, chain_idx);
                let curve_params = params.threshold.curve_params(&chain.downwards);
                let blend = if has_capture {
                    Some(Self::capture_blend(
                        &self.capture_smoothed,
                        params,
                        &curve_params,
                    ))
                } else {
                    None
                };

                Self::recompute_thresholds(
                    &self.freqs,
                    &self.ln_freqs,
                    &mut self.eq_power,
                    &mut self.downwards_thresholds_db[chain_idx],
                    &curve_params,
                    &params.threshold.eq.snapshot(),
                    blend.as_ref(),
                    CompressorDirection::Downwards,
                    chain_idx,
                    chain.downwards.threshold_offset_db.value(),
                );
            }
        }

        if self
            .should_update_upwards_thresholds
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            for (chain_idx, chain) in params.threshold.chains.iter().enumerate() {
                let has_capture = self.prepare_capture_curve(params, chain_idx);
                let curve_params = params.threshold.curve_params(&chain.upwards);
                let blend = if has_capture {
                    Some(Self::capture_blend(
                        &self.capture_smoothed,
                        params,
                        &curve_params,
                    ))
                } else {
                    None
                };

                Self::recompute_thresholds(
                    &self.freqs,
                    &self.ln_freqs,
                    &mut self.eq_power,
                    &mut self.upwards_thresholds_db[chain_idx],
                    &curve_params,
                    &params.threshold.eq.snapshot(),
                    blend.as_ref(),
                    CompressorDirection::Upwards,
                    chain_idx,
                    chain.upwards.threshold_offset_db.value(),
                );
            }
        }

        if self
            .should_update_process_weights
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            for (chain_idx, chain) in params.threshold.chains.iter().enumerate() {
                Self::recompute_process_weights(
                    &self.ln_freqs,
                    &mut self.process_weights[chain_idx],
                    chain.bypass_below_hz.value(),
                );
            }
        }

        if self
            .should_update_downwards_ratios
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            // If the high-frequency rolloff is enabled then higher frequency bins will have their
            // ratios reduced to reduce harshness. This follows the octave scale. It's easier to do
            // this cleanly using reciprocals.
            let target_ratio_recip = params.compressors.downwards.ratio.value().recip();
            let downwards_high_freq_ratio_rolloff =
                params.compressors.downwards.high_freq_ratio_rolloff.value();
            for (ln_freq, ratio) in self.ln_freqs.iter().zip(self.downwards_ratios.iter_mut()) {
                let octave_fraction = ln_freq / HIGH_FREQ_RATIO_ROLLOFF_FREQUENCY_LN;
                let rolloff_t = octave_fraction * downwards_high_freq_ratio_rolloff;

                // If the octave fraction times the rolloff amount is high, then this should get
                // closer to `high_freq_ratio_rolloff` (which is in [0, 1]).
                let ratio_recip = (target_ratio_recip * (1.0 - rolloff_t)) + rolloff_t;
                *ratio = ratio_recip.recip();
            }
        }

        if self
            .should_update_upwards_ratios
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            let target_ratio_recip = params.compressors.upwards.ratio.value().recip();
            let upwards_high_freq_ratio_rolloff =
                params.compressors.upwards.high_freq_ratio_rolloff.value();
            for (ln_freq, ratio) in self.ln_freqs.iter().zip(self.upwards_ratios.iter_mut()) {
                let octave_fraction = ln_freq / HIGH_FREQ_RATIO_ROLLOFF_FREQUENCY_LN;
                let rolloff_t = octave_fraction * upwards_high_freq_ratio_rolloff;

                let ratio_recip = (target_ratio_recip * (1.0 - rolloff_t)) + rolloff_t;
                *ratio = ratio_recip.recip();
            }
        }

        if self
            .should_update_downwards_knee_parabolas
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            let downwards_knee_width_db = params.compressors.downwards.knee_width_db.value();
            // The ratios are shared between the chains, but the thresholds they combine with are
            // not, so the coefficients have to be worked out for each chain separately
            for chain_idx in 0..NUM_CHAINS {
                for ((ratio, threshold_db), (knee_parabola_scale, knee_parambola_intercept)) in self
                    .downwards_ratios
                    .iter()
                    .zip(self.downwards_thresholds_db[chain_idx].iter())
                    .zip(
                        self.downwards_knee_parabola_scale[chain_idx]
                            .iter_mut()
                            .zip(self.downwards_knee_parabola_intercept[chain_idx].iter_mut()),
                    )
                {
                    (*knee_parabola_scale, *knee_parambola_intercept) =
                        downwards_soft_knee_coefficients(
                            *threshold_db,
                            downwards_knee_width_db,
                            *ratio,
                        );
                }
            }
        }

        if self
            .should_update_upwards_knee_parabolas
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            let upwards_knee_width_db = params.compressors.upwards.knee_width_db.value();
            // The ratios are shared between the chains, but the thresholds they combine with are
            // not, so the coefficients have to be worked out for each chain separately
            for chain_idx in 0..NUM_CHAINS {
                for ((ratio, threshold_db), (knee_parabola_scale, knee_parambola_intercept)) in self
                    .upwards_ratios
                    .iter()
                    .zip(self.upwards_thresholds_db[chain_idx].iter())
                    .zip(
                        self.upwards_knee_parabola_scale[chain_idx]
                            .iter_mut()
                            .zip(self.upwards_knee_parabola_intercept[chain_idx].iter_mut()),
                    )
                {
                    (*knee_parabola_scale, *knee_parambola_intercept) =
                        upwards_soft_knee_coefficients(
                            *threshold_db,
                            upwards_knee_width_db,
                            *ratio,
                        );
                }
            }
        }
    }
}

/// The level one channel's detector sees, with the other channels folded in.
///
/// `this_channel_t` and `other_channels_t` are weights that sum to one across all channels, so at
/// full linking every channel ends up detecting the average of them all.
#[inline]
fn linked_magnitude(
    magnitudes: &[Vec<f32>],
    channel_idx: usize,
    bin_idx: usize,
    this_channel_t: f32,
    other_channels_t: f32,
) -> f32 {
    magnitudes
        .iter()
        .enumerate()
        .map(|(other_channel_idx, magnitudes)| {
            let t = if other_channel_idx == channel_idx {
                this_channel_t
            } else {
                other_channels_t
            };

            // SAFETY: Every channel's magnitudes are resized together with the envelopes this is
            //         indexed alongside.
            unsafe { magnitudes.get_unchecked(bin_idx) * t }
        })
        .sum()
}

/// Which chain a channel belongs to.
///
/// The stereo mode decides what the two chains mean; by the time the audio gets here it has
/// already been converted, so this is just a mapping. A mono layout collapses onto the first.
fn chain_for_channel(channel_idx: usize) -> usize {
    channel_idx.min(NUM_CHAINS - 1)
}

/// Apply downwards compression to the input with the supplied parameters. All values are in
/// decibels.
fn compress_downwards(
    input_db: f32,
    threshold_db: f32,
    ratio: f32,
    knee_width_db: f32,
    knee_parabola_scale: f32,
    knee_parabola_intercept: f32,
) -> f32 {
    // The soft-knee option will fade in the compression curve when reaching the knee start until it
    // matches the hard-knee curve at the knee-end
    let knee_start_db = threshold_db - (knee_width_db / 2.0);
    let knee_end_db = threshold_db + (knee_width_db / 2.0);
    if input_db <= knee_start_db {
        input_db
    } else if input_db <= knee_end_db {
        // See the `knee_parabola_intercept` field documentation for the full formula. The entire
        // osft knee part can be skipped if `knee_width_db == 0.0`.
        let parabola_x = input_db + knee_parabola_intercept;
        input_db + (knee_parabola_scale * parabola_x * parabola_x)
    } else {
        threshold_db + ((input_db - threshold_db) / ratio)
    }
}

/// Apply upwards compression to the input with the supplied parameters. All values are in
/// decibels.
fn compress_upwards(
    input_db: f32,
    threshold_db: f32,
    ratio: f32,
    knee_width_db: f32,
    knee_parabola_scale: f32,
    knee_parabola_intercept: f32,
) -> f32 {
    // We'll keep the terminology consistent, start is below the threshold, and end is above the
    // threshold
    let knee_start_db = threshold_db - (knee_width_db / 2.0);
    let knee_end_db = threshold_db + (knee_width_db / 2.0);

    // This goes the other way around compared to the downwards compression
    if input_db >= knee_end_db {
        input_db
    } else if input_db >= knee_start_db {
        let parabola_x = input_db + knee_parabola_intercept;
        input_db + (knee_parabola_scale * parabola_x * parabola_x)
    } else {
        threshold_db + ((input_db - threshold_db) / ratio)
    }
}

/// Compute the `(scale, intercept)`/`(a, b)` coefficients for the parabolic formula `x + a * (x +
/// b)^2`. The formula is taken from the Digital Dynamic Range Compressor Design paper by Dimitrios
/// Giannoulis et. al. This version applies to downwards compression. It can be precalculated for
/// the regular modes, since it's dependent on the threshold it has to be recomputed for every
/// sample with the sidechain matching mode.
fn downwards_soft_knee_coefficients(
    threshold_db: f32,
    knee_width_db: f32,
    ratio: f32,
) -> (f32, f32) {
    let scale = if knee_width_db != 0.0 {
        (2.0 * knee_width_db * ratio).recip() - (2.0 * knee_width_db).recip()
    } else {
        1.0
    };
    let intercept = -threshold_db + (knee_width_db / 2.0);

    (scale, intercept)
}

/// [`downwards_soft_knee_coefficients()`], but for upwards compression.
fn upwards_soft_knee_coefficients(threshold_db: f32, knee_width_db: f32, ratio: f32) -> (f32, f32) {
    // For the upwards version the scale becomes negated
    let scale = if knee_width_db != 0.0 {
        -((2.0 * knee_width_db * ratio).recip() - (2.0 * knee_width_db).recip())
    } else {
        1.0
    };
    // And the `+ (knee/2)` becomes `- (knee/2)` in the intercept
    let intercept = -threshold_db - (knee_width_db / 2.0);

    (scale, intercept)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bypass weights over a spread of frequencies, for a given corner.
    fn bypass_weights(bypass_below_hz: f32, frequencies: &[f32]) -> Vec<f32> {
        let ln_freqs: Vec<f32> = frequencies.iter().map(|hz| hz.ln()).collect();
        let mut weights = vec![0.0; ln_freqs.len()];
        CompressorBank::recompute_process_weights(&ln_freqs, &mut weights, bypass_below_hz);

        weights
    }

    /// The parameter's leftmost position means off, so nothing anywhere may be held back.
    #[test]
    fn the_low_frequency_bypass_is_off_at_its_minimum() {
        for weight in bypass_weights(LF_BYPASS_OFF_HZ, &[20.0, 25.0, 40.0, 1000.0, 20_000.0]) {
            assert_eq!(weight, 1.0);
        }
    }

    #[test]
    fn the_low_frequency_bypass_stops_everything_below_its_corner() {
        let weights = bypass_weights(100.0, &[20.0, 50.0, 100.0]);
        for (weight, hz) in weights.iter().zip([20.0, 50.0, 100.0]) {
            assert_eq!(*weight, 0.0, "at {hz} Hz");
        }
    }

    #[test]
    fn the_low_frequency_bypass_leaves_everything_above_the_fade_alone() {
        // The fade is half an octave wide, so anything past about 142 Hz is fully compressed again
        let weights = bypass_weights(100.0, &[145.0, 1000.0, 20_000.0]);
        for (weight, hz) in weights.iter().zip([145.0, 1000.0, 20_000.0]) {
            assert_eq!(*weight, 1.0, "at {hz} Hz");
        }
    }

    /// A step here would be a step in the frequency response, and a step in frequency is a long
    /// tail in time. The fade is what keeps that from happening, so it gets a test.
    #[test]
    fn the_low_frequency_bypass_fades_rather_than_switching() {
        let frequencies = [100.0, 105.0, 110.0, 118.0, 126.0, 134.0, 142.0];
        let weights = bypass_weights(100.0, &frequencies);

        for pair in weights.windows(2) {
            assert!(pair[1] >= pair[0], "the fade has to climb, got {pair:?}");
        }
        // And it really is partway through the middle rather than jumping at one end
        let middle = weights[3];
        assert!(
            middle > 0.15 && middle < 0.85,
            "expected a gradual fade, got {middle}"
        );
    }

    /// The weights `update_envelopes` derives from the channel link amount.
    fn weights(link: f32, num_channels: f32) -> (f32, f32) {
        let other_channels_t = link / num_channels;
        (
            1.0 - (other_channels_t * (num_channels - 1.0)),
            other_channels_t,
        )
    }

    #[test]
    fn unlinked_channels_detect_only_themselves() {
        let magnitudes = vec![vec![1.0], vec![0.0]];
        let (this_t, other_t) = weights(0.0, 2.0);

        assert_eq!(
            linked_magnitude(&magnitudes, 0, 0, this_t, other_t),
            1.0,
            "the loud channel should see its own level"
        );
        assert_eq!(
            linked_magnitude(&magnitudes, 1, 0, this_t, other_t),
            0.0,
            "the silent channel should not see the loud one"
        );
    }

    #[test]
    fn fully_linked_channels_detect_the_same_level() {
        let magnitudes = vec![vec![1.0], vec![0.0]];
        let (this_t, other_t) = weights(1.0, 2.0);

        let left = linked_magnitude(&magnitudes, 0, 0, this_t, other_t);
        let right = linked_magnitude(&magnitudes, 1, 0, this_t, other_t);
        assert_eq!(
            left, right,
            "fully linked, both channels must detect the same level"
        );
        assert_eq!(left, 0.5, "which is the average of the two");
    }

    #[test]
    fn partial_linking_lands_between_the_two() {
        let magnitudes = vec![vec![1.0], vec![0.0]];
        let (this_t, other_t) = weights(0.5, 2.0);

        let right = linked_magnitude(&magnitudes, 1, 0, this_t, other_t);
        assert!(
            right > 0.0 && right < 0.5,
            "half linked should lift the silent channel part way, got {right}"
        );
    }
}
