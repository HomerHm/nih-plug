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

//! EQ-shaped nodes that deform the threshold curve.
//!
//! These are **not** filters. Nothing here touches the audio; each node only contributes a
//! frequency dependent offset that gets added to a compressor's threshold curve. They use the
//! magnitude responses of the corresponding analog filter prototypes purely so that the shapes
//! look and behave the way anyone who has used an EQ expects them to.
//!
//! Everything is evaluated in the power domain (`|H|^2`) and multiplied together, so a whole bank
//! costs one logarithm per bin instead of one per node per bin.

use nih_plug::prelude::*;
use std::sync::Arc;

/// The number of nodes available.
///
/// A single bank is shared by both compressors, with each node choosing which of them it applies
/// to, so this is the total rather than a per-compressor count.
///
/// Raising this later is safe. **Lowering it is not**: presets that used the higher-numbered nodes
/// would silently lose them.
pub const MAX_EQ_NODES: usize = 8;

/// Which of the two compressors something applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressorDirection {
    Upwards,
    Downwards,
}

impl CompressorDirection {
    pub fn name(self) -> &'static str {
        match self {
            CompressorDirection::Upwards => "Upwards",
            CompressorDirection::Downwards => "Downwards",
        }
    }
}

/// Which compressor's threshold curve a node deforms.
///
/// This replaces what used to be a single switch linking two separate banks. That switch had to
/// overwrite one bank with the other to take effect, which threw away whichever curve happened to
/// be on the losing side. Choosing per node means nothing is ever discarded.
#[derive(Enum, Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum EqNodeTarget {
    #[default]
    #[id = "both"]
    #[name = "Both"]
    Both,
    #[id = "downwards"]
    #[name = "Downwards"]
    Downwards,
    #[id = "upwards"]
    #[name = "Upwards"]
    Upwards,
}

impl EqNodeTarget {
    /// Whether a node with this target contributes to `direction`'s threshold curve.
    pub fn applies_to(self, direction: CompressorDirection) -> bool {
        match self {
            EqNodeTarget::Both => true,
            EqNodeTarget::Downwards => direction == CompressorDirection::Downwards,
            EqNodeTarget::Upwards => direction == CompressorDirection::Upwards,
        }
    }
}

/// Power ratios below this are clamped before being converted to decibels. A notch reaches exactly
/// zero, which would otherwise produce `-inf`.
///
/// This sits far below anything audible on purpose. Clamping at a more reasonable looking -100 dB
/// flattens the skirt of a 48 dB/octave cut barely more than an octave past its corner, which is
/// still well inside the range the curve gets drawn over. The threshold the curve feeds into is
/// clamped separately, where a floor actually belongs.
const MIN_POWER: f32 = 1e-30;

/// Default center frequencies, spread over the spectrum so enabling several nodes doesn't stack
/// them all on the same frequency.
const DEFAULT_FREQUENCIES: [f32; MAX_EQ_NODES] =
    [60.0, 120.0, 250.0, 500.0, 1000.0, 2500.0, 6000.0, 12_000.0];

/// The shape of a single threshold curve node.
///
/// The cut slopes are part of the type rather than a separate parameter, which keeps each node
/// down to four parameters.
#[derive(Enum, Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum EqNodeType {
    /// The node contributes nothing and is skipped entirely.
    #[default]
    #[id = "off"]
    #[name = "Off"]
    Off,
    #[id = "bell"]
    #[name = "Bell"]
    Bell,
    /// A full null at the center frequency. Ignores the gain parameter.
    #[id = "notch"]
    #[name = "Notch"]
    Notch,
    #[id = "low_shelf"]
    #[name = "Low Shelf"]
    LowShelf,
    #[id = "high_shelf"]
    #[name = "High Shelf"]
    HighShelf,
    #[id = "low_cut_12"]
    #[name = "Low Cut 12"]
    LowCut12,
    #[id = "low_cut_24"]
    #[name = "Low Cut 24"]
    LowCut24,
    #[id = "low_cut_48"]
    #[name = "Low Cut 48"]
    LowCut48,
    #[id = "high_cut_12"]
    #[name = "High Cut 12"]
    HighCut12,
    #[id = "high_cut_24"]
    #[name = "High Cut 24"]
    HighCut24,
    #[id = "high_cut_48"]
    #[name = "High Cut 48"]
    HighCut48,
}

impl EqNodeType {
    /// The Butterworth order for the cut types, i.e. the slope in dB/octave divided by six.
    fn cut_order(self) -> Option<i32> {
        match self {
            EqNodeType::LowCut12 | EqNodeType::HighCut12 => Some(2),
            EqNodeType::LowCut24 | EqNodeType::HighCut24 => Some(4),
            EqNodeType::LowCut48 | EqNodeType::HighCut48 => Some(8),
            _ => None,
        }
    }
}

/// A plain-data snapshot of one node.
///
/// The DSP and the editor both build their curves from these, so the drawn curve and the audible
/// one cannot drift apart.
#[derive(Debug, Clone, Copy)]
pub struct EqNode {
    pub node_type: EqNodeType,
    pub target: EqNodeTarget,
    pub center_frequency: f32,
    pub gain_db: f32,
    pub q: f32,
}

impl Default for EqNode {
    fn default() -> Self {
        EqNode {
            node_type: EqNodeType::Off,
            target: EqNodeTarget::Both,
            center_frequency: 1000.0,
            gain_db: 0.0,
            q: 1.0,
        }
    }
}

/// A snapshot of a whole bank of nodes.
#[derive(Debug, Clone, Copy)]
pub struct EqCurveParams {
    pub nodes: [EqNode; MAX_EQ_NODES],
}

impl Default for EqCurveParams {
    fn default() -> Self {
        EqCurveParams {
            nodes: [EqNode::default(); MAX_EQ_NODES],
        }
    }
}

/// One node with everything that doesn't depend on the bin frequency hoisted out.
#[derive(Debug, Clone, Copy)]
struct PreparedNode {
    /// Multiplying a frequency by this gives `omega`, avoiding a division per bin.
    inv_center_frequency: f32,
    shape: PreparedShape,
}

/// The per-shape constants. These all come from the analog prototypes in the RBJ audio EQ
/// cookbook, evaluated at `s = j*omega` where `omega` is the frequency relative to the node's
/// center frequency.
#[derive(Debug, Clone, Copy)]
enum PreparedShape {
    /// `H(s) = (s^2 + (A/Q)s + 1) / (s^2 + (1/(A*Q))s + 1)`
    Bell {
        /// `(A/Q)^2`
        numerator_coeff: f32,
        /// `(1/(A*Q))^2`
        denominator_coeff: f32,
    },
    /// `H(s) = (s^2 + 1) / (s^2 + (1/Q)s + 1)`
    Notch {
        /// `(1/Q)^2`
        denominator_coeff: f32,
    },
    /// `H(s) = A * (s^2 + (sqrt(A)/Q)s + A) / (A*s^2 + (sqrt(A)/Q)s + 1)`
    LowShelf {
        a: f32,
        a_squared: f32,
        a_over_q_squared: f32,
    },
    /// `H(s) = A * (A*s^2 + (sqrt(A)/Q)s + 1) / (s^2 + (sqrt(A)/Q)s + A)`
    HighShelf {
        a: f32,
        a_squared: f32,
        a_over_q_squared: f32,
    },
    /// Butterworth high-pass: `|H|^2 = omega^2n / (1 + omega^2n)`
    LowCut { order: i32 },
    /// Butterworth low-pass: `|H|^2 = 1 / (1 + omega^2n)`
    HighCut { order: i32 },
}

impl PreparedNode {
    /// Precompute a node's constants, or `None` if it's switched off.
    fn new(node: &EqNode) -> Option<Self> {
        if node.node_type == EqNodeType::Off {
            return None;
        }

        // A shelf or bell boosting by `gain_db` decibels needs `|H| = A^2`, hence the 40 rather
        // than the usual 20
        let a = 10.0f32.powf(node.gain_db / 40.0);
        let q = node.q.max(0.01);

        let shape = match node.node_type {
            EqNodeType::Off => unreachable!("Handled above"),
            EqNodeType::Bell => {
                let numerator = a / q;
                let denominator = 1.0 / (a * q);
                PreparedShape::Bell {
                    numerator_coeff: numerator * numerator,
                    denominator_coeff: denominator * denominator,
                }
            }
            EqNodeType::Notch => PreparedShape::Notch {
                denominator_coeff: (1.0 / q) * (1.0 / q),
            },
            EqNodeType::LowShelf => PreparedShape::LowShelf {
                a,
                a_squared: a * a,
                a_over_q_squared: a / (q * q),
            },
            EqNodeType::HighShelf => PreparedShape::HighShelf {
                a,
                a_squared: a * a,
                a_over_q_squared: a / (q * q),
            },
            EqNodeType::LowCut12 | EqNodeType::LowCut24 | EqNodeType::LowCut48 => {
                PreparedShape::LowCut {
                    order: node
                        .node_type
                        .cut_order()
                        .expect("Cut type without an order"),
                }
            }
            EqNodeType::HighCut12 | EqNodeType::HighCut24 | EqNodeType::HighCut48 => {
                PreparedShape::HighCut {
                    order: node
                        .node_type
                        .cut_order()
                        .expect("Cut type without an order"),
                }
            }
        };

        Some(PreparedNode {
            inv_center_frequency: node.center_frequency.max(1.0).recip(),
            shape,
        })
    }

    /// This node's squared magnitude response at `frequency`, in Hertz.
    #[inline]
    fn power_at(&self, frequency: f32) -> f32 {
        let omega = frequency * self.inv_center_frequency;
        let omega_squared = omega * omega;

        match self.shape {
            PreparedShape::Bell {
                numerator_coeff,
                denominator_coeff,
            } => {
                let real = 1.0 - omega_squared;
                let real_squared = real * real;
                (real_squared + (numerator_coeff * omega_squared))
                    / (real_squared + (denominator_coeff * omega_squared))
            }
            PreparedShape::Notch { denominator_coeff } => {
                let real = 1.0 - omega_squared;
                let real_squared = real * real;
                real_squared / (real_squared + (denominator_coeff * omega_squared))
            }
            PreparedShape::LowShelf {
                a,
                a_squared,
                a_over_q_squared,
            } => {
                let numerator_real = a - omega_squared;
                let denominator_real = 1.0 - (a * omega_squared);
                let shared = a_over_q_squared * omega_squared;
                a_squared * ((numerator_real * numerator_real) + shared)
                    / ((denominator_real * denominator_real) + shared)
            }
            PreparedShape::HighShelf {
                a,
                a_squared,
                a_over_q_squared,
            } => {
                let numerator_real = 1.0 - (a * omega_squared);
                let denominator_real = a - omega_squared;
                let shared = a_over_q_squared * omega_squared;
                a_squared * ((numerator_real * numerator_real) + shared)
                    / ((denominator_real * denominator_real) + shared)
            }
            PreparedShape::LowCut { order } => {
                let omega_pow = omega_squared.powi(order);
                omega_pow / (1.0 + omega_pow)
            }
            PreparedShape::HighCut { order } => (1.0 + omega_squared.powi(order)).recip(),
        }
    }
}

/// A bank of nodes with their constants precomputed, ready to be evaluated over a set of
/// frequencies.
///
/// Switched-off nodes are dropped while building this, so a bank with two active nodes costs two
/// nodes' worth of work rather than [`MAX_EQ_NODES`].
pub struct EqCurve {
    nodes: [Option<PreparedNode>; MAX_EQ_NODES],
    len: usize,
}

impl EqCurve {
    /// Precompute the constants for every active node in `params` that applies to `direction`.
    /// Allocation free, so this is safe to call from the audio thread.
    pub fn new(params: &EqCurveParams, direction: CompressorDirection) -> Self {
        let mut nodes = [None; MAX_EQ_NODES];
        let mut len = 0;
        for node in &params.nodes {
            if !node.target.applies_to(direction) {
                continue;
            }
            if let Some(prepared) = PreparedNode::new(node) {
                nodes[len] = Some(prepared);
                len += 1;
            }
        }

        EqCurve { nodes, len }
    }

    /// Whether every node is switched off, in which case evaluating this curve is a no-op.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Multiply each element of `power` by this curve's squared magnitude response at the matching
    /// frequency in `frequencies`.
    ///
    /// The node loop is on the outside so the per-shape branch happens once per node rather than
    /// once per bin. `power` should be filled with `1.0` before the first call.
    pub fn accumulate_power(&self, frequencies: &[f32], power: &mut [f32]) {
        nih_debug_assert_eq!(frequencies.len(), power.len());

        for node in self.nodes[..self.len].iter().flatten() {
            for (frequency, power) in frequencies.iter().zip(power.iter_mut()) {
                *power *= node.power_at(*frequency);
            }
        }
    }

    /// This curve's contribution at a single frequency, in decibels. Used by the editor, which
    /// draws far fewer points than there are bins and so doesn't benefit from the batched version.
    pub fn evaluate_db(&self, frequency: f32) -> f32 {
        let mut power = 1.0;
        for node in self.nodes[..self.len].iter().flatten() {
            power *= node.power_at(frequency);
        }

        power_to_db(power)
    }
}

/// Convert an accumulated power ratio to decibels, clamping the notch's zero to something finite.
#[inline]
pub fn power_to_db(power: f32) -> f32 {
    10.0 * power.max(MIN_POWER).log10()
}

/// The parameters for a single node.
#[derive(Params)]
pub struct EqNodeParams {
    #[id = "eqtype"]
    pub node_type: EnumParam<EqNodeType>,
    /// Which compressor's curve this node deforms.
    #[id = "eqtarget"]
    pub target: EnumParam<EqNodeTarget>,
    #[id = "eqfreq"]
    pub center_frequency: FloatParam,
    #[id = "eqgain"]
    pub gain_db: FloatParam,
    #[id = "eqq"]
    pub q: FloatParam,
}

impl EqNodeParams {
    /// Create the parameters for node `index` (zero based). Changing any of them marks the owning
    /// compressor's thresholds as needing a recompute.
    pub fn new(
        name_prefix: &str,
        index: usize,
        set_update_thresholds: Arc<dyn Fn(f32) + Send + Sync>,
    ) -> Self {
        let node_number = index + 1;

        EqNodeParams {
            node_type: EnumParam::new(
                format!("{name_prefix}Node {node_number} Type"),
                EqNodeType::Off,
            )
            .with_callback({
                let set_update_thresholds = set_update_thresholds.clone();
                Arc::new(move |_| set_update_thresholds(0.0))
            })
            .hide_in_generic_ui(),
            target: EnumParam::new(
                format!("{name_prefix}Node {node_number} Target"),
                EqNodeTarget::Both,
            )
            .with_callback({
                let set_update_thresholds = set_update_thresholds.clone();
                Arc::new(move |_| set_update_thresholds(0.0))
            })
            .hide_in_generic_ui(),
            center_frequency: FloatParam::new(
                format!("{name_prefix}Node {node_number} Freq"),
                DEFAULT_FREQUENCIES[index],
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
            gain_db: FloatParam::new(
                format!("{name_prefix}Node {node_number} Gain"),
                0.0,
                FloatRange::Linear {
                    min: -30.0,
                    max: 30.0,
                },
            )
            .with_callback(set_update_thresholds.clone())
            .with_unit(" dB")
            .with_step_size(0.1)
            .hide_in_generic_ui(),
            q: FloatParam::new(
                format!("{name_prefix}Node {node_number} Q"),
                1.0,
                FloatRange::Skewed {
                    min: 0.1,
                    max: 18.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_callback(set_update_thresholds)
            .with_step_size(0.01)
            .hide_in_generic_ui(),
        }
    }

    /// Read this node's current values into a plain-data snapshot.
    pub fn snapshot(&self) -> EqNode {
        EqNode {
            node_type: self.node_type.value(),
            target: self.target.value(),
            center_frequency: self.center_frequency.value(),
            gain_db: self.gain_db.value(),
            q: self.q.value(),
        }
    }
}

/// A compressor's whole bank of threshold curve nodes.
#[derive(Params)]
pub struct EqBankParams {
    /// The `array` attribute suffixes each node's parameter IDs with its position, so
    /// `eqfreq` becomes `eqfreq_1` through `eqfreq_6`. Combined with the `id_prefix` on the
    /// compressor a level up, the final IDs look like `upwards_eqfreq_1`.
    #[nested(array, group = "Node")]
    pub nodes: [EqNodeParams; MAX_EQ_NODES],
}

impl EqBankParams {
    pub fn new(name_prefix: &str, set_update_thresholds: Arc<dyn Fn(f32) + Send + Sync>) -> Self {
        EqBankParams {
            nodes: std::array::from_fn(|index| {
                EqNodeParams::new(name_prefix, index, set_update_thresholds.clone())
            }),
        }
    }

    /// Read every node's current values into a plain-data snapshot.
    pub fn snapshot(&self) -> EqCurveParams {
        EqCurveParams {
            nodes: std::array::from_fn(|index| self.nodes[index].snapshot()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decibels at `frequency` for a single node, as seen by the downwards compressor.
    fn node_db(node: EqNode, frequency: f32) -> f32 {
        let mut params = EqCurveParams::default();
        params.nodes[0] = node;
        EqCurve::new(&params, CompressorDirection::Downwards).evaluate_db(frequency)
    }

    /// A node of `node_type` at 1 kHz with the given gain.
    fn bell(gain_db: f32) -> EqNode {
        EqNode {
            node_type: EqNodeType::Bell,
            target: EqNodeTarget::Both,
            center_frequency: 1000.0,
            gain_db,
            q: 1.0,
        }
    }

    #[test]
    fn off_nodes_contribute_nothing() {
        let curve = EqCurve::new(&EqCurveParams::default(), CompressorDirection::Downwards);
        assert!(curve.is_empty());
        assert_eq!(curve.evaluate_db(1000.0), 0.0);
    }

    #[test]
    fn bell_hits_its_gain_at_the_center_frequency() {
        for gain_db in [-18.0, -6.0, 6.0, 18.0] {
            let db = node_db(
                EqNode {
                    node_type: EqNodeType::Bell,
                    target: EqNodeTarget::Both,
                    center_frequency: 1000.0,
                    gain_db,
                    q: 1.0,
                },
                1000.0,
            );
            assert!(
                (db - gain_db).abs() < 0.01,
                "bell with {gain_db} dB gain evaluated to {db} dB"
            );
        }
    }

    #[test]
    fn bell_decays_to_unity_away_from_its_center() {
        let node = EqNode {
            node_type: EqNodeType::Bell,
            target: EqNodeTarget::Both,
            center_frequency: 1000.0,
            gain_db: 12.0,
            q: 4.0,
        };
        assert!(node_db(node, 20.0).abs() < 0.1);
        assert!(node_db(node, 20_000.0).abs() < 0.1);
    }

    #[test]
    fn shelves_reach_their_gain_on_the_correct_side() {
        let low = EqNode {
            node_type: EqNodeType::LowShelf,
            target: EqNodeTarget::Both,
            center_frequency: 1000.0,
            gain_db: 12.0,
            q: 0.7,
        };
        assert!((node_db(low, 20.0) - 12.0).abs() < 0.5);
        assert!(node_db(low, 20_000.0).abs() < 0.5);

        let high = EqNode {
            node_type: EqNodeType::HighShelf,
            ..low
        };
        assert!(node_db(high, 20.0).abs() < 0.5);
        assert!((node_db(high, 20_000.0) - 12.0).abs() < 0.5);
    }

    #[test]
    fn cuts_are_minus_three_db_at_their_corner() {
        for node_type in [
            EqNodeType::LowCut12,
            EqNodeType::LowCut24,
            EqNodeType::LowCut48,
            EqNodeType::HighCut12,
            EqNodeType::HighCut24,
            EqNodeType::HighCut48,
        ] {
            let db = node_db(
                EqNode {
                    node_type,
                    target: EqNodeTarget::Both,
                    center_frequency: 1000.0,
                    gain_db: 0.0,
                    q: 1.0,
                },
                1000.0,
            );
            assert!(
                (db - -3.0103).abs() < 0.01,
                "{node_type:?} evaluated to {db} dB at its corner frequency"
            );
        }
    }

    #[test]
    fn cut_slopes_match_their_names() {
        // An octave below the corner a high-pass should be down by its slope
        for (node_type, expected_db_per_octave) in [
            (EqNodeType::LowCut12, 12.0),
            (EqNodeType::LowCut24, 24.0),
            (EqNodeType::LowCut48, 48.0),
        ] {
            let node = EqNode {
                node_type,
                target: EqNodeTarget::Both,
                center_frequency: 1000.0,
                gain_db: 0.0,
                q: 1.0,
            };
            // Measured well into the stopband where the response is a straight line
            let octave_apart = node_db(node, 62.5) - node_db(node, 125.0);
            assert!(
                (octave_apart + expected_db_per_octave).abs() < 0.5,
                "{node_type:?} fell {}, expected {expected_db_per_octave} dB per octave",
                -octave_apart
            );
        }
    }

    #[test]
    fn notch_nulls_at_its_center_and_recovers_beside_it() {
        let node = EqNode {
            node_type: EqNodeType::Notch,
            target: EqNodeTarget::Both,
            center_frequency: 1000.0,
            gain_db: 0.0,
            q: 4.0,
        };
        assert!(node_db(node, 1000.0) <= -60.0);
        assert!(node_db(node, 100.0).abs() < 0.5);
        assert!(node_db(node, 10_000.0).abs() < 0.5);
    }

    #[test]
    fn nodes_stack_additively_in_decibels() {
        let mut params = EqCurveParams::default();
        params.nodes[0] = EqNode {
            node_type: EqNodeType::Bell,
            target: EqNodeTarget::Both,
            center_frequency: 1000.0,
            gain_db: 6.0,
            q: 1.0,
        };
        params.nodes[1] = EqNode {
            node_type: EqNodeType::Bell,
            target: EqNodeTarget::Both,
            center_frequency: 1000.0,
            gain_db: 4.0,
            q: 1.0,
        };

        let db = EqCurve::new(&params, CompressorDirection::Downwards).evaluate_db(1000.0);
        assert!((db - 10.0).abs() < 0.01, "two stacked bells gave {db} dB");
    }

    #[test]
    fn a_node_only_reaches_the_compressors_it_targets() {
        for (target, expected_downwards, expected_upwards) in [
            (EqNodeTarget::Both, 6.0, 6.0),
            (EqNodeTarget::Downwards, 6.0, 0.0),
            (EqNodeTarget::Upwards, 0.0, 6.0),
        ] {
            let mut params = EqCurveParams::default();
            params.nodes[0] = EqNode {
                target,
                ..bell(6.0)
            };

            let downwards =
                EqCurve::new(&params, CompressorDirection::Downwards).evaluate_db(1000.0);
            let upwards = EqCurve::new(&params, CompressorDirection::Upwards).evaluate_db(1000.0);
            assert!(
                (downwards - expected_downwards).abs() < 0.01,
                "{target:?} gave the downwards curve {downwards} dB"
            );
            assert!(
                (upwards - expected_upwards).abs() < 0.01,
                "{target:?} gave the upwards curve {upwards} dB"
            );
        }
    }

    #[test]
    fn differently_targeted_nodes_do_not_bleed_into_each_other() {
        // What the old single link switch could not express: each curve shaped independently,
        // with nothing overwritten to achieve it
        let mut params = EqCurveParams::default();
        params.nodes[0] = EqNode {
            target: EqNodeTarget::Downwards,
            ..bell(6.0)
        };
        params.nodes[1] = EqNode {
            target: EqNodeTarget::Upwards,
            ..bell(-9.0)
        };

        let downwards = EqCurve::new(&params, CompressorDirection::Downwards).evaluate_db(1000.0);
        let upwards = EqCurve::new(&params, CompressorDirection::Upwards).evaluate_db(1000.0);
        assert!(
            (downwards - 6.0).abs() < 0.01,
            "downwards gave {downwards} dB"
        );
        assert!((upwards - -9.0).abs() < 0.01, "upwards gave {upwards} dB");
    }

    #[test]
    fn batched_and_single_evaluation_agree() {
        let mut params = EqCurveParams::default();
        params.nodes[0] = EqNode {
            node_type: EqNodeType::Bell,
            target: EqNodeTarget::Both,
            center_frequency: 800.0,
            gain_db: -9.0,
            q: 2.5,
        };
        params.nodes[1] = EqNode {
            node_type: EqNodeType::HighShelf,
            target: EqNodeTarget::Both,
            center_frequency: 5000.0,
            gain_db: 6.0,
            q: 0.7,
        };
        params.nodes[2] = EqNode {
            node_type: EqNodeType::LowCut24,
            target: EqNodeTarget::Both,
            center_frequency: 60.0,
            gain_db: 0.0,
            q: 1.0,
        };
        let curve = EqCurve::new(&params, CompressorDirection::Downwards);

        let frequencies = [30.0, 120.0, 800.0, 3000.0, 12_000.0];
        let mut power = [1.0; 5];
        curve.accumulate_power(&frequencies, &mut power);

        for (frequency, power) in frequencies.iter().zip(power.iter()) {
            let batched = power_to_db(*power);
            let single = curve.evaluate_db(*frequency);
            assert!(
                (batched - single).abs() < 0.001,
                "at {frequency} Hz: batched {batched} dB vs single {single} dB"
            );
        }
    }
}
