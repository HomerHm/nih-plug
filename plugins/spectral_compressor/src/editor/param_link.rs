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

//! A container view that mirrors parameter gestures between pairs of parameters.

use nih_plug::prelude::*;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::RawParamEvent;
use std::collections::HashSet;
use std::sync::Arc;

/// A container that keeps pairs of parameters in lockstep while a boolean parameter is enabled.
///
/// Widgets report edits by emitting [`RawParamEvent`]s that bubble up towards the root, so a view
/// wrapping those widgets can observe every gesture and emit a matching one for the paired
/// parameter. Doing the linking here rather than in the DSP means both parameters really do hold
/// the same value: each slider shows the same number, the host records automation for both, and
/// nothing has to special-case the link when the values are read.
///
/// Only the parameters that are actually paired are affected; everything else passes through.
pub struct ParamLink<L> {
    /// Read on every gesture, so toggling the link takes effect immediately.
    is_linked: L,
    /// Gestures on either side of a pair are mirrored onto the other side.
    pairs: Vec<(ParamPtr, ParamPtr)>,
    /// Parameters we have just emitted a mirrored gesture for. That gesture bubbles back through
    /// this view, and mirroring it a second time would ping-pong between the two forever. An entry
    /// is cleared once its gesture ends.
    suppressed: HashSet<ParamPtr>,
}

impl<L> ParamLink<L>
where
    L: 'static + Fn() -> bool,
{
    /// Wrap `content` so that gestures are mirrored between each `(a, b)` pair whenever
    /// `is_linked` returns true.
    pub fn new(
        cx: &mut Context,
        is_linked: L,
        pairs: Vec<(ParamPtr, ParamPtr)>,
        content: impl FnOnce(&mut Context),
    ) -> Handle<'_, Self> {
        Self {
            is_linked,
            pairs,
            suppressed: HashSet::new(),
        }
        .build(cx, |cx| content(cx))
    }

    /// The parameter `ptr` is paired with, if any. Pairs are symmetric.
    fn counterpart(&self, ptr: ParamPtr) -> Option<ParamPtr> {
        self.pairs.iter().find_map(|(a, b)| {
            if *a == ptr {
                Some(*b)
            } else if *b == ptr {
                Some(*a)
            } else {
                None
            }
        })
    }
}

impl<L> View for ParamLink<L>
where
    L: 'static + Fn() -> bool,
{
    fn element(&self) -> Option<&'static str> {
        Some("param-link")
    }

    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|param_event: &RawParamEvent, _| {
            let ptr = match param_event {
                RawParamEvent::BeginSetParameter(ptr)
                | RawParamEvent::EndSetParameter(ptr)
                | RawParamEvent::SetParameterNormalized(ptr, _) => *ptr,
                // Not a gesture, so there is nothing to mirror
                RawParamEvent::ParametersChanged => return,
            };

            // This is the echo of a gesture we emitted ourselves. Let it through untouched, and
            // stop suppressing the parameter once its gesture has ended.
            if self.suppressed.contains(&ptr) {
                if matches!(param_event, RawParamEvent::EndSetParameter(_)) {
                    self.suppressed.remove(&ptr);
                }
                return;
            }

            if !(self.is_linked)() {
                return;
            }
            let Some(counterpart) = self.counterpart(ptr) else {
                return;
            };

            // NOTE: The original event is deliberately not consumed, so the parameter the user
            //       actually grabbed still gets set by the root handler as usual.
            cx.emit(match param_event {
                RawParamEvent::BeginSetParameter(_) => {
                    RawParamEvent::BeginSetParameter(counterpart)
                }
                RawParamEvent::EndSetParameter(_) => RawParamEvent::EndSetParameter(counterpart),
                RawParamEvent::SetParameterNormalized(_, value) => {
                    RawParamEvent::SetParameterNormalized(counterpart, *value)
                }
                RawParamEvent::ParametersChanged => unreachable!("Filtered out above"),
            });
            self.suppressed.insert(counterpart);
        });
    }
}

/// Look up a parameter's [`ParamPtr`] by its unqualified ID within `params`.
///
/// The IDs here are the ones declared on the struct itself; any `id_prefix` from a `#[nested]`
/// attribute higher up is not part of them.
///
/// # Panics
///
/// Panics if no parameter with that ID exists, which would mean the ID was renamed without
/// updating the caller.
pub fn param_ptr_by_id<P: Params>(params: &Arc<P>, id: &str) -> ParamPtr {
    params
        .param_map()
        .into_iter()
        .find(|(param_id, _, _)| param_id == id)
        .map(|(_, param_ptr, _)| param_ptr)
        .unwrap_or_else(|| panic!("No parameter with ID '{id}', this is a bug"))
}
