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

//! Runs the plugin outside of a host.
//!
//! Useful for telling a problem in the plugin apart from a problem in the way a particular host
//! embeds its window, and for seeing the log output that a host would otherwise swallow.

use nih_plug::prelude::*;
use spectral_compressor::SpectralCompressor;

fn main() {
    nih_export_standalone::<SpectralCompressor>();
}
