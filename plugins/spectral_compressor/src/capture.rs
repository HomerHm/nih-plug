//! Storage and math for the captured sound signature.
//!
//! The user points this at a reference track, holds down capture for a while, and the averaged
//! frequency response becomes the base shape of the threshold curve. See the notes below for why
//! the averaging and the storage grid are the way they are; both were picked deliberately and both
//! give the wrong answer if changed to the obvious alternative.

use nih_plug::params::persist::PersistentField;
use nih_plug::prelude::*;
use realfft::num_complex::Complex32;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use crate::compressor_bank::NUM_CHAINS;

/// How many stereo modes there are. The capture is stored separately per mode because the two
/// chains hold completely different signals in each one: left/right versus mid/side. A right
/// channel capture applied to a side signal would be nonsense, since side is usually tens of
/// decibels down and has almost no low end.
pub const NUM_STEREO_MODES: usize = 2;

/// The number of points in the storage grid. Together with the frequency range below this works
/// out to just under 1/24th of an octave per point.
pub const CAPTURE_GRID_LEN: usize = 240;

/// The lowest frequency stored in the grid.
pub const CAPTURE_MIN_HZ: f32 = 20.0;
/// The highest frequency stored in the grid.
pub const CAPTURE_MAX_HZ: f32 = 20_000.0;

/// Frames whose broadband level sits more than this far below the loudest frame seen so far are
/// left out of the average.
///
/// This is not optional. The average is taken in the decibel domain (see [`CaptureSlot::push`]),
/// and that makes it sensitive to silence in a way a power average would not be: a stretch of
/// digital black is a stack of very negative decibel values that drags the whole curve toward flat.
/// A power average would weigh those frames at practically zero all by itself.
const GATE_RANGE_DB: f32 = 40.0;

/// Bin magnitudes are clamped to this before being converted to decibels, so a single empty bin
/// can't contribute negative infinity to the average.
const CAPTURE_FLOOR_DB: f32 = -140.0;

/// The natural logarithm of [`CAPTURE_MIN_HZ`].
const LN_MIN_HZ: f32 = 2.995_732_3; // 20.0f32.ln()
/// The natural logarithm of [`CAPTURE_MAX_HZ`].
const LN_MAX_HZ: f32 = 9.903_487_5; // 20_000.0f32.ln()

/// The spacing between two grid points, in nepers of frequency.
const GRID_LN_STEP: f32 = (LN_MAX_HZ - LN_MIN_HZ) / (CAPTURE_GRID_LEN - 1) as f32;

/// The spacing between two grid points, in octaves. Used to convert a smoothing width in octaves
/// into a number of grid points.
const GRID_OCTAVE_STEP: f32 = GRID_LN_STEP / std::f32::consts::LN_2;

/// The natural logarithm of the frequency stored at `idx`.
#[inline]
pub fn grid_ln_freq(idx: usize) -> f32 {
    LN_MIN_HZ + (GRID_LN_STEP * idx as f32)
}

/// Where `ln_freq` falls on the grid, as a continuous index. Values outside of
/// `0..CAPTURE_GRID_LEN - 1` are outside of the stored frequency range.
#[inline]
pub fn grid_position(ln_freq: f32) -> f32 {
    (ln_freq - LN_MIN_HZ) / GRID_LN_STEP
}

/// One captured curve. There is one of these per stereo mode per chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureSlot {
    /// The mean level per grid point, in decibels, stored exactly as it was measured. It is
    /// deliberately *not* normalized here: the curve is pinned to zero at the threshold center
    /// frequency when it gets used, and that center frequency is a live parameter, so normalizing
    /// at capture time would bake in a value that goes stale the moment the user drags it.
    pub curve_db: Vec<f32>,
    /// How many frames went into `curve_db`. Stored rather than derived so that loading a preset
    /// and then capturing some more keeps weighing the old and new frames correctly.
    pub frame_count: u32,
    /// The loudest broadband frame level seen so far, which is what the gate compares against.
    pub peak_level_db: f32,
}

impl Default for CaptureSlot {
    fn default() -> Self {
        Self {
            curve_db: vec![0.0; CAPTURE_GRID_LEN],
            frame_count: 0,
            peak_level_db: f32::NEG_INFINITY,
        }
    }
}

impl CaptureSlot {
    /// Whether this slot has nothing in it. The GUI grays out the amount control when this is true
    /// rather than silently doing nothing.
    pub fn is_empty(&self) -> bool {
        self.frame_count == 0
    }

    /// Throw away everything that was captured.
    pub fn clear(&mut self) {
        self.curve_db.fill(0.0);
        self.frame_count = 0;
        self.peak_level_db = f32::NEG_INFINITY;
    }

    /// Fold one frame's grid values into the running average, and report whether it was actually
    /// used. Returns `false` for frames the gate rejected.
    ///
    /// The average is taken over decibels, which weighs every frame the same no matter how loud it
    /// is. Averaging powers instead would weigh a frame that is twenty decibels louder a hundred
    /// times as heavily, so capturing a quiet intro on top of a loud drop would move the result by
    /// about a percent — the intro may as well not have been captured at all. Since the point of
    /// accumulating across several captures is to blend sections that differ in level, the decibel
    /// domain is the one that does what the feature says on the tin.
    pub fn push(&mut self, frame_db: &[f32]) -> bool {
        debug_assert_eq!(frame_db.len(), CAPTURE_GRID_LEN);
        if self.curve_db.len() != CAPTURE_GRID_LEN {
            return false;
        }

        // The mean over the grid doubles as this frame's broadband level. The grid is log spaced,
        // so every octave gets the same say in it, which is a better level estimate for this
        // purpose than a linear-frequency average would be.
        let frame_level_db = frame_db.iter().sum::<f32>() / CAPTURE_GRID_LEN as f32;
        if frame_level_db > self.peak_level_db {
            self.peak_level_db = frame_level_db;
        } else if frame_level_db < self.peak_level_db - GATE_RANGE_DB {
            return false;
        }

        // Updating the mean in place rather than keeping a running sum: the sum of a long capture
        // would lose precision, and this way the stored value is always directly usable.
        let n = self.frame_count as f32 + 1.0;
        for (mean, sample) in self.curve_db.iter_mut().zip(frame_db) {
            *mean += (*sample - *mean) / n;
        }
        self.frame_count = self.frame_count.saturating_add(1);

        true
    }

    /// Overwrite this slot with `other` without allocating, so the audio thread can pull in a
    /// preset that was just loaded.
    pub fn copy_from(&mut self, other: &CaptureSlot) {
        if self.curve_db.len() == other.curve_db.len() {
            self.curve_db.copy_from_slice(&other.curve_db);
            self.frame_count = other.frame_count;
            self.peak_level_db = other.peak_level_db;
        }
    }
}

/// Every captured curve, and the thing that gets persisted with the plugin state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureState {
    /// Bumped whenever the meaning of the fields below changes, so a future version can migrate an
    /// old preset instead of misreading it.
    pub version: u32,
    /// Indexed by [`slot_index()`]. Flat rather than nested because a flat `Vec` is what survives
    /// a round trip through serde most predictably.
    pub slots: Vec<CaptureSlot>,
}

/// The current format version written by [`CaptureState`].
pub const CAPTURE_STATE_VERSION: u32 = 1;

impl Default for CaptureState {
    fn default() -> Self {
        Self {
            version: CAPTURE_STATE_VERSION,
            slots: vec![CaptureSlot::default(); NUM_STEREO_MODES * NUM_CHAINS],
        }
    }
}

/// Where the curve for `stereo_mode_idx` and `chain_idx` lives in [`CaptureState::slots`].
#[inline]
pub fn slot_index(stereo_mode_idx: usize, chain_idx: usize) -> usize {
    (stereo_mode_idx * NUM_CHAINS) + chain_idx
}

impl CaptureState {
    /// Force the state back into a shape the rest of the code can rely on. Deserialization can
    /// hand back anything at all if the preset was written by a different version or edited by
    /// hand, and everything downstream indexes into these vectors from the audio thread.
    pub fn sanitize(&mut self) {
        self.slots
            .resize(NUM_STEREO_MODES * NUM_CHAINS, CaptureSlot::default());
        for slot in &mut self.slots {
            slot.curve_db.resize(CAPTURE_GRID_LEN, 0.0);
            if !slot.curve_db.iter().all(|value| value.is_finite()) {
                slot.clear();
            }
            if slot.frame_count == 0 {
                slot.clear();
            }
        }
        self.version = CAPTURE_STATE_VERSION;
    }

    pub fn slot(&self, stereo_mode_idx: usize, chain_idx: usize) -> &CaptureSlot {
        &self.slots[slot_index(stereo_mode_idx, chain_idx)]
    }

    pub fn slot_mut(&mut self, stereo_mode_idx: usize, chain_idx: usize) -> &mut CaptureSlot {
        &mut self.slots[slot_index(stereo_mode_idx, chain_idx)]
    }

    /// Overwrite every slot from `other` without allocating.
    pub fn copy_from(&mut self, other: &CaptureState) {
        if self.slots.len() != other.slots.len() {
            return;
        }
        for (dst, src) in self.slots.iter_mut().zip(&other.slots) {
            dst.copy_from(src);
        }
    }

    /// Clear both chains for one stereo mode.
    pub fn clear_mode(&mut self, stereo_mode_idx: usize) {
        for chain_idx in 0..NUM_CHAINS {
            self.slot_mut(stereo_mode_idx, chain_idx).clear();
        }
    }
}

/// The shared, persisted handle to the capture state.
pub type SharedCaptureState = Arc<PersistedCapture>;

/// Where the capture reads its audio from.
#[derive(Enum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureSource {
    /// The plugin's own input. Always does something, which is why it is the default.
    #[id = "main"]
    #[name = "Main"]
    Main,
    /// The sidechain input. This is the mode that makes matching another track practical: route
    /// the reference into the sidechain and capture without moving the plugin off the track you
    /// are working on.
    #[id = "sidechain"]
    #[name = "Sidechain"]
    Sidechain,
}

/// The capture state as the host sees it, plus a counter that lets the audio thread notice when
/// the host has replaced it.
///
/// The plain `Mutex` that nih-plug already knows how to persist would work for storage, but it
/// gives the audio thread no way to tell a preset load from its own last write. Wrapping it means
/// [`PersistentField::set`] can bump a generation counter that the audio thread polls for free.
#[derive(Debug, Default)]
pub struct PersistedCapture {
    state: Mutex<CaptureState>,
    /// Incremented every time the host writes new state in.
    generation: AtomicU32,
}

impl PersistedCapture {
    /// Read the state, for the GUI. Never call this from the audio thread; it can block.
    pub fn read<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&CaptureState) -> R,
    {
        match self.state.lock() {
            Ok(state) => f(&state),
            Err(poisoned) => f(&poisoned.into_inner()),
        }
    }

    pub fn generation(&self) -> u32 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Copy the audio thread's curves in, without blocking. Returns `false` when the lock was busy
    /// or when the host replaced the state in the meantime, in which case the caller should try
    /// again later rather than overwrite what the host wrote.
    ///
    /// Checking the generation while holding the lock is what makes that safe: [`Self::set()`]
    /// takes the same lock, so a write cannot be halfway through, and one that already finished is
    /// visible here.
    pub fn try_publish(&self, local: &CaptureState, expected_generation: u32) -> bool {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(_) => return false,
        };
        if self.generation.load(Ordering::SeqCst) != expected_generation {
            return false;
        }

        state.copy_from(local);
        true
    }

    /// Copy the host's curves out into the audio thread's buffers, without blocking or allocating.
    pub fn try_pull(&self, local: &mut CaptureState) -> bool {
        match self.state.try_lock() {
            Ok(state) => {
                local.copy_from(&state);
                true
            }
            Err(_) => false,
        }
    }
}

impl<'a> PersistentField<'a, CaptureState> for std::sync::Arc<PersistedCapture> {
    fn set(&self, new_value: CaptureState) {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        *state = new_value;
        // A preset can carry anything, and the audio thread indexes straight into these vectors
        state.sanitize();
        drop(state);

        self.generation.fetch_add(1, Ordering::SeqCst);
    }

    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&CaptureState) -> R,
    {
        self.read(f)
    }
}

/// How many `poll()` calls pass between refreshes of the shared state and of the threshold curves
/// while a capture is running.
///
/// Rebuilding the thresholds on every hop would be wasteful for something the user watches at
/// human speed, and the shared state only feeds the GUI, which redraws at its own rate anyway.
const CAPTURE_REFRESH_INTERVAL: u32 = 16;

/// The audio thread's side of the capture.
///
/// The curves live here as plain data rather than behind the shared lock, so folding a frame in
/// never has to take one. The copy in [`PersistedCapture`] is refreshed from here with a
/// non-blocking `try_lock`; if it happens to be busy the attempt is simply skipped, since this
/// side keeps the authoritative copy and the next attempt will get through.
pub struct CaptureBank {
    /// The audio thread's own copy of every curve.
    local: CaptureState,
    binner: CaptureBinner,
    /// Scratch holding one frame's grid values.
    frame_db: Vec<f32>,
    shared: SharedCaptureState,
    /// The shared state's generation as of the last pull, so a preset load gets noticed.
    last_generation: u32,
    /// Set by the editor while a capture is running.
    active: Arc<AtomicBool>,
    /// Set by the editor to throw away the current stereo mode's curves.
    clear_requested: Arc<AtomicBool>,
    /// Counts `poll()` calls towards the next refresh.
    polls_since_refresh: u32,
    /// Whether the curves changed since the thresholds were last rebuilt from them.
    dirty: bool,
    /// Whether a capture was running last time, so that stopping forces a final refresh.
    was_active: bool,
}

impl CaptureBank {
    /// Create the capture, along with the shared handles the parameters and the editor need. It
    /// owns them for the same reason the compressor bank owns its update flags: the parameter
    /// object is built from the compressor bank, so this side has to exist first.
    pub fn new() -> Self {
        let shared: SharedCaptureState = Arc::new(PersistedCapture::default());

        Self {
            last_generation: shared.generation(),
            local: CaptureState::default(),
            binner: CaptureBinner::default(),
            frame_db: vec![0.0; CAPTURE_GRID_LEN],
            shared,
            active: Arc::new(AtomicBool::new(false)),
            clear_requested: Arc::new(AtomicBool::new(false)),
            polls_since_refresh: 0,
            dirty: true,
            was_active: false,
        }
    }

    /// The persisted state, for the parameters to save and for the editor to draw.
    pub fn shared(&self) -> SharedCaptureState {
        self.shared.clone()
    }

    /// The flag the editor raises while the user holds capture.
    pub fn active_flag(&self) -> Arc<AtomicBool> {
        self.active.clone()
    }

    /// The flag the editor raises to throw the current stereo mode's curves away.
    pub fn clear_flag(&self) -> Arc<AtomicBool> {
        self.clear_requested.clone()
    }

    /// Whether a capture is running right now.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// The captured curve for this slot, or `None` when nothing has been captured into it. The
    /// values are raw measured decibels; pin them with [`CaptureCurve`] before use.
    pub fn curve(&self, stereo_mode_idx: usize, chain_idx: usize) -> Option<&[f32]> {
        let slot = self.local.slot(stereo_mode_idx, chain_idx);
        if slot.is_empty() {
            None
        } else {
            Some(&slot.curve_db)
        }
    }

    /// Whether the curves changed since this was last asked, which means the threshold arrays need
    /// rebuilding.
    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    /// Housekeeping that has to happen whether or not a capture is running: acting on a clear
    /// request, picking up state the host loaded, and refreshing the shared copy.
    ///
    /// Cheap enough to call for every channel of every hop; in the common case it is two relaxed
    /// atomic loads and a counter.
    pub fn poll(&mut self, stereo_mode_idx: usize) {
        // A preset the host just loaded wins over anything this side has, so it is checked first
        let generation = self.shared.generation();
        if generation != self.last_generation {
            if self.shared.try_pull(&mut self.local) {
                self.last_generation = generation;
                self.polls_since_refresh = 0;
                self.dirty = true;
            }

            // Either way, nothing gets published over state that has not been taken in yet
            return;
        }

        if self.clear_requested.swap(false, Ordering::SeqCst) {
            self.local.clear_mode(stereo_mode_idx);
            self.dirty = true;
            self.force_refresh();
        }

        let active = self.is_active();
        // Stopping is the moment the user is waiting on, so it never waits for the interval
        if self.was_active && !active {
            self.dirty = true;
            self.force_refresh();
        }
        self.was_active = active;

        if !active {
            return;
        }

        self.polls_since_refresh += 1;
        if self.polls_since_refresh >= CAPTURE_REFRESH_INTERVAL {
            self.dirty = true;
            self.force_refresh();
        }
    }

    /// Fold one channel's bin magnitudes into its curve. Only call this while a capture is
    /// running; see [`Self::is_active()`].
    pub fn push_frame(
        &mut self,
        magnitudes: &[f32],
        ln_freqs: &[f32],
        stereo_mode_idx: usize,
        chain_idx: usize,
    ) {
        if !self.binner.fold(magnitudes, ln_freqs, &mut self.frame_db) {
            return;
        }

        self.local
            .slot_mut(stereo_mode_idx, chain_idx)
            .push(&self.frame_db);
    }

    /// The same as [`Self::push_frame()`], but reading the FFT bins directly. This is what the
    /// main input uses, since its magnitudes have to be taken before compression scales the bins.
    pub fn push_bins(
        &mut self,
        bins: &[Complex32],
        ln_freqs: &[f32],
        stereo_mode_idx: usize,
        chain_idx: usize,
    ) {
        if !self.binner.fold_bins(bins, ln_freqs, &mut self.frame_db) {
            return;
        }

        self.local
            .slot_mut(stereo_mode_idx, chain_idx)
            .push(&self.frame_db);
    }

    /// Hand the local copy over to the GUI and the host, and reset the interval counter whether or
    /// not the handover got through.
    fn force_refresh(&mut self) {
        self.polls_since_refresh = 0;
        self.shared.try_publish(&self.local, self.last_generation);
    }
}

/// Folds a frame's FFT bin magnitudes onto the storage grid.
///
/// This walks the bins rather than the grid points, because the two do not line up in either
/// direction: at twenty hertz a large window still only has a handful of bins to offer, while near
/// twenty kilohertz hundreds of bins land in a single grid cell. Grid cells that no bin reached are
/// filled in afterwards by interpolating between the ones that were, which is exactly right here
/// since the grid index *is* log frequency.
pub struct CaptureBinner {
    sums: Vec<f32>,
    counts: Vec<u32>,
}

impl Default for CaptureBinner {
    fn default() -> Self {
        Self {
            sums: vec![0.0; CAPTURE_GRID_LEN],
            counts: vec![0; CAPTURE_GRID_LEN],
        }
    }
}

impl CaptureBinner {
    /// Fold `magnitudes` onto the grid, writing decibel values into `frame_db`. `ln_freqs` holds
    /// the natural logarithm of each bin's center frequency. Returns `false` when no bin at all
    /// landed inside the grid's frequency range, in which case `frame_db` is left untouched.
    pub fn fold(&mut self, magnitudes: &[f32], ln_freqs: &[f32], frame_db: &mut [f32]) -> bool {
        let num_bins = magnitudes.len().min(ln_freqs.len());
        self.accumulate(ln_freqs, num_bins, |bin_idx| magnitudes[bin_idx]);
        self.fill_gaps(frame_db)
    }

    /// The same as [`Self::fold()`], but taking the FFT bins directly. Used for the main input,
    /// where the magnitudes have to be read before compression has had a chance to scale the bins.
    pub fn fold_bins(
        &mut self,
        bins: &[Complex32],
        ln_freqs: &[f32],
        frame_db: &mut [f32],
    ) -> bool {
        let num_bins = bins.len().min(ln_freqs.len());
        self.accumulate(ln_freqs, num_bins, |bin_idx| bins[bin_idx].norm());
        self.fill_gaps(frame_db)
    }

    fn accumulate(
        &mut self,
        ln_freqs: &[f32],
        num_bins: usize,
        magnitude_at: impl Fn(usize) -> f32,
    ) {
        self.sums.fill(0.0);
        self.counts.fill(0);

        for bin_idx in 0..num_bins {
            let position = grid_position(ln_freqs[bin_idx]);
            // Written so that a NaN frequency falls out here rather than indexing with one
            if !(position > -0.5) || position > (CAPTURE_GRID_LEN - 1) as f32 + 0.5 {
                continue;
            }

            let cell = (position.round() as usize).min(CAPTURE_GRID_LEN - 1);
            self.sums[cell] +=
                util::gain_to_db_fast_epsilon(magnitude_at(bin_idx)).max(CAPTURE_FLOOR_DB);
            self.counts[cell] += 1;
        }
    }

    /// Turn the per-cell sums into means, interpolating across cells no bin reached. Cells past
    /// the outermost populated ones hold that outermost value, because there is no measurement out
    /// there to extrapolate from and inventing a slope would be worse than being flat.
    fn fill_gaps(&self, frame_db: &mut [f32]) -> bool {
        let first = match self.counts.iter().position(|count| *count > 0) {
            Some(idx) => idx,
            // Nothing landed on the grid at all, which would take an absurd sample rate. Report it
            // rather than filling the frame with garbage.
            None => return false,
        };
        let last = self
            .counts
            .iter()
            .rposition(|count| *count > 0)
            .expect("there is a populated cell because `first` was found");

        let mean_at = |idx: usize| self.sums[idx] / self.counts[idx] as f32;

        frame_db[..=first].fill(mean_at(first));
        frame_db[last..].fill(mean_at(last));

        // Walk between the populated cells, filling each run of empty ones with a straight line in
        // (grid index, decibel) space
        let mut previous = first;
        for current in (first + 1)..=last {
            if self.counts[current] == 0 {
                continue;
            }

            let start_db = mean_at(previous);
            let end_db = mean_at(current);
            let span = (current - previous) as f32;
            for (offset, cell) in ((previous + 1)..current).enumerate() {
                let t = (offset + 1) as f32 / span;
                frame_db[cell] = start_db + ((end_db - start_db) * t);
            }
            frame_db[current] = end_db;

            previous = current;
        }

        true
    }
}

/// Smooth `src` into `dst` with a box average `octaves` wide, which is the usual meaning of
/// "1/3 octave smoothing" and friends. An `octaves` of zero or less copies straight across.
///
/// The edges hold their outermost value rather than shrinking the window, so smoothing can't pull
/// the ends of the curve toward zero.
pub fn smooth_into(src: &[f32], dst: &mut [f32], octaves: f32) {
    debug_assert_eq!(src.len(), dst.len());

    let radius = if octaves > 0.0 {
        (octaves / GRID_OCTAVE_STEP / 2.0).round() as usize
    } else {
        0
    };
    if radius == 0 {
        dst.copy_from_slice(src);
        return;
    }

    let last = src.len() - 1;
    for (idx, value) in dst.iter_mut().enumerate() {
        let mut sum = 0.0;
        for offset in -(radius as isize)..=(radius as isize) {
            let tap = (idx as isize + offset).clamp(0, last as isize) as usize;
            sum += src[tap];
        }
        *value = sum / ((radius * 2) + 1) as f32;
    }
}

/// Samples a captured curve, pinned so that it reads zero decibels at the threshold center
/// frequency.
///
/// Pinning is what lets the capture stand in for the polynomial's shape without moving the overall
/// threshold level: the polynomial is zero at its own center frequency by construction, so a
/// capture that is also zero there can be crossfaded against it without the global threshold or
/// the offset changing meaning. Storing the curve unpinned and doing this at use time is what
/// keeps it correct while the user drags the center frequency around.
pub struct CaptureCurve<'a> {
    grid_db: &'a [f32],
    /// Subtracted from every lookup so the curve reads zero at the center frequency.
    pin_offset_db: f32,
}

impl<'a> CaptureCurve<'a> {
    pub fn new(grid_db: &'a [f32], ln_center_frequency: f32) -> Self {
        let pin_offset_db = sample_grid(grid_db, ln_center_frequency);
        Self {
            grid_db,
            pin_offset_db,
        }
    }

    /// The captured shape at this frequency, in decibels relative to the center frequency.
    #[inline]
    pub fn evaluate_ln(&self, ln_freq: f32) -> f32 {
        sample_grid(self.grid_db, ln_freq) - self.pin_offset_db
    }
}

/// Linearly interpolate the grid at `ln_freq`, holding the outermost values beyond either end.
#[inline]
fn sample_grid(grid_db: &[f32], ln_freq: f32) -> f32 {
    if grid_db.is_empty() {
        return 0.0;
    }

    let last = grid_db.len() - 1;
    let position = grid_position(ln_freq);
    if position <= 0.0 {
        return grid_db[0];
    }
    if position >= last as f32 {
        return grid_db[last];
    }

    let lower = position as usize;
    let t = position - lower as f32;
    grid_db[lower] + ((grid_db[lower + 1] - grid_db[lower]) * t)
}

/// How the captured curve is stored and how much of it is used.
#[derive(Params)]
pub struct CaptureParams {
    /// Every captured curve, saved with the plugin state. This is the field that makes the capture
    /// survive closing the project, and the one that will carry it into presets.
    #[persist = "capture"]
    pub state: SharedCaptureState,

    /// How much of the captured shape stands in for the polynomial's. At zero the threshold curve
    /// is bit for bit what it was before this feature existed, which is what makes the whole thing
    /// safe to leave in the signal path.
    #[id = "cap_amount"]
    pub amount: FloatParam,
    /// Which input the capture listens to.
    #[id = "cap_source"]
    pub source: EnumParam<CaptureSource>,
    /// The width of the box average applied to the curve, in octaves.
    ///
    /// Applied when the curve is used rather than when it is captured, so it can be changed
    /// without having to capture again.
    #[id = "cap_smooth"]
    pub smoothing_octaves: FloatParam,
    /// Below this frequency the capture fades out and the polynomial takes back over. Useful when
    /// the reference is in a different key and matching its sub bass would be wrong.
    #[id = "cap_low"]
    pub low_frequency: FloatParam,
    /// Above this frequency the capture fades out and the polynomial takes back over.
    #[id = "cap_high"]
    pub high_frequency: FloatParam,
}

impl CaptureParams {
    pub fn new(
        state: SharedCaptureState,
        set_update_thresholds: Arc<dyn Fn(f32) + Send + Sync>,
    ) -> Self {
        Self {
            state,

            // Defaulting to fully applied means that stopping a capture visibly does something.
            // With an empty slot this changes nothing either way, so it cannot surprise anyone who
            // never uses the feature.
            amount: FloatParam::new(
                "Capture Amount",
                1.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_callback(set_update_thresholds.clone())
            .with_unit("%")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            // The main input always captures something. Sidechain is the mode that makes matching
            // another track practical, but it silently captures nothing if none is routed, so it
            // is not what an unsuspecting first press should do.
            source: EnumParam::new("Capture Source", CaptureSource::Main),
            smoothing_octaves: FloatParam::new(
                "Capture Smoothing",
                0.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 2.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_callback(set_update_thresholds.clone())
            .with_unit(" oct")
            .with_step_size(0.01),
            low_frequency: FloatParam::new(
                "Capture Low",
                CAPTURE_MIN_HZ,
                FloatRange::Skewed {
                    min: CAPTURE_MIN_HZ,
                    max: CAPTURE_MAX_HZ,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_callback(set_update_thresholds.clone())
            .with_value_to_string(formatters::v2s_f32_hz_then_khz(0))
            .with_string_to_value(formatters::s2v_f32_hz_then_khz()),
            high_frequency: FloatParam::new(
                "Capture High",
                CAPTURE_MAX_HZ,
                FloatRange::Skewed {
                    min: CAPTURE_MIN_HZ,
                    max: CAPTURE_MAX_HZ,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_callback(set_update_thresholds)
            .with_value_to_string(formatters::v2s_f32_hz_then_khz(0))
            .with_string_to_value(formatters::s2v_f32_hz_then_khz()),
        }
    }
}

/// How wide the fade at either end of the capture band is, in octaves. Fixed rather than exposed:
/// a hard edge would leave a step in the threshold curve, and the exact width is not something
/// anyone would want to tune.
const CAPTURE_FADE_OCTAVES: f32 = 1.0;

/// Everything needed to blend one chain's captured curve into one threshold curve.
///
/// The capture stands in for the polynomial's *shape* rather than adding to it. Adding would
/// double up the tilt — matching a reference that is already close to pink noise would land at
/// about -6 dB/octave against a baseline that is already at -3 — so what happens here is a
/// crossfade between the two shapes, with the level left alone.
pub struct CaptureBlend<'a> {
    curve: CaptureCurve<'a>,
    /// The polynomial's intercept, subtracted to get at its shape on its own.
    polynomial_intercept_db: f32,
    amount: f32,

    /// The band the capture applies over, in nepers, along with where its fades reach zero.
    ln_low: f32,
    ln_low_zero: f32,
    ln_high: f32,
    ln_high_zero: f32,
    /// Whether the band covers everything, so the fades can be skipped.
    full_range: bool,
}

impl<'a> CaptureBlend<'a> {
    pub fn new(
        grid_db: &'a [f32],
        ln_center_frequency: f32,
        polynomial_intercept_db: f32,
        amount: f32,
        low_frequency: f32,
        high_frequency: f32,
    ) -> Self {
        let fade = CAPTURE_FADE_OCTAVES * std::f32::consts::LN_2;
        let ln_low = low_frequency.ln();
        let ln_high = high_frequency.ln();

        Self {
            curve: CaptureCurve::new(grid_db, ln_center_frequency),
            polynomial_intercept_db,
            amount,
            ln_low,
            ln_low_zero: ln_low - fade,
            ln_high,
            ln_high_zero: ln_high + fade,
            full_range: low_frequency <= CAPTURE_MIN_HZ && high_frequency >= CAPTURE_MAX_HZ,
        }
    }

    /// How much of the capture applies at this frequency, from zero to one. Outside the band this
    /// falls off over an octave, which is what "the cut part goes back to the pink noise curve"
    /// means in practice.
    #[inline]
    pub fn weight_ln(&self, ln_freq: f32) -> f32 {
        if self.full_range {
            return 1.0;
        }

        let below = smoothstep(self.ln_low_zero, self.ln_low, ln_freq);
        let above = 1.0 - smoothstep(self.ln_high, self.ln_high_zero, ln_freq);
        below.min(above)
    }

    /// How far the capture moves the threshold at this frequency, given what the polynomial says
    /// there.
    ///
    /// Returning a delta rather than a blended value is deliberate: adding a zero is exact in
    /// floating point, so a curve with no capture in play comes out bit for bit the same as it did
    /// before this feature existed.
    #[inline]
    pub fn delta_ln(&self, ln_freq: f32, polynomial_db: f32) -> f32 {
        let weight = self.amount * self.weight_ln(ln_freq);
        if weight <= 0.0 {
            return 0.0;
        }

        let polynomial_shape_db = polynomial_db - self.polynomial_intercept_db;
        (self.curve.evaluate_ln(ln_freq) - polynomial_shape_db) * weight
    }
}

/// Zero at or below `edge0`, one at or above `edge1`, and a smooth ramp in between.
///
/// A raised cosine would look the same but costs a transcendental per bin, and this runs across
/// every bin of every threshold array.
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }

    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - (2.0 * t))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The grid constants are written out as literals so they can be `const`, so this makes sure
    /// they still match what they claim to be.
    #[test]
    fn grid_constants_match_their_frequencies() {
        assert!((LN_MIN_HZ - CAPTURE_MIN_HZ.ln()).abs() < 1e-5);
        assert!((LN_MAX_HZ - CAPTURE_MAX_HZ.ln()).abs() < 1e-5);
    }

    #[test]
    fn grid_spans_the_intended_range() {
        assert!((grid_ln_freq(0).exp() - CAPTURE_MIN_HZ).abs() < 0.01);
        assert!((grid_ln_freq(CAPTURE_GRID_LEN - 1).exp() - CAPTURE_MAX_HZ).abs() < 1.0);
    }

    /// The whole point of the grid is that it is close to 1/24th of an octave everywhere.
    #[test]
    fn grid_resolution_is_about_a_twentyfourth_of_an_octave() {
        assert!((1.0 / GRID_OCTAVE_STEP - 24.0).abs() < 0.5);
    }

    #[test]
    fn grid_position_inverts_grid_ln_freq() {
        for idx in [0, 1, 57, 199, CAPTURE_GRID_LEN - 1] {
            assert!((grid_position(grid_ln_freq(idx)) - idx as f32).abs() < 1e-3);
        }
    }

    /// A decibel domain average puts the result halfway between two frames. This is the property
    /// that makes capturing a quiet intro on top of a loud drop worth doing at all, so it gets a
    /// test to stop anyone from quietly turning it into a power average.
    #[test]
    fn averaging_weighs_every_frame_the_same_regardless_of_level() {
        let mut slot = CaptureSlot::default();
        // Twenty decibels apart, which is a hundred to one in power
        assert!(slot.push(&[40.0; CAPTURE_GRID_LEN]));
        assert!(slot.push(&[60.0; CAPTURE_GRID_LEN]));

        assert_eq!(slot.frame_count, 2);
        for value in &slot.curve_db {
            assert!(
                (value - 50.0).abs() < 1e-3,
                "expected the midpoint, got {value}"
            );
        }
    }

    #[test]
    fn the_gate_rejects_silence_but_keeps_quiet_music() {
        let mut slot = CaptureSlot::default();
        assert!(slot.push(&[0.0; CAPTURE_GRID_LEN]));

        // Twenty decibels down is a quiet section, and has to survive
        assert!(slot.push(&[-20.0; CAPTURE_GRID_LEN]));
        // Eighty decibels down is silence between takes, and must not drag the curve toward flat
        assert!(!slot.push(&[-80.0; CAPTURE_GRID_LEN]));
        assert_eq!(slot.frame_count, 2);
    }

    #[test]
    fn clearing_resets_the_gate_as_well() {
        let mut slot = CaptureSlot::default();
        assert!(slot.push(&[0.0; CAPTURE_GRID_LEN]));
        slot.clear();

        assert!(slot.is_empty());
        // Without the peak being reset this quiet frame would still be gated out
        assert!(slot.push(&[-80.0; CAPTURE_GRID_LEN]));
    }

    #[test]
    fn folding_interpolates_bins_that_are_sparser_than_the_grid() {
        let mut binner = CaptureBinner::default();
        let mut frame_db = [0.0; CAPTURE_GRID_LEN];

        // Two bins an octave apart with a twelve decibel difference between them. Every grid point
        // in between has to be filled by interpolation, because no bin reached it.
        let ln_freqs = [1000.0f32.ln(), 2000.0f32.ln()];
        let magnitudes = [util::db_to_gain(0.0), util::db_to_gain(12.0)];
        assert!(binner.fold(&magnitudes, &ln_freqs, &mut frame_db));

        let at = |hz: f32| sample_grid(&frame_db, hz.ln());
        assert!((at(1000.0) - 0.0).abs() < 0.2, "got {}", at(1000.0));
        assert!((at(2000.0) - 12.0).abs() < 0.2, "got {}", at(2000.0));
        // Halfway in octaves is where the straight line puts the midpoint
        assert!((at(1414.0) - 6.0).abs() < 0.3, "got {}", at(1414.0));
    }

    #[test]
    fn folding_holds_the_edges_rather_than_extrapolating() {
        let mut binner = CaptureBinner::default();
        let mut frame_db = [0.0; CAPTURE_GRID_LEN];

        let ln_freqs = [1000.0f32.ln(), 2000.0f32.ln()];
        let magnitudes = [util::db_to_gain(0.0), util::db_to_gain(12.0)];
        assert!(binner.fold(&magnitudes, &ln_freqs, &mut frame_db));

        // Below the lowest bin and above the highest one the curve is flat, not a runaway slope
        assert!((frame_db[0] - 0.0).abs() < 0.2);
        assert!((frame_db[CAPTURE_GRID_LEN - 1] - 12.0).abs() < 0.2);
    }

    #[test]
    fn folding_reports_failure_when_no_bin_lands_on_the_grid() {
        let mut binner = CaptureBinner::default();
        let mut frame_db = [7.0; CAPTURE_GRID_LEN];

        let ln_freqs = [1.0f32.ln()];
        let magnitudes = [1.0];
        assert!(!binner.fold(&magnitudes, &ln_freqs, &mut frame_db));
        // The caller's buffer is left alone so it can't be mistaken for a real measurement
        assert!(frame_db.iter().all(|value| *value == 7.0));
    }

    #[test]
    fn pinning_makes_the_curve_read_zero_at_the_center_frequency() {
        // A curve that rises by one decibel per grid point, so nothing about it is symmetric
        let grid: Vec<f32> = (0..CAPTURE_GRID_LEN).map(|idx| idx as f32).collect();
        let ln_center = 1000.0f32.ln();
        let curve = CaptureCurve::new(&grid, ln_center);

        assert!(curve.evaluate_ln(ln_center).abs() < 1e-3);
        // Relative shape is preserved: one grid step up is one decibel up
        let one_step_up = grid_ln_freq(grid_position(ln_center).round() as usize + 1);
        let expected = 1.0 + (grid_position(ln_center).round() - grid_position(ln_center));
        assert!((curve.evaluate_ln(one_step_up) - expected).abs() < 1e-2);
    }

    #[test]
    fn smoothing_flattens_a_spike_without_moving_the_average() {
        let mut src = vec![0.0; CAPTURE_GRID_LEN];
        src[120] = 24.0;
        let mut dst = vec![0.0; CAPTURE_GRID_LEN];

        smooth_into(&src, &mut dst, 1.0);

        assert!(
            dst[120] < 6.0,
            "the spike should be flattened, got {}",
            dst[120]
        );
        assert!(dst[119] > 0.0, "it should have spread outwards");

        let src_mean = src.iter().sum::<f32>();
        let dst_mean = dst.iter().sum::<f32>();
        assert!((src_mean - dst_mean).abs() < 0.1);
    }

    #[test]
    fn zero_smoothing_is_a_straight_copy() {
        let src: Vec<f32> = (0..CAPTURE_GRID_LEN).map(|idx| (idx % 7) as f32).collect();
        let mut dst = vec![0.0; CAPTURE_GRID_LEN];

        smooth_into(&src, &mut dst, 0.0);

        assert_eq!(src, dst);
    }

    #[test]
    fn sanitizing_repairs_a_state_that_deserialized_into_the_wrong_shape() {
        let mut state = CaptureState {
            version: 0,
            slots: vec![CaptureSlot {
                curve_db: vec![1.0; 3],
                frame_count: 5,
                peak_level_db: 0.0,
            }],
        };
        state.sanitize();

        assert_eq!(state.version, CAPTURE_STATE_VERSION);
        assert_eq!(state.slots.len(), NUM_STEREO_MODES * NUM_CHAINS);
        for slot in &state.slots {
            assert_eq!(slot.curve_db.len(), CAPTURE_GRID_LEN);
        }
    }

    #[test]
    fn sanitizing_throws_out_a_curve_with_non_finite_values() {
        let mut state = CaptureState::default();
        state.slot_mut(0, 0).curve_db[10] = f32::NAN;
        state.slot_mut(0, 0).frame_count = 3;
        state.sanitize();

        assert!(state.slot(0, 0).is_empty());
    }

    /// The audio thread copies preset data in without allocating, so the copy has to actually
    /// carry everything the average depends on.
    #[test]
    fn copying_carries_the_frame_count_so_capture_can_continue() {
        let mut source = CaptureState::default();
        assert!(source.slot_mut(1, 0).push(&[12.0; CAPTURE_GRID_LEN]));

        let mut destination = CaptureState::default();
        destination.copy_from(&source);

        assert_eq!(destination.slot(1, 0).frame_count, 1);
        assert!((destination.slot(1, 0).curve_db[0] - 12.0).abs() < 1e-3);
        // The other slots are untouched, which is what makes switching stereo modes lossless
        assert!(destination.slot(0, 0).is_empty());
    }

    #[test]
    fn slots_are_addressed_without_overlapping() {
        let mut seen = vec![];
        for stereo_mode_idx in 0..NUM_STEREO_MODES {
            for chain_idx in 0..NUM_CHAINS {
                seen.push(slot_index(stereo_mode_idx, chain_idx));
            }
        }
        seen.sort_unstable();
        seen.dedup();

        assert_eq!(seen.len(), NUM_STEREO_MODES * NUM_CHAINS);
    }

    /// A capture curve that falls by `slope_db_per_neper` around one kilohertz, offset by an
    /// arbitrary level to prove the level gets normalized away.
    fn sloped_grid(slope_db_per_neper: f32, level_db: f32) -> Vec<f32> {
        let ln_center = 1000.0f32.ln();
        (0..CAPTURE_GRID_LEN)
            .map(|idx| (slope_db_per_neper * (grid_ln_freq(idx) - ln_center)) + level_db)
            .collect()
    }

    fn full_range_blend(grid: &[f32], intercept_db: f32, amount: f32) -> CaptureBlend<'_> {
        CaptureBlend::new(
            grid,
            1000.0f32.ln(),
            intercept_db,
            amount,
            CAPTURE_MIN_HZ,
            CAPTURE_MAX_HZ,
        )
    }

    /// The reason the capture crossfades with the polynomial's shape instead of being added to it.
    /// Adding would land a reference that already looks like the baseline at twice the tilt, so
    /// matching something pink against a pink baseline would come out at -6 dB/octave.
    #[test]
    fn matching_a_reference_shaped_like_the_baseline_changes_nothing() {
        let intercept_db = -12.0;
        let ln_center = 1000.0f32.ln();
        let slope = -3.0;
        // The capture sits forty decibels away from the polynomial, which pinning has to remove
        let grid = sloped_grid(slope, 40.0);
        let blend = full_range_blend(&grid, intercept_db, 1.0);

        for hz in [50.0f32, 200.0, 1000.0, 5000.0, 15000.0] {
            let ln_freq = hz.ln();
            let polynomial_db = intercept_db + (slope * (ln_freq - ln_center));
            let blended = polynomial_db + blend.delta_ln(ln_freq, polynomial_db);

            assert!(
                (blended - polynomial_db).abs() < 0.05,
                "at {hz} Hz the blend moved {polynomial_db} to {blended}"
            );
        }
    }

    /// Fully applied, the shape comes from the capture and the level still comes from the
    /// polynomial's intercept.
    #[test]
    fn a_full_blend_takes_the_shape_from_the_capture_and_the_level_from_the_polynomial() {
        let intercept_db = -12.0;
        let ln_center = 1000.0f32.ln();
        let grid = sloped_grid(-6.0, 40.0);
        let blend = full_range_blend(&grid, intercept_db, 1.0);

        // The polynomial is flat here, so anything but the capture's own slope would be wrong
        for hz in [200.0f32, 1000.0, 5000.0] {
            let ln_freq = hz.ln();
            let blended = intercept_db + blend.delta_ln(ln_freq, intercept_db);
            let expected = intercept_db + (-6.0 * (ln_freq - ln_center));

            assert!(
                (blended - expected).abs() < 0.05,
                "at {hz} Hz: {blended} instead of {expected}"
            );
        }
    }

    /// The property that makes this safe to leave in the signal path: at zero the threshold curve
    /// is untouched, and exactly so rather than nearly so.
    #[test]
    fn a_zero_amount_moves_the_curve_by_exactly_nothing() {
        let grid = sloped_grid(-6.0, 40.0);
        let blend = full_range_blend(&grid, -12.0, 0.0);

        for hz in [20.0f32, 100.0, 1000.0, 19_000.0] {
            let delta = blend.delta_ln(hz.ln(), -12.0);
            assert_eq!(delta, 0.0, "at {hz} Hz");
        }
    }

    #[test]
    fn half_an_amount_lands_halfway() {
        let intercept_db = -12.0;
        let grid = sloped_grid(-6.0, 40.0);

        let full = full_range_blend(&grid, intercept_db, 1.0).delta_ln(100.0f32.ln(), intercept_db);
        let half = full_range_blend(&grid, intercept_db, 0.5).delta_ln(100.0f32.ln(), intercept_db);

        assert!(full.abs() > 1.0, "the test frequency has to actually move");
        assert!((half - (full * 0.5)).abs() < 1e-3);
    }

    /// What "the cut part goes back to the pink noise curve" means: outside the band the capture
    /// stops contributing entirely.
    #[test]
    fn cutting_the_low_end_hands_it_back_to_the_polynomial() {
        let intercept_db = -12.0;
        let grid = sloped_grid(-6.0, 40.0);
        let blend = CaptureBlend::new(
            &grid,
            1000.0f32.ln(),
            intercept_db,
            1.0,
            200.0,
            CAPTURE_MAX_HZ,
        );

        // Inside the band the capture applies in full
        assert!((blend.weight_ln(1000.0f32.ln()) - 1.0).abs() < 1e-3);
        assert!((blend.weight_ln(200.0f32.ln()) - 1.0).abs() < 1e-3);
        // A full octave below the corner it is gone, and the polynomial is back on its own
        assert!(blend.weight_ln(100.0f32.ln()).abs() < 1e-3);
        assert_eq!(blend.delta_ln(100.0f32.ln(), intercept_db), 0.0);
        // And in between it is partway, rather than a step that would leave a ledge in the curve
        let halfway = blend.weight_ln(141.0f32.ln());
        assert!(
            halfway > 0.1 && halfway < 0.9,
            "expected a gradual fade, got {halfway}"
        );
    }

    #[test]
    fn cutting_the_high_end_hands_it_back_too() {
        let grid = sloped_grid(-6.0, 40.0);
        let blend = CaptureBlend::new(&grid, 1000.0f32.ln(), -12.0, 1.0, CAPTURE_MIN_HZ, 5000.0);

        assert!((blend.weight_ln(5000.0f32.ln()) - 1.0).abs() < 1e-3);
        assert!(blend.weight_ln(10_000.0f32.ln()).abs() < 1e-3);
    }

    /// Dragging the two corners past each other should trail off to nothing rather than do
    /// something wild.
    #[test]
    fn an_inverted_band_applies_nothing() {
        let grid = sloped_grid(-6.0, 40.0);
        let blend = CaptureBlend::new(&grid, 1000.0f32.ln(), -12.0, 1.0, 8000.0, 200.0);

        for hz in [20.0f32, 100.0, 1000.0, 8000.0, 19_000.0] {
            assert!(blend.weight_ln(hz.ln()) < 0.5, "at {hz} Hz");
        }
    }

    #[test]
    fn the_default_band_covers_everything() {
        let grid = sloped_grid(-6.0, 40.0);
        let blend = full_range_blend(&grid, -12.0, 1.0);

        for hz in [20.0f32, 1000.0, 20_000.0] {
            assert_eq!(blend.weight_ln(hz.ln()), 1.0, "at {hz} Hz");
        }
    }
}
