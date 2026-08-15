"""ViBe, transcribed from Barnich and Van Droogenbroeck 2011, Appendix A.

IEEE TIP 20(6):1709-1724. ViBe is NOT in OpenCV at any version, so the arm
must be written. Appendix A is a complete C-like listing including default
values, so this is transcription and validation rather than reconstruction.

PARAMETERS, the paper's own, declared as defaults and not chosen here:
    N    = 20   samples per pixel
    R    = 20   sphere radius
    Nmin =  2   minimum cardinality to be background
    phi  = 16   time subsampling factor

PROVENANCE OF EACH DIFFERS AND THE CHAPTER MUST SAY SO. Nmin and N were swept
by the authors on their "pets" sequence by PCC: Nmin best at 2 and 3, set to 2
because a rise in Nmin raises cost; N saturates above 20, set to 20 at the
start of the plateau. R and phi were NOT swept. The paper says only that in
their experience these work well, and calls R = 20 "an educated choice"
corresponding to a perceptible colour difference. So two defaults are measured
optima on one sequence and two are author judgement.

VIBE IS NON-DETERMINISTIC BY DESIGN. The paper states that neither the
sample-selection policy nor the spatial propagation is deterministic, and that
rerunning the same sequence always gives slightly different results. The RNG
is therefore seeded and the seed recorded on every run. The original Chapter 4
code used an unseeded ViBe RNG, which is one of the reasons its runs were not
reproducible.

THREE DEVIATIONS FROM THE LITERAL LISTING, all deliberate, all reportable.

1. CLASSIFY-THEN-UPDATE, not the listing's interleaved loop. Appendix A walks
   pixels in raster order, classifying and updating in the same pass, so a
   spatial propagation from one pixel can reach a pixel not yet classified in
   the same frame. This implementation classifies every pixel against the
   model as it stood at the end of the previous frame, then applies all
   updates. That matches the paper's own formal statement, which compares the
   current value to M^{t-1}, and it is what makes the operation vectorisable
   at 1080p and 4K. It is not bit-identical to the listing.

2. GRAYSCALE ONLY. Appendix A is written for grayscale, and the paper states
   R = 20 for MONOCHROMATIC images. It evaluates an RGB version too but gives
   no separate radius for it. A colour variant would therefore need a radius
   the paper does not supply, which would be a chosen parameter. Out of scope
   for 1b-M; flagged rather than silently decided.

3. SCATTER COLLISIONS. When two pixels propagate into the same neighbour in
   one frame, numpy's fancy assignment keeps the last write. The listing has
   the same last-write-wins behaviour in raster order, so the semantics match
   even though the winner may differ. Deterministic under a fixed seed.

NOTHING HERE IS TUNED. Every value is the paper's, and any 1b-M output using
this class is labelled provisional.
"""

from __future__ import annotations

import numpy as np

# Paper defaults, Appendix A. Do not change these to fit a result.
DEFAULT_N = 20
DEFAULT_R = 20
DEFAULT_NMIN = 2
DEFAULT_PHI = 16

# 8-connected neighbourhood, the paper's choice for initialisation and the
# larger of the two options it allows for propagation.
NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1), (0, 1),
              (1, -1), (1, 0), (1, 1)]


class ViBe:
    """Visual Background Extractor, grayscale.

    Usage mirrors OpenCV's subtractors so the sweep code can treat all three
    arms alike:

        vibe = ViBe(seed=0)
        mask = vibe.apply(frame_bgr_or_gray)

    apply() returns a uint8 mask with 255 for foreground and 0 for background,
    matching OpenCV's convention.
    """

    def __init__(self, seed: int, n_samples: int = DEFAULT_N,
                 radius: int = DEFAULT_R, n_min: int = DEFAULT_NMIN,
                 phi: int = DEFAULT_PHI):
        if seed is None:
            raise ValueError(
                "ViBe requires an explicit seed. The algorithm is "
                "non-deterministic by design and an unseeded run is not "
                "reproducible, which is how the original Chapter 4 code "
                "failed.")
        self.seed = int(seed)
        self.N = int(n_samples)
        self.R = int(radius)
        self.Nmin = int(n_min)
        self.phi = int(phi)
        self.rng = np.random.default_rng(self.seed)
        self.samples: np.ndarray | None = None   # (H, W, N) uint8
        self.frame_index = 0

    # ------------------------------------------------------------- setup ---
    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 3:
            # BGR to gray with the standard luma weights. Kept explicit rather
            # than calling cv2 so the class has no OpenCV dependency.
            b, g, r = (frame[:, :, i].astype(np.float32) for i in range(3))
            return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
        raise ValueError(f"unexpected frame shape {frame.shape}")

    def _initialise(self, gray: np.ndarray) -> None:
        """Fill each pixel's model with values drawn uniformly from its
        8-connected neighbourhood in the first frame (paper, Sec III-B).

        The paper notes this introduces a GHOST wherever a moving object
        occupies the first frame, and that the update mechanism absorbs it
        over subsequent frames. That is expected behaviour, not a fault.
        """
        H, W = gray.shape
        self.samples = np.empty((H, W, self.N), np.uint8)
        # Precompute the eight shifted planes once; each of the N draws then
        # selects per pixel among them. np.roll wraps at the border, which
        # differs from the listing's implicit edge handling but only affects
        # the outermost row and column of the very first frame.
        planes = [np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
                  for dy, dx in NEIGHBOURS]
        for k in range(self.N):
            pick = self.rng.integers(0, len(NEIGHBOURS), (H, W))
            out = np.empty_like(gray)
            for j in range(len(NEIGHBOURS)):
                sel = pick == j
                if sel.any():
                    out[sel] = planes[j][sel]
            self.samples[:, :, k] = out

    # ------------------------------------------------------------- apply ---
    def apply(self, frame: np.ndarray) -> np.ndarray:
        gray = self._to_gray(frame)
        if self.samples is None:
            self._initialise(gray)
            self.frame_index = 1
            # The paper's claim is usable segmentation from the SECOND frame.
            # The first therefore returns all background rather than a
            # meaningless mask.
            return np.zeros(gray.shape, np.uint8)

        H, W = gray.shape
        v = gray.astype(np.int16)

        # Classification: count model samples within the sphere of radius R.
        # The listing stops once Nmin matches are found; counting all N is
        # equivalent in outcome and is what makes this vectorisable.
        close = (np.abs(self.samples.astype(np.int16) - v[:, :, None])
                 < self.R)
        count = close.sum(axis=2)
        is_bg = count >= self.Nmin
        mask = np.where(is_bg, 0, 255).astype(np.uint8)

        # Conservative update: only pixels classified background may update.
        p_self = (self.rng.random((H, W)) < 1.0 / self.phi) & is_bg
        if p_self.any():
            idx = self.rng.integers(0, self.N, (H, W))
            ys, xs = np.nonzero(p_self)
            self.samples[ys, xs, idx[ys, xs]] = gray[ys, xs]

        # Spatial propagation, an INDEPENDENT 1/phi draw per the listing.
        p_nbr = (self.rng.random((H, W)) < 1.0 / self.phi) & is_bg
        if p_nbr.any():
            ys, xs = np.nonzero(p_nbr)
            pick = self.rng.integers(0, len(NEIGHBOURS), ys.shape)
            offs = np.array(NEIGHBOURS)
            ny = ys + offs[pick, 0]
            nx = xs + offs[pick, 1]
            inside = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            ny, nx = ny[inside], nx[inside]
            src_y, src_x = ys[inside], xs[inside]
            k = self.rng.integers(0, self.N, ny.shape)
            self.samples[ny, nx, k] = gray[src_y, src_x]

        self.frame_index += 1
        return mask

    # ------------------------------------------------------------- state ---
    def describe(self) -> dict:
        return {"method": "ViBe", "N": self.N, "R": self.R,
                "Nmin": self.Nmin, "phi": self.phi, "seed": self.seed,
                "colour": "grayscale",
                "source": "Barnich & Van Droogenbroeck 2011, Appendix A",
                "deviations": ["classify-then-update, not interleaved",
                               "grayscale only, paper gives no colour radius",
                               "scatter collisions resolved last-write-wins"],
                "frames_seen": self.frame_index}
