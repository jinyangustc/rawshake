from collections import deque
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy import signal as scipy_signal

from .geophone import Channel


def condition(
    counts: list[int],
    fs: int,
    hp: float | None = None,
    lp: float | None = None,
) -> npt.NDArray[np.float64]:
    """
    Condition a window of raw ADC counts: remove DC bias and linear drift,
    then optionally apply high-pass and/or low-pass filters.

    Uses zero-phase filtering (sosfiltfilt) so no phase distortion is introduced.
    Requires the full window to be available at once.

    Parameters
    ----------
    counts : list[int]
        Raw ADC counts from a single channel.
    fs : int
        Sample rate in Hz.
    hp : float or None, optional
        High-pass corner frequency in Hz. None skips the filter.
    lp : float or None, optional
        Low-pass corner frequency in Hz. None skips the filter.

    Returns
    -------
    npt.NDArray[np.float64]
        Conditioned samples as floats, centered around zero.
    """
    x = np.asarray(counts, dtype=np.float64)
    x -= np.median(x)
    x = scipy_signal.detrend(x, type='linear')
    if hp is not None:
        sos = scipy_signal.butter(4, hp, btype='highpass', fs=fs, output='sos')
        x = scipy_signal.sosfiltfilt(sos, x)
    if lp is not None:
        sos = scipy_signal.butter(4, lp, btype='lowpass', fs=fs, output='sos')
        x = scipy_signal.sosfiltfilt(sos, x)
    return x.astype(np.float64)


@dataclass
class RollingConditioner:
    """
    Conditions geophone samples using a rolling buffer.

    Maintains a history of `seconds` seconds per channel so that
    condition_window has sufficient left-context for filtering. Note that
    the newest samples are always at the trailing edge of the window, so
    right-edge filter transients still affect the most recent output.

    Parameters
    ----------
    fs : int
        Sample rate in Hz.
    seconds : int
        Rolling buffer length in seconds. Should be long enough that the
        filter has settled well before the trailing edge (rule of thumb:
        at least 3 / hp seconds).
    hp : float or None
        High-pass corner frequency in Hz.
    lp : float or None
        Low-pass corner frequency in Hz.
    """

    fs: int
    seconds: int = 5
    hp: float | None = None
    lp: float | None = None
    _buffers: dict[Channel, deque[int]] = field(default_factory=dict, init=False)

    def push(
        self,
        samples: dict[Channel, list[int]],
    ) -> dict[Channel, npt.NDArray[np.float64]]:
        out: dict[Channel, npt.NDArray[np.float64]] = {}
        for ch, new_samples in samples.items():
            if ch not in self._buffers:
                self._buffers[ch] = deque(maxlen=self.fs * self.seconds)
            self._buffers[ch].extend(new_samples)
            conditioned = condition(
                list(self._buffers[ch]), self.fs, hp=self.hp, lp=self.lp
            )
            out[ch] = conditioned[-len(new_samples) :]
        return out
