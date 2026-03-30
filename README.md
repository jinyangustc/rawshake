# rawshake: Direct Raspberry Shake Reader

A Python library for reading raw data directly from Raspberry Shake geophones over
serial connection, without requiring Shake OS or any other seismology software.

## Requirements

- Python 3.10+
- `pyserial`, `numpy`, `scipy`
- A Raspberry Shake geophone (RS1D or RS4D)
- Any 32/64-bit Linux system (tested on Ubuntu)

## Features

- Self-contained: does not depend on Shake OS or other seismology software
- Support for RS1D and RS4D Raspberry Shake devices
- Thread-safe background reading with message queuing
- Rolling signal conditioner for real-time detrending and filtering

## Installation

Using `uv` (recommended):

```bash
uv add git+https://github.com/jinyangustc/rawshake.git
```

Or add to your project's `pyproject.toml`:

```toml
[project]
dependencies = [
    "rawshake @ git+https://github.com/jinyangustc/rawshake.git",
]
```

Using pip:

```bash
pip install git+https://github.com/jinyangustc/rawshake.git
```

## Quick Start

```python
from rawshake import GeoReader
from rawshake.geophone import get_samples

reader = GeoReader()
reader.start()

try:
    while True:
        msg = reader.get(timeout=1.0)
        if msg:
            timestamp, samples = get_samples(msg)
            print(f"Timestamp: {timestamp}")
            print(f"Samples: {samples}")
except KeyboardInterrupt:
    reader.stop()
```

## Signal Conditioning

Raw geophone counts include a DC bias from the analog front end (the single-supply
ADC chain adds an offset so the AC geophone signal sits within the ADC's input range).
For most analysis you will want to remove this bias and optionally bandpass-filter the
signal before use.

`RollingConditioner` maintains a rolling history buffer per channel and applies
zero-phase conditioning (DC removal, detrend, optional high-pass / low-pass) to each
incoming batch of samples. Keeping several seconds of history gives the filter
sufficient context to avoid edge transients on the samples you actually use.

```python
from rawshake import GeoReader, RollingConditioner
from rawshake.geophone import get_samples

reader = GeoReader()
reader.start()

# Buffer 5 s of history; high-pass at 4.5 Hz (geophone corner frequency)
conditioner = RollingConditioner(fs=200, seconds=5, hp=4.5)

try:
    while True:
        msg = reader.get(timeout=1.0)
        if msg:
            timestamp, raw = get_samples(msg)
            conditioned = conditioner.push(raw)
            # conditioned[ch] is a float64 array of the newly arrived samples,
            # centered around zero and filtered
except KeyboardInterrupt:
    reader.stop()
```

For one-shot processing of an already-collected window, use `condition` directly:

```python
from rawshake.processing import condition

filtered = condition(raw_counts, fs=200, hp=4.5, lp=40.0)
```

## Configuration

The GeoReader class accepts the following configuration parameters:

- `port`: Serial port to connect to (default: `'/dev/serial0'`)
- `baudrate`: Baud rate for serial communication (default: `230400`)
- `timeout`: Serial port timeout in seconds (default: `None`)
- `poll_interval`: How frequently to poll for new data in seconds (default: `0.1`)
- `n_frames`: Number of frames to collect for each message (default: `4`)
- `frame_interval`: Time interval between frames in milliseconds (default: `250`)

## Message Format

The library processes geophone data into `GeoMsg` objects containing:

- Timestamp information
- Channel data (RS1D: SH3; RS4D: EH3, EN1, EN2, EN3)
- Frame metadata

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
