# rawshake: Direct Raspberry Shake Reader

A Python library for reading raw data directly from Raspberry Shake geophones over
serial connection, without requiring Shake OS or any other seismology software.

## Requirements

- Python 3.10+
- `pyserial`
- A Raspberry Shake geophone (RS1D or RS4D)
- Any 32/64-bit Linux system (tested on Ubuntu)

## Features

- Self-contained: does no depend on Shake OS or other seismology software
- Support for RS1D and RS4D Raspberry Shake devices
- Thread-safe background reading with message queuing

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
from rawshake.geophone import GeoReader

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

## Logging

Configure logging using the `get_logger` function:

```python
from rawshake.geophone import get_logger
import logging

# Use custom logger
my_logger = logging.getLogger('my_app')
get_logger(user_logger=my_logger)

# Or configure the default logger
get_logger(level=logging.DEBUG)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
