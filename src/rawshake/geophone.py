import json
import logging
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import serial

RS1D_CHANNELS = ['SH3']
RS4D_CHANNELS = ['EH3', 'EN1', 'EN2', 'EN3']

logger = logging.getLogger(__name__)


def get_logger(
    user_logger: logging.Logger | None = None,
    level: int | None = None,
    handler: logging.Handler | None = None,
) -> None:
    """
    Configure logging for the library.

    Parameters
    ----------
    user_logger : logging.Logger or None, optional
        If provided, the library will use this logger instead of its own
    level : int or None, optional
        If provided and no user_logger is given, sets the log level
    handler : logging.Handler or None, optional
        If provided and no user_logger is given, adds this handler

    Notes
    -----
    This function configures the global logger for the library. If a user_logger
    is provided, it will be used directly. Otherwise, it configures the default
    logger with the specified level (defaults to DEBUG) and handler (defaults to
    StreamHandler).
    """
    global logger

    if user_logger is not None:
        # Use the user-provided logger
        logger = user_logger
        return

    # Configure our default logger
    if level is None:
        level = logging.DEBUG

    if handler is None:
        handler = logging.StreamHandler()
        # formatter = logging.Formatter(
        #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        # )
        # handler.setFormatter(formatter)

    logger.setLevel(level)

    # Remove all existing handlers to avoid duplicate messages
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    logger.addHandler(handler)
    logger.propagate = False  # Prevent propagation to root logger


# Set up default configuration when module is imported
get_logger()


@dataclass
class GeoMsgFrame:
    """
    A single frame of geophone data.

    Parameters
    ----------
    _timestamp : dict[str, int]
        Dictionary containing timestamp information with 'MSEC' key and timestamp_ns
    _channels : dict[str, dict]
        Dictionary mapping channel names to their data

    Notes
    -----
    Each frame contains timestamp information and channel data for one sampling period (250ms).
    """

    _timestamp: dict[str, int]
    _channels: dict[str, dict]

    @property
    def timestamp(self) -> int:
        return self._timestamp['MSEC']


@dataclass
class GeoMsg:
    """
    A complete geophone message containing multiple frames.

    Parameters
    ----------
    frames : list[GeoMsgFrame]
        List of GeoMsgFrame objects containing the data
    frame_interval : int, optional
        Frame length in milliseconds, by default 250
    n_frames : int, optional
        Number of frames in the message, by default 4

    Notes
    -----
    Represents a complete set of geophone data across multiple time frames,
    typically used for processing continuous data streams.
    """

    frames: list[GeoMsgFrame]
    frame_interval: int = 250
    n_frames: int = 4


def hex_to_signed(hex_str: str, bits: int = 16) -> int:
    """Convert a hex string to a signed integer with given bit length."""
    value = int(hex_str, 16)
    if value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value


def get_samples(msg: GeoMsg) -> tuple[int, dict[str, list[int]]]:
    """
    Extract timestamp and sample data from a GeoMsg object.

    Parameters
    ----------
    msg : GeoMsg
        The geophone message to process

    Returns
    -------
    tuple[int, dict[str, list[int]]]
        A tuple containing:
        - timestamp in nanoseconds
        - dictionary mapping channel names to lists of signed integer samples

    Notes
    -----
    Converts hexadecimal sample values to signed integers and organizes them by channel.
    Ensures samples are in chronological order and no frames are missing.
    """
    assert len(msg.frames) == msg.n_frames

    result = defaultdict(list)
    prev_timestamp = -1
    for frame in msg.frames:
        for ch, ch_data in frame._channels.items():
            result[ch].extend(ch_data['DS'])
            assert frame.timestamp > prev_timestamp
            prev_timestamp = frame.timestamp

    for k, v in result.items():
        result[k] = [hex_to_signed(x, bits=16) for x in v]
    return msg.frames[0]._timestamp['timestamp_ns'], dict(result)


class GeoMsgAssembler:
    """
    Assembles complete geophone messages from individual serial data segments.

    Parameters
    ----------
    device_type : str
        Type of device ('RS1D' or 'RS4D')
    n_frames : int
        Number of frames to collect for a complete message
    frame_interval : int, optional
        Frame length in milliseconds, by default 250

    Notes
    -----
    This class maintains internal buffers for both raw serial messages and assembled frames.
    It handles the assembly of complete GeoMsg objects from individual serial data segments,
    ensuring proper ordering and completeness of the data.
    """

    def __init__(
        self,
        device_type: str,
        n_frames: int,
        frame_interval: int = 250,
    ):
        self.device_type = device_type
        self.n_frames = n_frames
        self.frame_interval = frame_interval

        if self.device_type == 'RS1D':
            self.channels = sorted(RS1D_CHANNELS)
        elif self.device_type == 'RS4D':
            self.channels = sorted(RS4D_CHANNELS)
        else:
            raise ValueError(f'unsuported device type: {self.device_type}')

        # good frame:
        #    RS1D: {MSEC}{SH3} | {MSEC}{SH3}
        #    RS4D: {MSEC}{EN1}{EN2}{EN3}{EH3} | {MSEC}{EN1}{EN2}{EN3}{EH3}
        self.serial_dq = deque()  # type: deque[dict]
        self.frame_dq = deque()  # type: deque[GeoMsgFrame]

    def add(self, serial_msg: dict) -> None:
        self.serial_dq.append(serial_msg)

        # seek the beginning of a frame
        while self.serial_dq and 'MSEC' not in self.serial_dq[0]:
            self.serial_dq.popleft()

        # a frame contains 1 header and n_channels data segments
        if len(self.serial_dq) < 1 + len(self.channels):
            # not enough data
            return

        channels = [self.serial_dq[i + 1]['CN'] for i in range(len(self.channels))]
        if sorted(channels) == self.channels:
            # a complete frame
            header = self.serial_dq.popleft()
            segments = [self.serial_dq.popleft() for _ in range(len(self.channels))]
            frame = GeoMsgFrame(header, {x['CN']: x for x in segments})
            self.frame_dq.append(frame)
        else:
            # we should have enough serial messages, but cannot assemble a complete frame
            # potential message loss due to corruption
            header = self.serial_dq.popleft()
            logger.warning(f'Drop {header["MSEC"]} at {time.time_ns()}')

    def get(self) -> GeoMsg | None:
        """Get a complete GeoMsg from the assembler. Return None if no one is available."""
        if len(self.frame_dq) < self.n_frames:
            return None

        if self.n_frames < 2:
            frame = self.frame_dq.popleft()
            msg = GeoMsg([frame], self.frame_interval, self.n_frames)
            return msg

        for i in range(1, self.n_frames):
            if (
                self.frame_dq[i].timestamp - self.frame_dq[i - 1].timestamp
                != self.frame_interval
            ):
                # missing frames between i and i-1, drop all frames until i-1 including i-1
                frames = [self.frame_dq.popleft() for _ in range(i)]
                logger.warning(
                    f'Drop frames {[x.timestamp for x in frames]} due to missing frame'
                )
                return None

        frames = [self.frame_dq.popleft() for _ in range(self.n_frames)]
        msg = GeoMsg(frames, self.frame_interval, self.n_frames)
        return msg

    def __repr__(self) -> str:
        serial_buffer = []
        for x in self.serial_dq:
            if 'MSEC' in x:
                serial_buffer.append(f'MSEC-{x["MSEC"]}')
            else:
                serial_buffer.append('DS')
        frame_buffer = [str(x.timestamp) for x in self.frame_dq]
        return f'serial_buffer={serial_buffer}, frame_buffer={frame_buffer}'


def parse_buffer(buffer: str, decoder: json.JSONDecoder) -> Tuple[List[Dict], str]:
    """
    Parse complete JSON objects from the buffer.

    Parameters
    ----------
    buffer : str
        The string buffer containing JSON data.
    decoder : json.JSONDecoder
        An instance of json.JSONDecoder.

    Returns
    -------
    Tuple[List[Dict], str]
        A tuple containing:
        - messages : List[Dict]
            A list of parsed serial messages (JSON objects as dictionaries).
        - buffer : str
            The remaining buffer after extracting complete serial messages.

    Notes
    -----
    The function attempts to parse complete JSON objects from the start of the buffer.
    Invalid messages and non-JSON data are filtered out.
    """
    messages = []
    while buffer:
        try:
            # try to decode a JSON object from the start of the buffer
            msg, idx = decoder.raw_decode(buffer)
            if 'MSEC' in msg:
                msg['timestamp_ns'] = time.time_ns()
            messages.append(msg)
            # remove the processed message and any leading whitespace
            buffer = buffer[idx:].lstrip()
        except json.JSONDecodeError:
            # find the next possible start of a JSON object
            start_idx = buffer.find('{')
            if start_idx == -1:
                # no valid JSON start found; clear the buffer
                buffer = ''
                break
            elif start_idx > 0:
                # discard any incomplete leading data and try again
                buffer = buffer[start_idx:]
                continue
            else:
                # buffer starts with '{' but isn't a complete JSON object
                # wait for more data
                break

    # filter out invalid messages
    messages = [x for x in messages if isinstance(x, dict)]

    return messages, buffer


def read_geophone(
    msg_queue: queue.Queue,
    port: str = '/dev/serial0',
    baudrate: int = 230_400,
    timeout: int | None = None,
    poll_interval: float = 0.1,
    n_frames: int = 4,
    frame_interval: int = 250,
    stop_event: threading.Event | None = None,
):
    """
    Read and parse raw data from the geophone of Raspberry Shake over the serial
    connection.

    Parameters
    ----------
    msg_queue : queue.Queue
        Queue to put parsed GeoMsg objects into
    port : str, optional
        Serial port to connect to, by default '/dev/serial0'
    baudrate : int, optional
        Baud rate for serial communication, by default 230400
    timeout : int or None, optional
        Serial port timeout in seconds, by default None
    poll_interval : float, optional
        How frequently the code polls the serial port for new data in seconds, by default 0.1s
    n_frames : int, optional
        Number of frames to collect for each message, by default 4
    frame_interval : int, optional
        Frame length in milliseconds, by default 250
    stop_event : threading.Event or None, optional
        Event to signal the thread to stop, by default None

    Notes
    -----
    This function runs in an infinite loop until interrupted:
    1. Reads available data from the serial port
    2. Decodes the data as UTF-8
    3. Parses complete JSON messages
    4. Assembles messages into GeoMsg objects and puts them in the queue

    The function can be interrupted by setting the stop_event or with Ctrl+C (KeyboardInterrupt).
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        logger.info(f'Connected to {port} at {baudrate} baud')
    except Exception as e:
        logger.error(f'Error opening {port}: {e}')
        return

    buffer = ''
    decoder = json.JSONDecoder()
    assembler = None

    try:
        while not (stop_event and stop_event.is_set()):
            in_waiting = f' in waiting: {ser.in_waiting} '
            logger.debug(f'{in_waiting:-^80}')
            raw_data = ser.read(ser.in_waiting)
            if not raw_data:
                time.sleep(poll_interval)
                continue

            try:
                buffer += raw_data.decode('utf-8')
                logger.debug(f' RAW | {raw_data}')
                logger.debug(f'BUFF | {buffer}')
            except UnicodeDecodeError:
                logger.warning('Received undecodable bytes, skipping chunks.')
                time.sleep(poll_interval)
                continue

            msgs, buffer = parse_buffer(buffer, decoder)
            for i, msg in enumerate(msgs):
                if assembler is None and 'MA' in msg:
                    # initialize assembler
                    # Example: 'RS1D-8-4.11'
                    device_type = msg['MA'].split('-')[0]
                    assembler = GeoMsgAssembler(device_type, n_frames, frame_interval)

                i = f'M{i}'
                logger.debug(f'{i:>4} | {msg}')

                if assembler:
                    assembler.add(msg)
                    geo_msg = assembler.get()

                    if geo_msg:
                        msg_queue.put(geo_msg)

            logger.debug(f'ASMB | {assembler}')
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        logger.info('Existing read_geohpone')
    except Exception as e:
        logger.error(f'Error reading from port {port}: {e}')
    finally:
        ser.close()
        logger.info(f'Connection to {port} closed.')


class GeoReader:
    """
    Thread-safe geophone reader that runs in the background.

    Parameters
    ----------
    port : str, optional
        Serial port to connect to, by default '/dev/serial0'
    baudrate : int, optional
        Baud rate for serial communication, by default 230400
    timeout : int or None, optional
        Serial port timeout in seconds, by default None
    poll_interval : float, optional
        How frequently to poll the serial port for new data in seconds, by default 0.1
    n_frames : int, optional
        Number of frames to collect for each message, by default 4
    frame_interval : int, optional
        Time interval between frames in milliseconds, by default 250

    Notes
    -----
    This class manages a background thread that continuously reads from the geophone
    and puts complete messages into a thread-safe queue. The main thread can retrieve
    messages using the get() method.
    """

    def __init__(
        self,
        port: str = '/dev/serial0',
        baudrate: int = 230_400,
        timeout: int | None = None,
        poll_interval: float = 0.1,
        n_frames: int = 4,
        frame_interval: int = 250,
    ):
        self.msg_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=read_geophone,
            args=(self.msg_queue,),
            kwargs={
                'port': port,
                'baudrate': baudrate,
                'timeout': timeout,
                'poll_interval': poll_interval,
                'n_frames': n_frames,
                'frame_interval': frame_interval,
                'stop_event': self.stop_event,
            },
            daemon=True,
        )

    def start(self):
        """Start the reader thread"""
        self.thread.start()

    def stop(self):
        """Stop the reader thread"""
        self.stop_event.set()
        self.thread.join()

    def get(self, timeout: float | None = None) -> GeoMsg | None:
        """
        Get the next message from the queue.

        Parameters
        ----------
        timeout : float or None, optional
            How long to wait for a message in seconds, by default None
            If None, wait indefinitely

        Returns
        -------
        GeoMsg or None
            The next complete geophone message, or None if timeout occurred

        Notes
        -----
        This method is thread-safe and can be called from any thread.
        """
        try:
            return self.msg_queue.get(timeout=timeout)
        except queue.Empty:
            return None


def _main():
    reader = GeoReader()
    reader.start()

    try:
        while True:
            msg = reader.get(timeout=1.0)
            if msg:
                logger.debug(f' GEO | {msg}')
                ts, samples = get_samples(msg)
                logger.debug(f'SAMP | {ts} {samples}')

    except KeyboardInterrupt:
        print('\nStopping reader...')
        reader.stop()


if __name__ == '__main__':
    _main()
