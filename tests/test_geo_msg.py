import json

import pytest

from rawshake.geophone import parse_buffer


@pytest.fixture
def timestamp_msg():
    msg = '{"MSEC": 1463804500,"LS": 0}'
    return msg


@pytest.fixture
def rs1d_msg():
    msg = '{"MA": "RS1D-8-4.11","DF": "1.0","CN": "SH3","TS": 0,"TSM": 0,"TQ": 45,"SI":  5000,"DS": ["3836","3E38","38D5","28E3","1741","A1A","9CC","207A","4ABE","7429","8956","84DD","6911","3EF0","187B","61A","B5E","212E","3B28","4C11","4ADF","3692","1966","56B","673","1E36","440B","6D3B","8F8B","9F0A","9638","7D60","621B","496F","36D7","2E70","3042","34F0","36CF","3A7D","43F5","4E08","4F5E","43BD","2E14","1359","FFFFFA12","FFFFEA97","FFFFEBF6","381"]}'
    return msg


@pytest.fixture
def rs4d_msg():
    # TODO: add a rs4d messag example
    msg = ''
    return msg


def test_parse(timestamp_msg, rs1d_msg, rs4d_msg):
    decoder = json.JSONDecoder()
    good_t_msg, buffer = parse_buffer(timestamp_msg, decoder)
    assert good_t_msg[0]['MSEC'] == 1463804500
    assert len(buffer) == 0

    good_rs1d_msg, buffer = parse_buffer(rs1d_msg, decoder)
    assert good_rs1d_msg[0]['MA'] == 'RS1D-8-4.11'
    assert good_rs1d_msg[0]['CN'] == 'SH3'
    assert len(good_rs1d_msg[0]['DS']) == 50
    assert len(buffer) == 0

    # TODO: update with rs4d message
    good_rs4d_msg, buffer = parse_buffer(rs4d_msg, decoder)
    assert len(good_rs4d_msg) == 0
    assert len(buffer) == 0


class TestParseBufferCorruption:
    """Tests for parse_buffer corruption recovery."""

    def setup_method(self):
        self.decoder = json.JSONDecoder()
        self.valid_msg = '{"MSEC": 1000,"LS": 0}'

    def test_corrupted_brace_before_valid_msg(self):
        """Corrupted '{garbage' followed by a valid message -> skip and recover."""
        buffer = '{garbage' + self.valid_msg
        msgs, remaining = parse_buffer(buffer, self.decoder)
        assert len(msgs) == 1
        assert msgs[0]['MSEC'] == 1000
        assert remaining == ''

    def test_corrupted_brace_before_multiple_valid(self):
        """Corrupted block followed by multiple valid messages."""
        valid2 = '{"MSEC": 2000,"LS": 0}'
        buffer = '{corrupt' + self.valid_msg + valid2
        msgs, remaining = parse_buffer(buffer, self.decoder)
        assert len(msgs) == 2
        assert msgs[0]['MSEC'] == 1000
        assert msgs[1]['MSEC'] == 2000
        assert remaining == ''

    def test_last_brace_incomplete_waits(self):
        """A single incomplete '{...' at end of buffer -> wait for more data."""
        buffer = '{"MSEC": 100'
        msgs, remaining = parse_buffer(buffer, self.decoder)
        assert len(msgs) == 0
        assert remaining == buffer

    def test_valid_then_incomplete_tail(self):
        """Valid message followed by incomplete tail -> parse valid, keep tail."""
        buffer = self.valid_msg + '{"MSEC": 200'
        msgs, remaining = parse_buffer(buffer, self.decoder)
        assert len(msgs) == 1
        assert msgs[0]['MSEC'] == 1000
        assert remaining == '{"MSEC": 200'

    def test_oversized_single_block_discarded(self):
        """A single '{' followed by >_MAX_MSG_SIZE bytes -> discard."""
        from rawshake.geophone import _MAX_MSG_SIZE

        buffer = '{' + 'x' * _MAX_MSG_SIZE
        msgs, remaining = parse_buffer(buffer, self.decoder)
        assert len(msgs) == 0
        assert remaining == ''

    def test_non_brace_garbage_skipped(self):
        """Non-'{' garbage before a valid message -> skip to next '{'."""
        buffer = 'garbage' + self.valid_msg
        msgs, remaining = parse_buffer(buffer, self.decoder)
        assert len(msgs) == 1
        assert msgs[0]['MSEC'] == 1000
        assert remaining == ''

    def test_incremental_recovery(self):
        """Simulate incremental reads: corrupted head, then valid data arrives."""
        # First read: only corrupted data, no second '{'
        buffer = '{corrupt_data'
        msgs, buffer = parse_buffer(buffer, self.decoder)
        assert len(msgs) == 0
        assert buffer == '{corrupt_data'  # waiting

        # Second read: valid message arrives
        buffer += self.valid_msg
        msgs, buffer = parse_buffer(buffer, self.decoder)
        assert len(msgs) == 1
        assert msgs[0]['MSEC'] == 1000
        assert buffer == ''
