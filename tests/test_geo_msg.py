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
