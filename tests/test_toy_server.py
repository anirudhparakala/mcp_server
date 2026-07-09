import sys
from pathlib import Path

# examples/ is not an installed package; add it to the path for the test run
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from toy_server import ping, echo, greeting


def test_ping_returns_pong():
    assert ping() == "pong"


def test_echo_returns_input_unchanged():
    assert echo("hello world") == "hello world"


def test_echo_handles_empty_string():
    assert echo("") == ""


def test_greeting_formats_name():
    assert greeting("Ada") == "Hello, Ada!"
