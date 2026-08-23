from pathlib import Path

from orbit.cli import build_parser


def test_path_ph_parser_selects_separate_ph() -> None:
    args = build_parser().parse_args([
        "path-ph", "--root", "/tmp/project", "--target", "Bi1",
        "--run-id", "run-1", "--source", "helper", "--iteration", "10",
        "--submit",
    ])
    assert args.command == "path-ph"
    assert args.root == Path("/tmp/project")
    assert args.response_kind == "ph"
    assert args.separate_response_root is True
    assert args.submit is True


def test_path_polarization_parser_selects_separate_polarization() -> None:
    args = build_parser().parse_args([
        "path-polarization", "--target", "Bi2", "--run-id", "run-2",
    ])
    assert args.command == "path-polarization"
    assert args.response_kind == "polarization"
    assert args.separate_response_root is True
    assert args.submit is False


def test_combined_response_parser_remains_backward_compatible() -> None:
    args = build_parser().parse_args([
        "path-response", "--target", "Bi1", "--run-id", "run-3",
        "--kind", "ph",
    ])
    assert args.command == "path-response"
    assert args.kind == "ph"
    assert args.separate_response_root is False
