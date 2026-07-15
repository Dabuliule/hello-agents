from codecraft.tui.theme import (
    TUIColorScheme,
    TUIThemeMode,
    parse_osc_background,
    resolve_color_scheme,
)


def test_explicit_tui_theme_skips_terminal_detection():
    def unexpected_query() -> tuple[int, int, int] | None:
        raise AssertionError("explicit themes must not query the terminal")

    assert (
        resolve_color_scheme(
            TUIThemeMode.LIGHT,
            env={"COLORFGBG": "15;0"},
            terminal_background=unexpected_query,
        )
        == TUIColorScheme.LIGHT
    )
    assert (
        resolve_color_scheme(
            TUIThemeMode.DARK,
            env={"COLORFGBG": "0;15"},
            terminal_background=unexpected_query,
        )
        == TUIColorScheme.DARK
    )


def test_auto_tui_theme_uses_colorfgbg_without_querying_terminal():
    def unexpected_query() -> tuple[int, int, int] | None:
        raise AssertionError("COLORFGBG should avoid an OSC query")

    assert (
        resolve_color_scheme(
            TUIThemeMode.AUTO,
            env={"COLORFGBG": "15;0"},
            terminal_background=unexpected_query,
        )
        == TUIColorScheme.DARK
    )
    assert (
        resolve_color_scheme(
            TUIThemeMode.AUTO,
            env={"COLORFGBG": "0;15"},
            terminal_background=unexpected_query,
        )
        == TUIColorScheme.LIGHT
    )


def test_auto_tui_theme_uses_osc_background_and_light_fallback():
    assert (
        resolve_color_scheme(
            TUIThemeMode.AUTO,
            env={},
            terminal_background=lambda: (24, 26, 30),
        )
        == TUIColorScheme.DARK
    )
    assert (
        resolve_color_scheme(
            TUIThemeMode.AUTO,
            env={},
            terminal_background=lambda: (242, 243, 245),
        )
        == TUIColorScheme.LIGHT
    )
    assert (
        resolve_color_scheme(
            TUIThemeMode.AUTO,
            env={},
            terminal_background=lambda: None,
        )
        == TUIColorScheme.LIGHT
    )


def test_parse_osc_background_supports_xterm_rgb_and_hex_responses():
    assert parse_osc_background(b"\x1b]11;rgb:ffff/8000/0000\x1b\\") == (
        255,
        128,
        0,
    )
    assert parse_osc_background(b"prefix\x1b]11;#102030\x07suffix") == (
        16,
        32,
        48,
    )
    assert parse_osc_background(b"\x1b]10;rgb:ffff/ffff/ffff\x1b\\") is None
