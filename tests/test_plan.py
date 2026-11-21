"""Planning maths, pinned against the upstream implementation's behaviour."""

from __future__ import annotations

import pytest

from app import plan as planning

AREA_720 = 1280 * 720
AREA_704 = 1024 * 704
AREA_480 = 832 * 480


class TestOutputSize:
    """Snapshots from the ported get_size_less_than_area()."""

    @pytest.mark.parametrize(
        "height,width,area,expected",
        [
            # A 16:9 still pads to 1280x704, not 1280x720: 720 is not a multiple
            # of the 64-pixel grid, and 768 rows would exceed the area budget.
            (1080, 1920, AREA_720, (704, 1280)),
            (720, 1280, AREA_720, (704, 1280)),
            (2160, 3840, AREA_720, (704, 1280)),
            (1080, 1920, AREA_480, (448, 832)),
            (1080, 1920, AREA_704, (640, 1088)),
            # Square stills fill the budget exactly.
            (1080, 1080, AREA_720, (960, 960)),
            # Smaller than the budget: pad, never upscale.
            (240, 320, AREA_720, (256, 320)),
            # Extreme panoramas keep their ratio and are not enlarged.
            (200, 2000, AREA_720, (256, 2048)),
        ],
    )
    def test_snapshot(self, height, width, area, expected):
        assert planning.output_size(height, width, area) == expected

    @pytest.mark.parametrize("height,width", [(1080, 1920), (1080, 1080), (300, 500), (4000, 3000)])
    def test_invariants(self, height, width):
        for area in (AREA_480, AREA_704, AREA_720):
            out_h, out_w = planning.output_size(height, width, area)
            assert out_h % 64 == 0 and out_w % 64 == 0
            assert out_h * out_w <= area
            if height * width <= area:
                assert out_h * out_w >= height * width * 0.99

    def test_never_upscales(self):
        out_h, out_w = planning.output_size(240, 320, AREA_720)
        assert out_w <= 320 and out_h <= 240 + 64

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            planning.output_size(0, 1920, AREA_720)
        with pytest.raises(ValueError):
            planning.output_size(1080, 1920, 0)


class TestClipCount:
    def test_rounds_up_to_cover_the_audio(self):
        assert planning.clip_count(0.5) == 1
        assert planning.clip_count(4.99) == 1
        assert planning.clip_count(5.01) == 2
        assert planning.clip_count(9.99) == 2
        assert planning.clip_count(42.0) == 9

    def test_exact_multiple_does_not_add_a_clip(self):
        # Upstream's int(n/batch)+1 asks for 2 clips at exactly 5.000 s.
        assert planning.clip_count(5.0) == 1
        assert planning.upstream_clip_count(5.0) == 2
        assert planning.clip_count(10.0) == 2
        assert planning.upstream_clip_count(10.0) == 3

    def test_saving_is_only_at_the_boundary(self):
        assert planning.upstream_clip_count(42.5) == planning.clip_count(42.5) == 9
        assert planning.upstream_clip_count(5.0) - planning.clip_count(5.0) == 1

    def test_smaller_infer_frames_means_more_clips(self):
        assert planning.clip_count(10.0, 48) == 4
        assert planning.clip_count(10.0, 160) == 1

    def test_rejects_zero(self):
        with pytest.raises(ValueError):
            planning.clip_count(0)


class TestFrames:
    def test_start_from_ref_keeps_every_frame(self):
        assert planning.output_frames(1, 80, True) == 80
        assert planning.output_frames(9, 80, True) == 720

    def test_without_start_from_ref_three_frames_are_dropped(self):
        assert planning.output_frames(1, 80, False) == 77
        assert planning.output_frames(9, 80, False) == 717

    def test_seconds(self):
        assert planning.output_frames(1) / planning.FPS == 5.0

    def test_rejects_zero_clips(self):
        with pytest.raises(ValueError):
            planning.output_frames(0)


class TestPlan:
    def test_basic_plan(self):
        result = planning.plan((1080, 1920), 4.99, target_area=AREA_480)
        assert (result.out_width, result.out_height) == (832, 448)
        assert result.clips == 1
        assert result.frames == 80
        assert result.video_seconds == 5.0
        assert result.audio_truncated is False
        assert result.preview is False
        assert result.warnings == ()

    def test_preview_truncates_audio(self):
        result = planning.plan((1080, 1920), 12.0, target_area=AREA_480, max_clips=1)
        assert result.clips == 1
        assert result.preview is True
        assert result.audio_truncated is True
        assert "preview" in result.summary()

    def test_small_still_warns(self):
        result = planning.plan((200, 200), 4.0, target_area=AREA_480)
        assert any("never upscales" in warning for warning in result.warnings)

    def test_long_performance_warns(self):
        result = planning.plan((1080, 1920), 240.0, target_area=AREA_480)
        assert any("clips" in warning for warning in result.warnings)

    def test_short_audio_warns(self):
        result = planning.plan((1080, 1920), 0.4, target_area=AREA_480)
        assert any("under a second" in warning for warning in result.warnings)

    def test_infer_frames_must_be_on_the_four_frame_grid(self):
        with pytest.raises(ValueError, match="multiple of 4"):
            planning.plan((1080, 1920), 4.0, infer_frames=81)
        with pytest.raises(ValueError, match="between"):
            planning.plan((1080, 1920), 4.0, infer_frames=1600)

    def test_summary_mentions_resolution_and_clips(self):
        result = planning.plan((1080, 1920), 12.3, target_area=AREA_480)
        text = result.summary()
        assert "832x448" in text
        assert "clips" in text
        assert "video" in text

    def test_as_dict_round_trips(self):
        result = planning.plan((1080, 1920), 4.0, target_area=AREA_480)
        payload = result.as_dict()
        assert payload["clips"] == 1
        assert payload["fps"] == 16
