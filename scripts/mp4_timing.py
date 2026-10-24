"""Finalize duration fields for the single-track, constant-rate MP4 recorder."""

from fractions import Fraction
import math
from pathlib import Path
import struct


def _boxes(contents, start, end):
    while start < end:
        if end - start < 8:
            raise ValueError("Truncated MP4 box header")
        size, kind = struct.unpack_from(">I4s", contents, start)
        header_size = 8
        if size == 1:
            if end - start < 16:
                raise ValueError("Truncated extended MP4 box header")
            size = struct.unpack_from(">Q", contents, start + 8)[0]
            header_size = 16
        elif size == 0:
            size = end - start
        if size < header_size or start + size > end:
            raise ValueError("Invalid MP4 box boundary")
        yield kind, start + header_size, start + size
        start += size


def finalize_mp4_timing(path, frame_count, frame_rate):
    """Write real duration without re-encoding or changing fragment timestamps.

    The Isaac Sim encoder can finalize fragmented MP4s with 0xffffffff in
    duration fields. Some players display this sentinel as thousands of hours.
    This recorder has exactly one video track and no audio; all duration fields
    therefore describe frame_count / frame_rate seconds, including outcome cards.
    """
    if frame_count <= 0 or not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("A positive frame count and finite frame rate are required")
    path = Path(path)
    contents = bytearray(path.read_bytes())
    headers = []

    def visit(start, end):
        for kind, payload, limit in _boxes(contents, start, end):
            if kind in (b"moov", b"trak", b"mdia", b"mvex"):
                visit(payload, limit)
            elif kind in (b"mvhd", b"mdhd", b"tkhd", b"mehd"):
                headers.append((kind, payload, limit))

    visit(0, len(contents))
    for kind in (b"mvhd", b"mdhd", b"tkhd"):
        if sum(header[0] == kind for header in headers) != 1:
            raise ValueError("Expected one movie header and one video track")
    movie_header = next(header for header in headers if header[0] == b"mvhd")
    movie_version = contents[movie_header[1]]
    if movie_version not in (0, 1):
        raise ValueError("Unsupported MP4 header version")
    movie_timescale = struct.unpack_from(
        ">I", contents, movie_header[1] + (20 if movie_version else 12)
    )[0]
    duration = Fraction(frame_count, 1) / Fraction(str(frame_rate))
    changes = []
    for kind, payload, limit in headers:
        version = contents[payload]
        if version not in (0, 1):
            raise ValueError("Unsupported MP4 duration version")
        if kind in (b"mvhd", b"mdhd"):
            scale_offset = payload + (20 if version else 12)
            timescale = struct.unpack_from(">I", contents, scale_offset)[0]
            duration_offset = scale_offset + 4
        else:
            timescale = movie_timescale
            duration_offset = payload + (
                (28 if version else 20) if kind == b"tkhd" else 4
            )
        if timescale <= 0:
            raise ValueError("MP4 timescale must be positive")
        duration_format = ">Q" if version else ">I"
        if duration_offset + struct.calcsize(duration_format) > limit:
            raise ValueError("Truncated MP4 duration")
        before = struct.unpack_from(duration_format, contents, duration_offset)[0]
        ticks = round(duration * timescale)
        struct.pack_into(duration_format, contents, duration_offset, ticks)
        changes.append({
            "box": kind.decode(), "timescale": timescale,
            "previous_duration_ticks": before, "duration_ticks": ticks,
        })
    # Equal-size header writes preserve every sample, chunk offset and fragment.
    path.write_bytes(contents)
    return {"duration_seconds": float(duration), "headers": changes}
