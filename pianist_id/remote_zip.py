"""Read single members out of a large remote zip over HTTP range requests.

MAESTRO publishes its audio only as one 108 GB zip, but the project needs about 200
excerpts. A zip is randomly accessible when the server honours Range: the central
directory sits at the end, so a few small requests give the member offsets and each
member can then be streamed on its own. Google Cloud Storage honours Range.

Members are deflated, so there is no seeking inside one; reaching a point means
decompressing everything before it, though stopping early is free. And one member
usually holds several wanted excerpts, since a competition round is one file containing
several works. `read_spans` reads them in one ordered pass and is the function callers
should use.
"""

from __future__ import annotations

import io
import struct
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np

UA = {"User-Agent": "PianistIdentification/0.1"}
RETRIES = 4
CHUNK = 1 << 20


class HttpRangeFile(io.RawIOBase):
    """A seekable read-only file backed by HTTP range requests."""

    def __init__(self, url: str, timeout: int = 120):
        self.url = url
        self.timeout = timeout
        self.pos = 0
        self.n_requests = 0
        self.n_bytes = 0
        with self._get("bytes=0-0") as r:
            cr = r.headers.get("Content-Range")
            if cr is None:
                raise RuntimeError(f"server ignored Range for {url}")
            self.size = int(cr.split("/")[-1])

    def _get(self, rng: str):
        last = None
        for attempt in range(RETRIES):
            try:
                req = urllib.request.Request(self.url, headers=dict(UA, Range=rng))
                return urllib.request.urlopen(req, timeout=self.timeout)
            except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"range {rng} failed after {RETRIES} tries: {last}")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def readinto(self, buf):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        with self._get(f"bytes={self.pos}-{end}") as r:
            data = r.read()
        buf[: len(data)] = data
        self.pos += len(data)
        self.n_requests += 1
        self.n_bytes += len(data)
        return len(data)


@dataclass
class WavFormat:
    channels: int
    sample_rate: int
    bits: int
    data_offset: int
    data_size: int

    @property
    def frame_bytes(self) -> int:
        return self.channels * self.bits // 8

    def byte_at(self, seconds: float) -> int:
        """Byte offset of a time, measured from the start of the member."""
        frame = int(round(seconds * self.sample_rate))
        return self.data_offset + frame * self.frame_bytes


def parse_wav_header(head: bytes) -> WavFormat:
    """Walk the RIFF chunks to find fmt and data.

    Not a fixed 44-byte assumption: real files carry LIST/INFO or fact chunks first,
    and guessing the offset would shift every excerpt.
    """
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE stream")
    pos = 12
    fmt = None
    while pos + 8 <= len(head):
        cid = head[pos:pos + 4]
        size = struct.unpack("<I", head[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"fmt ":
            ch, sr = struct.unpack("<HI", head[body + 2:body + 8])
            bits = struct.unpack("<H", head[body + 14:body + 16])[0]
            fmt = (ch, sr, bits)
        elif cid == b"data":
            if fmt is None:
                raise ValueError("data chunk before fmt chunk")
            return WavFormat(fmt[0], fmt[1], fmt[2], body, size)
        pos = body + size + (size & 1)
    raise ValueError("no data chunk within the inspected header bytes")


def _to_mono_float(raw: bytes, fmt: WavFormat) -> np.ndarray:
    if fmt.bits != 16:
        raise ValueError(f"only 16-bit PCM supported, got {fmt.bits}")
    usable = len(raw) - (len(raw) % fmt.frame_bytes)
    a = np.frombuffer(raw[:usable], dtype="<i2").astype(np.float32) / 32768.0
    if fmt.channels > 1:
        a = a.reshape(-1, fmt.channels).mean(axis=1)
    return a


class RemoteZip:
    """A zip served over HTTP, read without downloading the whole archive."""

    def __init__(self, url: str):
        self.raw = HttpRangeFile(url)
        self.zf = zipfile.ZipFile(io.BufferedReader(self.raw, CHUNK))

    def namelist(self):
        return self.zf.namelist()

    def getinfo(self, name):
        return self.zf.getinfo(name)

    @property
    def stats(self):
        return {"requests": self.raw.n_requests,
                "megabytes": round(self.raw.n_bytes / 1e6, 1)}

    def read_spans(self, member: str, spans):
        """Decode several [start, end] second-spans of one WAV member.

        One ordered forward pass, abandoned after the last span, so cost scales with
        the furthest span rather than the whole member. Yields `(index_into_spans,
        samples, sample_rate)`, samples mono float32. A deflate stream cannot be
        rewound, so a span starting before the read head is skipped rather than
        returning the wrong audio.
        """
        order = sorted(range(len(spans)), key=lambda k: spans[k][0])
        with self.zf.open(member) as fh:
            pending = fh.read(CHUNK)
            fmt = parse_wav_header(pending)
            head = 0  # member byte offset of pending[0]

            def advance(n: int) -> bytes:
                """Consume exactly n bytes from the member, or as many as remain."""
                nonlocal pending, head
                out = bytearray()
                while n > 0:
                    if not pending:
                        pending = fh.read(CHUNK)
                        if not pending:
                            break
                    take = pending[:n]
                    out += take
                    pending = pending[len(take):]
                    head += len(take)
                    n -= len(take)
                return bytes(out)

            limit = fmt.data_offset + fmt.data_size
            for k in order:
                start, end = spans[k]
                b0 = fmt.byte_at(start)
                b1 = min(fmt.byte_at(end), limit)
                if b1 <= b0 or b0 < head:
                    continue
                advance(b0 - head)
                raw = advance(b1 - b0)
                if raw:
                    yield k, _to_mono_float(raw, fmt), fmt.sample_rate
