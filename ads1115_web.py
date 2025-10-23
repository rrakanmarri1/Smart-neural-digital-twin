#!/usr/bin/env python3
# FastAPI web service to read ADS1115 (A0 and A1) on Raspberry Pi via I2C
# Endpoints:
#   GET /           -> health
#   GET /read       -> JSON reading for A0 and A1
#
# Query parameters for /read (optional):
#   addr: I2C address (default 0x48). Accepts 0x48 or 72 formats.
#   gain: ADS1115 gain (allowed: 2/3, 1, 2, 4, 8, 16; default=1)
#   dr: data rate SPS (Adafruit backend): one of [8,16,32,64,128,250,475,860]
#   scale0, offset0: convert voltage on A0 -> engineering units: value = voltage*scale0 + offset0
#   scale1, offset1: same for A1
#
# Example:
#   GET /read?addr=0x48&gain=1&scale0=100.0&offset0=0&scale1=1.0&offset1=0

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# Try Adafruit CircuitPython ADS1x15 first
_ADAFRUIT_OK = False
try:
    import board
    import busio
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    _ADAFRUIT_OK = True
except Exception:
    _ADAFRUIT_OK = False

# Fallback to smbus if Adafruit is not available
_SMBUS_OK = False
try:
    import smbus2 as smbus  # prefer smbus2
    _SMBUS_OK = True
except Exception:
    try:
        import smbus  # fall back to smbus
        _SMBUS_OK = True
    except Exception:
        _SMBUS_OK = False


# --------- Utility helpers ---------

def parse_addr(addr: Optional[str | int]) -> int:
    if addr is None:
        return 0x48
    if isinstance(addr, int):
        return addr
    s = addr.strip().lower()
    if s.startswith("0x"):
        return int(s, 16)
    return int(s)

def gain_to_range_volts(gain: float | int) -> float:
    # ADS1115 PGA full-scale ranges
    # gain: 2/3 => ±6.144V; 1 => ±4.096V; 2 => ±2.048V; 4 => ±1.024V; 8 => ±0.512V; 16 => ±0.256V
    if gain == 2/3 or gain == (2/3):
        return 6.144
    if gain == 1:
        return 4.096
    if gain == 2:
        return 2.048
    if gain == 4:
        return 1.024
    if gain == 8:
        return 0.512
    if gain == 16:
        return 0.256
    # default safe
    return 4.096

def clamp_gain(g: float | int) -> float | int:
    allowed = [2/3, 1, 2, 4, 8, 16]
    # handle string freak cases: we only get numeric here from query
    if g in allowed:
        return g
    # attempt to coerce common floats like 0.666666.. to 2/3
    if isinstance(g, float) and abs(g - (2/3)) < 1e-3:
        return 2/3
    return 1

def parse_float(val: Optional[str], default: float) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except Exception:
        return default


# --------- Backends ---------

class ADS1115AdafruitBackend:
    def __init__(self, addr: int, gain: float | int = 1, data_rate: int = 128):
        self.addr = addr
        self.gain = clamp_gain(gain)
        self.dr = data_rate
        # Setup I2C via CircuitPython
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.ads = ADS.ADS1115(self.i2c, address=self.addr)
        # Configure gain and data rate
        self.ads.gain = self.gain
        # Map data rate if valid
        if self.dr in (8, 16, 32, 64, 128, 250, 475, 860):
            self.ads.data_rate = self.dr
        # Create channels for A0 and A1 (single-ended)
        self.chan0 = AnalogIn(self.ads, ADS.P0)
        self.chan1 = AnalogIn(self.ads, ADS.P1)

    def read(self) -> Dict[str, Any]:
        # Voltage and raw values direct from library
        v0 = float(self.chan0.voltage)
        v1 = float(self.chan1.voltage)
        r0 = int(self.chan0.value)  # 16-bit scaled to 0..65535 by library
        r1 = int(self.chan1.value)
        # The library scales raw value to 0..65535; we can provide an approx 15-bit signed raw too
        # But most users want voltage.
        return {
            "A0": {"raw": r0, "voltage": v0},
            "A1": {"raw": r1, "voltage": v1},
            "backend": "adafruit",
            "gain": self.gain,
            "data_rate": getattr(self.ads, "data_rate", self.dr),
        }


class ADS1115SMBusBackend:
    # Registers and bit definitions for ADS1115
    REG_CONVERSION = 0x00
    REG_CONFIG = 0x01

    # MUX bits for single-ended AINx/GND: 100 (A0), 101 (A1)
    MUX_A0 = 0x4000  # 100 << 12
    MUX_A1 = 0x5000  # 101 << 12

    # PGA mapping per gain setting
    PGA_BITS = {
        2/3: 0x0000,  # ±6.144V
        1:   0x0200,  # ±4.096V
        2:   0x0400,  # ±2.048V
        4:   0x0600,  # ±1.024V
        8:   0x0800,  # ±0.512V
        16:  0x0A00,  # ±0.256V
    }

    MODE_SINGLE = 0x0100
    OS_SINGLE_START = 0x8000

    # Data rates (default 128 SPS)
    DR_BITS = {
        8:   0x0000,
        16:  0x0020,
        32:  0x0040,
        64:  0x0060,
        128: 0x0080,
        250: 0x00A0,
        475: 0x00C0,
        860: 0x00E0
    }

    COMP_DISABLE = 0x0003  # disable comparator

    def __init__(self, addr: int, gain: float | int = 1, data_rate: int = 128, bus_id: int = 1):
        self.addr = addr
        self.bus = smbus.SMBus(bus_id)
        self.gain = clamp_gain(gain)
        self.pga_bits = self.PGA_BITS.get(self.gain, self.PGA_BITS[1])
        self.dr_bits = self.DR_BITS.get(data_rate, self.DR_BITS[128])

    def _single_read_channel(self, mux_bits: int) -> int:
        # Build config: start single conversion | MUX | PGA | single-shot | data rate | disable comp
        config = (
            self.OS_SINGLE_START
            | mux_bits
            | self.pga_bits
            | self.MODE_SINGLE
            | self.dr_bits
            | self.COMP_DISABLE
        )
        # Write config register MSB first
        msb = (config >> 8) & 0xFF
        lsb = config & 0xFF
        self.bus.write_i2c_block_data(self.addr, self.REG_CONFIG, [msb, lsb])

        # Wait for conversion completion
        # Conversion time depends on data rate; use a safe small sleep
        time.sleep(1.0 / 128.0 if self.dr_bits == self.DR_BITS[128] else 0.010)

        # Read conversion register (16-bit signed)
        data = self.bus.read_i2c_block_data(self.addr, self.REG_CONVERSION, 2)
        raw = (data[0] << 8) | data[1]
        # Convert to signed 16-bit
        if raw > 0x7FFF:
            raw = raw - 0x10000
        return raw

    def read(self) -> Dict[str, Any]:
        raw0 = self._single_read_channel(self.MUX_A0)
        raw1 = self._single_read_channel(self.MUX_A1)
        fs = gain_to_range_volts(self.gain)
        lsb = fs / 32768.0  # volts per bit for ADS1115
        v0 = raw0 * lsb
        v1 = raw1 * lsb
        return {
            "A0": {"raw": int(raw0), "voltage": float(v0)},
            "A1": {"raw": int(raw1), "voltage": float(v1)},
            "backend": "smbus",
            "gain": self.gain,
            "data_rate": next((k for k, v in self.DR_BITS.items() if v == self.dr_bits), 128),
        }


# --------- FastAPI app ---------

app = FastAPI(title="ADS1115 Reader", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "ADS1115 reader running",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "adafruit_backend": _ADAFRUIT_OK,
        "smbus_backend": _SMBUS_OK,
        "usage": "/read?addr=0x48&gain=1&scale0=1&offset0=0&scale1=1&offset1=0",
    }


@app.get("/read")
def read_ads(
    addr: Optional[str] = Query(default="0x48", description="I2C address (e.g., 0x48, 0x49)"),
    gain: float | int = Query(default=1, description="Gain: one of [2/3,1,2,4,8,16]"),
    dr: int = Query(default=128, description="Data rate (SPS). Adafruit backend only: [8,16,32,64,128,250,475,860]"),
    scale0: Optional[str] = Query(default=None, description="Scale factor for A0 engineering units"),
    offset0: Optional[str] = Query(default=None, description="Offset for A0 engineering units"),
    scale1: Optional[str] = Query(default=None, description="Scale factor for A1 engineering units"),
    offset1: Optional[str] = Query(default=None, description="Offset for A1 engineering units"),
) -> Dict[str, Any]:
    """
    Read A0 and A1 from ADS1115 and return JSON with raw and voltage.
    Optionally apply y = scale*voltage + offset to produce 'eng_value'.
    """
    i2c_addr = parse_addr(addr)
    g = clamp_gain(gain)
    s0 = parse_float(scale0, 1.0)
    o0 = parse_float(offset0, 0.0)
    s1 = parse_float(scale1, 1.0)
    o1 = parse_float(offset1, 0.0)

    # Choose backend
    backend_name = None
    try:
        if _ADAFRUIT_OK:
            backend = ADS1115AdafruitBackend(i2c_addr, gain=g, data_rate=dr)
            backend_name = "adafruit"
        elif _SMBUS_OK:
            backend = ADS1115SMBusBackend(i2c_addr, gain=g, data_rate=dr)
            backend_name = "smbus"
        else:
            return {
                "error": "No ADS1115 backend available. Install 'adafruit-circuitpython-ads1x15' or 'smbus2'.",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

        data = backend.read()
        v0 = float(data["A0"]["voltage"])
        v1 = float(data["A1"]["voltage"])
        eng0 = v0 * s0 + o0
        eng1 = v1 * s1 + o1

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "address": hex(i2c_addr),
            "backend": backend_name,
            "gain": data.get("gain", g),
            "data_rate": data.get("data_rate", dr),
            "channels": {
                "A0": {
                    "raw": data["A0"]["raw"],
                    "voltage": round(v0, 6),
                    "eng_value": round(eng0, 6),
                    "scale": s0,
                    "offset": o0,
                },
                "A1": {
                    "raw": data["A1"]["raw"],
                    "voltage": round(v1, 6),
                    "eng_value": round(eng1, 6),
                    "scale": s1,
                    "offset": o1,
                },
            },
        }
    except Exception as e:
        return {
            "error": str(e),
            "address": hex(i2c_addr),
            "backend": backend_name or "none",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
