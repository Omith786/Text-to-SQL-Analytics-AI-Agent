"""Build the bundled sample DuckDB database.

The agent ships with a small, self-contained database so it runs with no network
access and no external accounts. The data here is *synthetic*: it is generated
deterministically to mirror the shape of well-known public datasets, namely
maritime AIS vessel-tracking data and EIA energy statistics. It is not real
observational data and should not be presented as such. The schema and value
ranges are chosen to be realistic enough for meaningful analytical questions.

Running this module writes ``data/sample.duckdb`` (a few MB). The build is
seeded, so repeated runs produce an identical database. The file is not checked
into version control; it is regenerated on demand from this module.
"""

from __future__ import annotations

import datetime as dt
import random
from pathlib import Path

import duckdb

from .config import DEFAULT_DB_PATH

# A fixed seed keeps the bundled database reproducible across builds and machines.
SEED = 20260927

DB_PATH = DEFAULT_DB_PATH

# Reference data kept deliberately small. Coordinates are approximate real-world
# port locations; they are public geographic facts, not proprietary data.
PORTS = [
    ("Rotterdam", "Netherlands", 51.95, 4.14),
    ("Singapore", "Singapore", 1.26, 103.83),
    ("Shanghai", "China", 31.23, 121.47),
    ("Felixstowe", "United Kingdom", 51.96, 1.35),
    ("Hamburg", "Germany", 53.55, 9.99),
    ("Antwerp", "Belgium", 51.26, 4.40),
    ("Los Angeles", "United States", 33.74, -118.27),
    ("Fujairah", "United Arab Emirates", 25.12, 56.34),
]

VESSEL_TYPES = ["Crude Oil Tanker", "LNG Carrier", "Container Ship", "Bulk Carrier", "Chemical Tanker"]
FLAGS = ["Panama", "Liberia", "Marshall Islands", "Singapore", "Malta", "United Kingdom"]

# EIA-style energy products and the regions they are reported for.
ENERGY_PRODUCTS = ["Crude Oil", "Natural Gas", "Diesel", "Gasoline", "Jet Fuel"]
ENERGY_REGIONS = ["US Gulf Coast", "Northwest Europe", "Singapore", "Middle East"]


def _daterange(start: dt.date, days: int):
    for offset in range(days):
        yield start + dt.timedelta(days=offset)


def build(db_path: Path = DB_PATH) -> Path:
    """Generate the sample database at ``db_path`` and return the path.

    Any existing file at the target path is replaced so builds are idempotent.
    """
    rng = random.Random(SEED)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = duckdb.connect(str(db_path))
    try:
        _create_schema(con)
        _load_ports(con)
        vessels = _load_vessels(con, rng, count=60)
        _load_port_calls(con, rng, vessels, calls=900)
        _load_energy(con, rng)
        con.execute("CHECKPOINT")
    finally:
        con.close()
    return db_path


def _create_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE ports (
            port_id      INTEGER PRIMARY KEY,
            name         VARCHAR NOT NULL,
            country      VARCHAR NOT NULL,
            latitude     DOUBLE  NOT NULL,
            longitude    DOUBLE  NOT NULL
        );

        CREATE TABLE vessels (
            mmsi          BIGINT PRIMARY KEY,
            name          VARCHAR NOT NULL,
            vessel_type   VARCHAR NOT NULL,
            flag          VARCHAR NOT NULL,
            length_m      INTEGER NOT NULL,
            gross_tonnage INTEGER NOT NULL,
            built_year    INTEGER NOT NULL
        );

        CREATE TABLE port_calls (
            call_id       INTEGER PRIMARY KEY,
            mmsi          BIGINT  NOT NULL REFERENCES vessels(mmsi),
            port_id       INTEGER NOT NULL REFERENCES ports(port_id),
            arrival_ts    TIMESTAMP NOT NULL,
            departure_ts  TIMESTAMP NOT NULL,
            cargo_tonnes  DOUBLE  NOT NULL
        );

        CREATE TABLE energy_prices (
            price_date  DATE    NOT NULL,
            product     VARCHAR NOT NULL,
            region      VARCHAR NOT NULL,
            price_usd   DOUBLE  NOT NULL,
            PRIMARY KEY (price_date, product, region)
        );
        """
    )


def _load_ports(con: duckdb.DuckDBPyConnection) -> None:
    rows = [(i + 1, name, country, lat, lon) for i, (name, country, lat, lon) in enumerate(PORTS)]
    con.executemany("INSERT INTO ports VALUES (?, ?, ?, ?, ?)", rows)


def _load_vessels(con: duckdb.DuckDBPyConnection, rng: random.Random, count: int) -> list[int]:
    rows = []
    for i in range(count):
        mmsi = 200000000 + rng.randint(0, 99999999)
        vtype = rng.choice(VESSEL_TYPES)
        # Length correlates loosely with type so aggregates look plausible.
        base_len = {"LNG Carrier": 290, "Crude Oil Tanker": 300, "Container Ship": 260,
                    "Bulk Carrier": 225, "Chemical Tanker": 150}[vtype]
        length = base_len + rng.randint(-40, 40)
        gt = int(length * length * rng.uniform(1.6, 2.4))
        rows.append((mmsi, f"MV {vtype.split()[0].upper()}-{i:03d}", vtype,
                     rng.choice(FLAGS), length, gt, rng.randint(1998, 2022)))
    con.executemany("INSERT INTO vessels VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    return [r[0] for r in rows]


def _load_port_calls(con: duckdb.DuckDBPyConnection, rng: random.Random,
                     vessels: list[int], calls: int) -> None:
    start = dt.datetime(2024, 1, 1)
    rows = []
    for call_id in range(1, calls + 1):
        mmsi = rng.choice(vessels)
        port_id = rng.randint(1, len(PORTS))
        arrival = start + dt.timedelta(hours=rng.randint(0, 24 * 365))
        stay_hours = rng.randint(8, 96)
        departure = arrival + dt.timedelta(hours=stay_hours)
        cargo = round(rng.uniform(5000, 320000), 1)
        rows.append((call_id, mmsi, port_id, arrival, departure, cargo))
    con.executemany("INSERT INTO port_calls VALUES (?, ?, ?, ?, ?, ?)", rows)


def _load_energy(con: duckdb.DuckDBPyConnection, rng: random.Random) -> None:
    # Weekly prices across 2024 for each product/region, with a gentle random walk
    # around a product-specific base so trends and comparisons are answerable.
    base = {"Crude Oil": 82.0, "Natural Gas": 11.0, "Diesel": 105.0,
            "Gasoline": 98.0, "Jet Fuel": 110.0}
    rows = []
    for product in ENERGY_PRODUCTS:
        for region in ENERGY_REGIONS:
            price = base[product] * rng.uniform(0.9, 1.1)
            for day in _daterange(dt.date(2024, 1, 1), 364):
                if day.weekday() != 0:  # weekly series, reported on Mondays
                    continue
                price = max(5.0, price + rng.gauss(0, base[product] * 0.03))
                rows.append((day, product, region, round(price, 2)))
    con.executemany("INSERT INTO energy_prices VALUES (?, ?, ?, ?)", rows)


def ensure(db_path: Path = DB_PATH) -> Path:
    """Build the database only if it does not already exist. Returns the path."""
    if not Path(db_path).exists():
        build(Path(db_path))
    return Path(db_path)


if __name__ == "__main__":
    path = build()
    size_kb = path.stat().st_size / 1024
    print(f"Built {path} ({size_kb:.0f} KB)")
