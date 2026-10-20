# Belt Data Requirements

*What to collect for the Belt RUL Predictor and why it matters.*

The predictor accepts **any CSV layout**. Your column names, units and column order don't need to match ours. On upload, an AI assistant suggests how your columns match the model's fields, and your engineer reviews and approves it. What *does* matter is **what you measure**, and above all **recording when belts actually fail or are replaced**.

---

## 1. The single most important thing: record failures and replacements

To predict *remaining* life, the model must learn from belts whose *end* of life is known.

For every belt that fails, is replaced, or is taken out of service, record:

| Field | Example | Notes |
|---|---|---|
| Belt / asset ID | `CV-03-BELT-B` | Same ID used in the inspection log |
| Operating hour at failure / replacement | `11 840` | Or the date, if you log hour-meter readings by date |
| Reason | `Longitudinal rip`, `Wear limit`, `Planned replacement` | Free text is fine |

**What happens without it:** the system can still train, but it has to *estimate* each belt's end of life. It does this by extending the wear, elongation and crack trends up to assumed failure limits (wear 100%, elongation 4%, crack 50 mm). Those estimates are much less reliable.

- On our test data, the estimated end-of-life was **about 20% later than the true one**, so predicted remaining life was **optimistic**.
- Real belts degrade less regularly than our test data, so expect a larger error on real data.
- The dashboard clearly labels any model trained on estimated life.

> **Request:** please start recording failure/replacement hours now, including for belts already in service. Every recorded failure directly improves accuracy.

---

## 2. Inspection readings (one row per belt per inspection)

Ranked by value to the model. Units are only a preference: any unit works as long as it is consistent and stated in the column name or header (e.g. `Temp (°F)`).

### Essential
| Measurement | Preferred unit | Why |
|---|---|---|
| Belt / asset ID | — | Connects readings of the same belt over time |
| Operating hours (hour-meter) | h | Age of the belt. A timestamp works if no hour-meter exists, but idle time then counts as running time |
| At least one degradation measure: | | Tracks how close the belt is to failure |
| • Cover wear / remaining cover thickness | % of allowable wear, or mm | |
| • Permanent elongation | % of original length | |
| • Longest crack / tear length | mm | |

### Strongly recommended (operating conditions)
Load (% of rated capacity), belt speed (m/s), temperature (°C), vibration (mm/s RMS), drive torque (N·m) or motor current (A), belt tension (N), humidity (%), number of maintenance actions so far.

### Useful (recorded once per belt)
Rubber hardness (Shore A), belt thickness (mm), top-cover thickness (mm), tensile rating (kN/m or N/mm), width (mm), length (m or mm), pulley misalignment (mm).

**Other sensors you already have** (e.g. motor current, idler temperatures) are welcome. The mapping step can keep them as extra model inputs.

---

## 3. Frequency and format

- **Inspection interval:** at least every 250 operating hours, more often near end of life. At least **3 inspections per belt** are needed before its trend can be used.
- **Format:** CSV (export from Excel, SCADA or CMMS is fine). One header row. One row per belt per inspection.
- **Missing values:** leave the cell empty. Don't write `0`, because `0` is read as a real measurement.
- **Don't** reuse a belt ID for the replacement belt. Give the new belt a new ID.

### Example

```csv
Asset,Date,Hour Meter,Load %,Speed m/s,Temp F,Vib mm/s,Wear %,Elongation %,Crack mm,Failure Hour
CV3-B,2025-01-04,1200,72,3.1,88,3.9,8.5,0.31,1.2,
CV3-B,2025-02-01,1450,75,3.1,91,4.2,10.1,0.38,1.9,
CV3-B,2025-11-20,11840,78,3.0,95,7.8,96.0,3.6,41.0,11840
```

---

## 4. Privacy note

During column mapping, the column names, basic statistics (min/max/mean) and **up to 5 sample values per column** are sent to Google Gemini to suggest the mapping. Full datasets are never sent, and training runs locally. If this is not acceptable, the system can run without an AI key: it falls back to name matching, and the engineer maps the columns by hand.
