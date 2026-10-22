# Belt Data Requirements

*What to record so the Belt RUL system can predict the remaining life of your power-transmission belts (classical wrapped, wedge, raw-edge cogged, Poly-V …).*

You can record data in the app, through the **Fleet** pages, or upload it as a CSV in any column layout. On upload, the columns are matched to the system's fields with AI help and an engineer approves the match, so your existing spreadsheets work as they are. What matters is **what is measured**, and above all **when belts actually fail or are replaced**.

---

## 1. Most important: record every failure and replacement

The model learns remaining life from belts whose **end of life is known**. For every belt that fails, is replaced, or is taken out of service, record:

| Field | Example | Notes |
|---|---|---|
| Belt / drive ID | `CT-PUMP-07-B1` | The same ID used for its inspections |
| Operating hours at failure or replacement | `11 840` | From the hour meter, or computed from run time |
| Outcome and reason | `Failed – rib cracking`, `Planned replacement` | Planned replacements matter too |

In the app, this is **End of life → Record** on the belt's page.

**Belts without a recorded end** (still running, or replaced before failing) are still useful: they are treated as *censored* — the model learns they lasted at least until their last inspection — instead of guessing a failure time. At least a handful of belts with a recorded failure are needed.

> Please start recording failures and replacements now, including for belts already in service. Each recorded failure directly improves accuracy.

## 2. Confirm the failure limits for each belt type

Each belt type (for example Poly-F Plus, SpacesaVer SPB, or BX cogged) has its own replacement limits:
- **Elongation:** the % stretch of pitch length at which the drive runs out of take-up or the belt must be replaced.
- **Sidewall wear:** 100% = the allowable wear, for example the belt riding down in the groove to its limit.
- **Crack depth:** in mm, the depth at which a crack means the belt must be replaced.

The values in the app at installation are **placeholders**. Please confirm them from JK Fenner data or your site standards.

---

## 3. Belt and drive details (once per belt)

| Field | Unit | Why it matters |
|---|---|---|
| Belt type / section | e.g. `B`, `SPB`, `BX`, `Poly-F Plus` | Selects the failure limits; cogged and wedge belts behave differently |
| Pitch (datum) length | mm | Sets how often the belt flexes |
| Driver and driven pulley diameters | mm | Small pulleys raise bending stress sharply |
| Driver speed | rpm | Belt speed and flex frequency |
| Number of pulleys / idlers | count | Flexes per revolution |
| Centre distance, rated power | mm, kW | Context for load and tension |

The system computes belt speed, bending frequency and speed ratio from these values.

## 4. Inspection readings (one row per belt per inspection)

| Priority | Measurement | Unit |
|---|---|---|
| **Essential** | Operating hours since installation | h |
| **Essential** | At least one of: elongation (% of pitch length), sidewall wear (% of allowable), deepest crack | %, %, mm |
| Recommended | Load (% of rated power), belt surface temperature, slip, vibration | %, °C, %, mm/s RMS |
| Recommended | Static belt tension, number of re-tensions so far | N, count |

Other sensors you already have (motor current, bearing temperature …) are welcome. The mapping step can keep them as extra inputs.

## 5. Frequency and format

- **Inspect at least every 500 operating hours**, and more often as a belt approaches its limits. A belt needs at least **3 inspections** before its trend can be used.
- **Missing values:** leave the cell empty. Never write `0`, because `0` is read as a real measurement.
- **New belt, new ID:** don't reuse the old belt's ID for its replacement.
- **CSV uploads:** one header row and one row per belt per inspection. Exports from Excel, SCADA or a maintenance system are fine.
- **Bulk import in the app:** on the Fleet page, **Import from CSV** loads many belts or inspections at once (download the template there). The file is checked first and nothing is saved until it is error-free. Use plain numbers (`1800`, not `1,800`), a dot for decimals, dates like `2025-01-31`, and the units shown in the template.

```csv
Drive,Section,Run Hrs,Small Pulley mm,Big Pulley mm,Belt Len mm,RPM,Load %,Temp C,Slip %,Stretch %,Side wear %,Crack mm,Failure Hour
CT-PUMP-07-B1,SPB,500,112,250,1800,1440,82,48,0.9,0.35,4,,
CT-PUMP-07-B1,SPB,3000,112,250,1800,1440,85,51,1.1,0.70,20,0.1,
CT-PUMP-07-B1,SPB,7400,112,250,1800,1440,88,56,1.6,1.45,52,1.5,7400
```

---

## 6. Privacy

During column mapping, Google Gemini receives the column names, basic statistics and **up to 5 sample values per column**. Full datasets are never sent, and training and predictions run on the hosted server. If this is not acceptable, the AI step can be switched off and an engineer maps the columns by hand.
