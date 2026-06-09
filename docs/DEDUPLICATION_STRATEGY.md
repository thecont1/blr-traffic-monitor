# Traffic Data Deduplication Strategy (Cycle-Aware)

## Problem

The workflows run via cron-job.org at 3am daily (dedup) and twice/hour (snapshot). Late triggers can still occur due to CI congestion. When runs cross the clock-hour boundary, naive "same-hour" deduplication can keep both points even though they belong to the same intended sampling window.

## Goal

Keep **exactly one reading per intended hourly collection cycle per route**, while still preserving redundancy when one run fails.

## Core Idea: Cycle Bucketing Instead of Clock-Hour Bucketing

Define one collection cycle as:

- Start: `HH:10`
- End: `(HH+1):09`

So each cycle still has the two intended slots:

- Offset `0` minutes → `HH:10` (primary)
- Offset `30` minutes → `HH:40` (backup)

This design correctly groups delayed `:40` executions that show up just after the next hour.

## Selection Algorithm

When multiple readings exist in the same `(cycle_start_hour, route_code)` bucket, keep one
using this priority:

1. **Nearest to intended offsets** (`0` or `30` minutes from cycle start)
2. **Earlier timestamp wins ties** (first sample in the cycle)

Score used in code: `(distance_to_target, timestamp)` (lower is better).

## Implementation

### Script: `tools/fix_timestamps.py`

**Usage:**

```bash
# Preview changes (dry-run)
python tools/fix_timestamps.py

# Apply changes
python tools/fix_timestamps.py --apply
```

### Key Functions

1. **`get_cycle_start(ts)`**
   - Maps a timestamp to cycle start at `HH:10`
   - If minute `< 10`, shifts to previous hour's cycle

2. **`cycle_distance_to_target(ts, cycle_start)`**
   - Computes distance (in minutes) to nearest target offset `[0, 30]`

3. **`select_best_reading(entries)`**
   - Chooses best reading in a cycle bucket using the score above

4. **`deduplicate_same_cycle(rows)`**
   - Deduplicates full CSV by `(cycle_date, cycle_hour, route_code)`
   - Keeps one best row per cycle bucket

## Edge Cases

### Case 1: Exact Duplicate Timestamp

```
2026-02-04,00:10,ROUTE_A,25,10.0
2026-02-04,00:10,ROUTE_A,25,10.0
```

**Action**: Keep first occurrence only.

### Case 2: Two Different Times in Same Cycle

```
2026-02-04,00:11,ROUTE_A,25,10.0
2026-02-04,00:43,ROUTE_A,26,10.1
```

**Action**: Keep whichever is closer to offsets `0` or `30`; tie → earlier timestamp.

### Case 3: Delayed Backup Run Crossing Hour

```
2026-02-04,00:10,ROUTE_A,25,10.0  # primary slot
2026-02-04,01:02,ROUTE_A,26,10.1  # delayed backup slot
```

Both are in the same cycle (`00:10` to `01:09`).

**Action**: Keep one best reading (not both), preventing semantic duplication.

## GitHub Actions Integration

Workflow triggered exclusively by cron-job.org via Cloudflare worker:

Two cron jobs configured at cron-job.org:
- "traffiCOracle BLR Dedup" - runs at 3am daily, hits `https://worker.dev?type=dedup`
- "traffiCOracle BLR Snapshot" - runs twice/hour, hits `https://worker.dev?type=snapshot`

Worker validates `X-Cron-Secret` header and dispatches the workflow with `inputs.job_type`.

```yaml
on:
  workflow_dispatch:
    inputs:
      job_type:
        description: "Which job to run (snapshot or dedup)"
        default: "snapshot"
        type: choice
        options:
          - snapshot
          - dedup
```

## Monitoring Output

The script logs:

- Total rows before/after
- Rows removed
- Number of same-cycle collisions
- Up to 5 examples of cycle conflict resolution

Example format:

```
cycle 2026-02-04 00:10 ROUTE_A -> kept 00:10 from ['00:10', '00:43']
```

## Tuning if schedule changes

If collection times change, update these constants in `tools/fix_timestamps.py`:

```python
CYCLE_START_MINUTE = 10
TARGET_OFFSETS = [0, 30]
```

For example, for `:15` and `:45`, use `CYCLE_START_MINUTE = 15` and keep `[0, 30]`.

## Conclusion

Cycle-aware deduplication now ensures:

- ✅ One retained point per intended hourly cycle per route
- ✅ Correct handling of delayed backups crossing clock-hour boundaries
- ✅ Deterministic, explainable keep/discard logic
- ✅ Controlled long-term data growth without removing redundancy intent