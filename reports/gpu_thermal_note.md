# Thermal logging, closed as a documented failure

Dated 12 September 2026.

Sustained thermal capture for the classifier matrix was attempted and failed
silently. A sampling loop was launched on 24 August 2026 at 02:38:09, writing a
timestamp, temperature, graphics clock and utilisation once every 60 seconds to
`reports/gpu_thermal.csv`. The output file was recreated shortly after launch,
so the loop's file descriptor was left pointing at an unlinked inode while a
new file of the same name was written on disk. Samples continued to be produced
for nineteen days and none of them reached any file that can be read.

The failure was not detected while it was occurring, no error was raised, and
the loop was still running when it was found on 12 September 2026. It has since
been terminated.

## The record that exists

Six lines of `reports/gpu_thermal.csv`, one header and five samples, spanning
2026-08-24T02:38:10 to 2026-08-24T02:42:10:

| Timestamp | Temperature, C | Clock, MHz | Utilisation, per cent |
| --- | --- | --- | --- |
| 2026-08-24T02:38:10+01:00 | 82 | 1620 | 77 |
| 2026-08-24T02:39:10+01:00 | 82 | 1607 | 82 |
| 2026-08-24T02:40:10+01:00 | 82 | 1721 | 69 |
| 2026-08-24T02:41:10+01:00 | 82 | 1657 | 100 |
| 2026-08-24T02:42:10+01:00 | 82 | 1632 | 97 |

These five samples are the sole thermal record for this chapter.

## What follows for the chapter

No re-collection is possible. The runs the record was wanted for are complete
and evaluation is closed by the ruling of 12 September 2026, so no future run
can supply the missing temperature and clock history.

The cost material therefore makes no thermal claim beyond these six lines.
Timing conditions are stated without a temperature and clock record, and the
absence is reported as this documented failure rather than as a gap to be
filled later.
