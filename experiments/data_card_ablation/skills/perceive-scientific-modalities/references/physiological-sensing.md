# Physiological and environmental sensing

Use these checks for wearable channels, physiological measurements,
environmental sensors, per-subject files, sessions, events, and survey tables.

## Observation units and time

- Distinguish subject, session, sample, event interval, and survey-response
  objects. Do not collapse them into one table merely because they share an ID.
- Identify absolute versus relative time, timestamp units, timezone when stated,
  nominal and observed sampling cadence, irregular gaps, and duplicated times.
- Treat a sequence index as elapsed time only when its unit or cadence is
  established. Check whether it resets by subject or session.
- Check whether channels are truly synchronized. Repeated values can indicate a
  lower-rate channel aligned onto a higher-rate timeline rather than independent
  measurements at every row.

## Channels and metadata

- Map each channel name to a grounded measurement meaning and unit. Treat common
  names such as EDA, BVP, heart rate, skin temperature, pressure, humidity, or
  particle count as clues; confirm their meaning and units from available
  metadata or value evidence.
- Distinguish raw sensor signals, device-derived summaries, environmental
  readings, quality indicators, subject metadata, and human-reported events.
- Identify subject and session encodings, per-subject file partitions, and any
  repeated or missing participant records. Treat numeric-looking participant
  identifiers as categorical keys when they identify people rather than
  quantities.

## Cross-object relationships

- State how sensor samples align to sessions, how event intervals align to a
  timeline, and how survey rows join to participants or dates.
- Preserve ordinal responses, binary flags, and their missing codes as distinct
  encodings; do not infer their meanings from numeric order alone.
- Record overlap, gaps, inconsistent identifiers, and many-to-one relationships
  when observed.
- Keep a reported condition or event label separate from physiological evidence;
  do not infer clinical state, causality, or diagnostic meaning from a signal.

Describe sensing semantics without recommending filtering, resampling, feature
engineering, classification, or clinical interpretation.
