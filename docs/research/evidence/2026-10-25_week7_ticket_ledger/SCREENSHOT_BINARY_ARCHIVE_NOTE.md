# Screenshot Binary Archive Note — Week 7 Sunday Pricing Evidence

The user supplied **21 FanDuel pricing screenshots** in the conversation used to construct the 2026-10-25 Sunday tickets.

## What is durably committed in GitHub

- exact screenshot filename -> player/market mapping;
- SHA-256 for every original JPEG;
- exact readable FanDuel price transcription;
- every chat-created ticket leg and ticket revision.

See:
- `pricing_screenshot_manifest.csv`
- `pricing_transcription.csv`
- `ticket_legs.csv`

## Raw JPEG binary limitation

The ChatGPT conversation file service exposes these images to the assistant, but the installed GitHub connector's write surface accepts text/blob content and does **not** accept the conversation file reference or local container path. The runtime also has no authenticated network route for a direct git push.

Therefore the repository must **not** claim that raw JPEG files were committed when they were not.

## Lossless archives created from the original conversation attachments

### Batch 1
- images: **15**
- ZIP size: **560,362 bytes**
- ZIP SHA-256: `85d0edd0a57bf5d3c92e6027adc4710f598eccf902ac58e7d30635a19ada3b6a`
- archive name: `WEEK7_PRICING_SCREENSHOTS_BATCH1.zip`

### Batch 2
- images: **6**
- ZIP size: **203,090 bytes**
- ZIP SHA-256: `a747cc1d47cb8a54cb1720c9057b0f765282ccd2076e7b93f12fb4a68e345401`
- archive name: `WEEK7_PRICING_SCREENSHOTS_BATCH2.zip`

Those archives are lossless ZIPs of the exact original JPEG attachment bytes. Any later binary import into GitHub should be verified against the per-image SHA-256s in `pricing_screenshot_manifest.csv` and the archive SHA-256s above.

## Scientific use

The screenshots are downstream sportsbook/ticket-construction evidence only. They are **not football-model inputs** and must never be used upstream to reconstruct a PRISM projection.
