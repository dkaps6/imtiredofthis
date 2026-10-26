# Week 7 Sunday + SNF Ticket Creation Ledger — 2026-10-25

Status: **CHAT-CREATED PROSPECTIVE EVIDENCE PRESERVATION**

This file preserves the ticket constructions created in the GPT/user conversation for the 2026 Week-7 Sunday slate and KC-SEA Sunday Night Football.

## Evidence rules

- This is a record of what was actually constructed in chat, including superseded drafts.
- It is **not** a postgame-optimized reconstruction.
- `INITIAL_DRAFT` Sunday non-RB cards are preserved because they materially differed from the later price-informed final cards.
- `FINAL` means the last card version presented in chat for that portfolio; it does **not** imply the user necessarily placed every card.
- ATD cards are preserved as an explicitly **uncertified / fun-money** lane and must not be used as validation of PRISM yardage science.
- The SNF ATD card labelled `SEA_SCRIPT_LABEL_AS_CREATED` is preserved exactly as created even though Kenneth Walker III was a KC player in the 2026 slate. Do not silently repair historical ticket evidence.
- User-supplied FanDuel screenshot **content is preserved in this evidence package as an exact pricing transcription plus per-image SHA-256 manifest**. The current GitHub connector cannot accept raw JPEG bytes from ChatGPT conversation attachments, so the 21 JPEG binaries are not falsely claimed as committed. Two lossless ZIP archives of the originals were created in the chat workspace and are identified in `SCREENSHOT_BINARY_ARCHIVE_NOTE.md` for external handoff.
- `pricing_transcription.csv` contains only prices that were clearly readable/transcribed in the conversation. Cropped/uncertain prices are intentionally omitted rather than guessed.
- The canonical paid Week-7 market boundary remains `data/market_track_record/`; these screenshots are downstream human ticket-construction evidence, not model inputs.

## Sunday non-RB — initial draft before final pricing refinement

### SAFE — `SUN-NRB-D1-SAFE`

1. **Aaron Rodgers** — pass_yards 200+ (-245)
2. **Brock Purdy** — pass_yards 200+ (-300)
3. **Tre Tucker** — rec_yards 30+ (-220)

### MODERATE — `SUN-NRB-D1-MOD`

1. **Geno Smith** — pass_yards 200+ (-215)
2. **DK Metcalf** — rec_yards 40+ (-310)
3. **Greg Dulcich** — rec_yards 20+ (-136)
4. **George Kittle** — rec_yards 30+

### MEDIUM — `SUN-NRB-D1-MED`

1. **J.J. McCarthy** — pass_yards 175+ (-240)
2. **Aaron Rodgers** — pass_yards 225+ (-132)
3. **Tre Tucker** — rec_yards 40+ (-122)
4. **Juwan Johnson** — rec_yards 25+ (-210)
5. **Cooper Kupp** — rec_yards 20+ (-168)

### HIGH — `SUN-NRB-D1-HIGH`

1. **Brock Purdy** — pass_yards 225+ (-166)
2. **Geno Smith** — pass_yards 200+ (-215)
3. **DK Metcalf** — rec_yards 50+ (-180)
4. **Nico Collins** — rec_yards 40+ (-470)
5. **Greg Dulcich** — rec_yards 25+
6. **George Kittle** — rec_yards 40+ (-500)
7. **AJ Barner** — rec_yards 20+ (-180)

### LOTTERY — `SUN-NRB-D1-LOT`

1. **J.J. McCarthy** — pass_yards 200+ (-130)
2. **Kyler Murray** — pass_yards 200+ (-136)
3. **Tyler Shough** — pass_yards 200+ (-245)
4. **Nico Collins** — rec_yards 40+ (-470)
5. **Jaxon Smith-Njigba** — rec_yards 60+ (-450)
6. **Juwan Johnson** — rec_yards 30+ (-144)
7. **Cooper Kupp** — rec_yards 20+ (-168)
8. **AJ Barner** — rec_yards 20+ (-180)
9. **Puka Nacua** — rec_yards 50+


## Sunday non-RB — final price-informed portfolio

### SAFE — `SUN-NRB-F-SAFE`

1. **Brock Purdy** — pass_yards 200+ (-300)
2. **Aaron Rodgers** — pass_yards 200+ (-245)
3. **Tre Tucker** — rec_yards 30+ (-220)
4. **Greg Dulcich** — rec_yards 20+ (-136)

### MODERATE — `SUN-NRB-F-MOD`

1. **Geno Smith** — pass_yards 200+ (-215)
2. **DK Metcalf** — rec_yards 40+ (-310)
3. **Kyler Murray** — pass_yards 200+ (-136)
4. **Greg Dulcich** — rec_yards 20+ (-136)
5. **Cooper Kupp** — rec_yards 20+ (-168)

### MEDIUM — `SUN-NRB-F-MED`

1. **Brock Purdy** — pass_yards 225+ (-166)
2. **J.J. McCarthy** — pass_yards 175+ (-240)
3. **Tre Tucker** — rec_yards 40+ (-122)
4. **Juwan Johnson** — rec_yards 25+ (-210)
5. **AJ Barner** — rec_yards 20+ (-180)
6. **Cooper Kupp** — rec_yards 20+ (-168)

### HIGH — `SUN-NRB-F-HIGH`

1. **Brock Purdy** — pass_yards 225+ (-166)
2. **Aaron Rodgers** — pass_yards 225+ (-132)
3. **Kyler Murray** — pass_yards 200+ (-136)
4. **Tyler Shough** — pass_yards 225+ (-125)
5. **Tre Tucker** — rec_yards 40+ (-122)
6. **Greg Dulcich** — rec_yards 20+ (-136)
7. **Nico Collins** — rec_yards 60+ (-166)
8. **Jaxon Smith-Njigba** — rec_yards 80+ (-192)

### LOTTERY — `SUN-NRB-F-LOT`

1. **Brock Purdy** — pass_yards 250+ (+108)
2. **Aaron Rodgers** — pass_yards 250+ (+140)
3. **Kyler Murray** — pass_yards 225+ (+140)
4. **Tyler Shough** — pass_yards 225+ (-125)
5. **J.J. McCarthy** — pass_yards 200+ (-130)
6. **Tre Tucker** — rec_yards 40+ (-122)
7. **Juwan Johnson** — rec_yards 30+ (-144)
8. **Greg Dulcich** — rec_yards 20+ (-136)
9. **AJ Barner** — rec_yards 20+ (-180)
10. **Nico Collins** — rec_yards 60+ (-166)
11. **Jaxon Smith-Njigba** — rec_yards 80+ (-192)


## Sunday RB-integrated — final portfolio

### SAFE — `SUN-RB-F-SAFE`

1. **Brock Purdy** — pass_yards 200+ (-300)
2. **Aaron Rodgers** — pass_yards 200+ (-245)
3. **Chase Brown** — rush_yards 30+ (-250)
4. **Tre Tucker** — rec_yards 30+ (-220)

### MODERATE — `SUN-RB-F-MOD`

1. **Geno Smith** — pass_yards 200+ (-215)
2. **DK Metcalf** — rec_yards 40+ (-310)
3. **Jonathan Taylor** — rush_yards 50+ (-340)
4. **Juwan Johnson** — rec_yards 25+ (-210)
5. **Jaxon Smith-Njigba** — rec_yards 70+ (-290)

### MEDIUM — `SUN-RB-F-MED`

1. **Brock Purdy** — pass_yards 225+ (-166)
2. **J.J. McCarthy** — pass_yards 175+ (-240)
3. **Nico Collins** — rec_yards 50+ (-270)
4. **Jaylen Warren** — rush_yards 50+ (-270)
5. **Chase Brown** — rush_yards 40+ (-122)
6. **Christian McCaffrey** — rec_yards 25+ (-400)

### HIGH — `SUN-RB-F-HIGH`

1. **Aaron Rodgers** — pass_yards 225+ (-132)
2. **Geno Smith** — pass_yards 225+ (-112)
3. **J.J. McCarthy** — pass_yards 200+ (-130)
4. **Tre Tucker** — rec_yards 40+ (-122)
5. **Juwan Johnson** — rec_yards 30+ (-144)
6. **Jonathan Taylor** — rush_yards 60+ (-200)
7. **Jaxon Smith-Njigba** — rec_yards 80+ (-192)
8. **Nico Collins** — rec_yards 60+ (-166)

### LOTTERY — `SUN-RB-F-LOT`

1. **Brock Purdy** — pass_yards 250+ (+108)
2. **Aaron Rodgers** — pass_yards 250+ (+140)
3. **Geno Smith** — pass_yards 225+ (-112)
4. **J.J. McCarthy** — pass_yards 200+ (-130)
5. **DK Metcalf** — rec_yards 50+ (-180)
6. **Juwan Johnson** — rec_yards 30+ (-144)
7. **Chase Brown** — rush_yards 50+ (+152)
8. **Jonathan Taylor** — rush_yards 60+ (-200)
9. **Jaylen Warren** — rush_yards 50+ (-270)
10. **Jaxon Smith-Njigba** — rec_yards 80+ (-192)
11. **Nico Collins** — rec_yards 60+ (-166)


## Sunday ATD — fun / uncertified lane

### CHALK — `SUN-ATD-CHALK`

1. **Jonathan Taylor** — anytime_td YES
2. **Christian McCaffrey** — anytime_td YES
3. **Derrick Henry** — anytime_td YES

### BALANCED — `SUN-ATD-BAL`

1. **Christian McCaffrey** — anytime_td YES
2. **George Kittle** — anytime_td YES
3. **Jaxon Smith-Njigba** — anytime_td YES
4. **Jonathan Taylor** — anytime_td YES

### AGGRESSIVE — `SUN-ATD-AGG`

1. **George Kittle** — anytime_td YES
2. **Amon-Ra St. Brown** — anytime_td YES
3. **Ja'Marr Chase** — anytime_td YES
4. **Nico Collins** — anytime_td YES

### LOTTERY — `SUN-ATD-LOT`

1. **George Kittle** — anytime_td YES
2. **Jaxon Smith-Njigba** — anytime_td YES
3. **Amon-Ra St. Brown** — anytime_td YES
4. **Ja'Marr Chase** — anytime_td YES
5. **Trey McBride** — anytime_td YES
6. **Brock Bowers** — anytime_td YES

### RB_HAMMER — `SUN-ATD-RBH`

1. **Jahmyr Gibbs** — anytime_td YES
2. **Derrick Henry** — anytime_td YES
3. **Christian McCaffrey** — anytime_td YES
4. **Jonathan Taylor** — anytime_td YES


## KC-SEA SNF ATD — detour / fun lane

### CORE — `SNF-ATD-CORE`

1. **Kenneth Walker III** — anytime_td YES
2. **Jaxon Smith-Njigba** — anytime_td YES

### BALANCED — `SNF-ATD-BAL`

1. **Kenneth Walker III** — anytime_td YES
2. **Jaxon Smith-Njigba** — anytime_td YES
3. **Travis Kelce** — anytime_td YES

### AGGRESSIVE — `SNF-ATD-AGG`

1. **Kenneth Walker III** — anytime_td YES
2. **Jaxon Smith-Njigba** — anytime_td YES
3. **Travis Kelce** — anytime_td YES
4. **AJ Barner** — anytime_td YES

### LOTTERY — `SNF-ATD-LOT`

1. **Kenneth Walker III** — anytime_td YES
2. **Jaxon Smith-Njigba** — anytime_td YES
3. **Travis Kelce** — anytime_td YES
4. **AJ Barner** — anytime_td YES
5. **Patrick Mahomes** — anytime_td YES

### SEA_SCRIPT_LABEL_AS_CREATED — `SNF-ATD-SEA-SCRIPT`

1. **Kenneth Walker III** — anytime_td YES
2. **Jaxon Smith-Njigba** — anytime_td YES
3. **AJ Barner** — anytime_td YES


## KC-SEA SNF regular alt-line portfolio

### SAFE — `SNF-REG-SAFE`

1. **Patrick Mahomes** — pass_yards 175+
2. **Sam Darnold** — pass_yards 175+
3. **Jaxon Smith-Njigba** — rec_yards 60+
4. **AJ Barner** — rec_yards 15+

### MODERATE — `SNF-REG-MOD`

1. **Patrick Mahomes** — pass_yards 200+
2. **Sam Darnold** — pass_yards 200+
3. **Jaxon Smith-Njigba** — rec_yards 70+
4. **Travis Kelce** — rec_yards 25+
5. **Cooper Kupp** — rec_yards 15+

### MEDIUM — `SNF-REG-MED`

1. **Patrick Mahomes** — pass_yards 200+
2. **Sam Darnold** — pass_yards 200+
3. **Jaxon Smith-Njigba** — rec_yards 80+
4. **Travis Kelce** — rec_yards 30+
5. **AJ Barner** — rec_yards 20+
6. **Kenneth Walker III** — rush_yards 50+

### HIGH — `SNF-REG-HIGH`

1. **Patrick Mahomes** — pass_yards 200+
2. **Sam Darnold** — pass_yards 200+
3. **Jaxon Smith-Njigba** — rec_yards 80+
4. **Travis Kelce** — rec_yards 40+
5. **AJ Barner** — rec_yards 20+
6. **Cooper Kupp** — rec_yards 20+
7. **Kenneth Walker III** — rush_yards 60+

### LOTTERY — `SNF-REG-LOT`

1. **Patrick Mahomes** — pass_yards 225+
2. **Sam Darnold** — pass_yards 225+
3. **Jaxon Smith-Njigba** — rec_yards 90+
4. **Travis Kelce** — rec_yards 40+
5. **AJ Barner** — rec_yards 25+
6. **Cooper Kupp** — rec_yards 25+
7. **Kenneth Walker III** — rush_yards 70+



## Pricing screenshot mapping

| Screenshot | Player | Market | Batch |
|---|---|---|---|
| IMG_4448.jpeg | Aaron Rodgers | pass_yards | batch_1 |
| IMG_4449.jpeg | Brock Purdy | pass_yards | batch_1 |
| IMG_4451.jpeg | Christian McCaffrey | rec_yards | batch_1 |
| IMG_4452.jpeg | Tre Tucker | rec_yards | batch_1 |
| IMG_4450(1).jpeg | Jaylen Warren | rush_yards | batch_1 |
| IMG_4453.jpeg | Geno Smith | pass_yards | batch_1 |
| IMG_4454.jpeg | DK Metcalf | rec_yards | batch_1 |
| IMG_4455.jpeg | Juwan Johnson | rec_yards | batch_1 |
| IMG_4456.jpeg | Chase Brown | rush_yards | batch_1 |
| IMG_4457.jpeg | Jonathan Taylor | rush_yards | batch_1 |
| IMG_4458.jpeg | J.J. McCarthy | pass_yards | batch_1 |
| IMG_4459.jpeg | Woody Marks | rush_yards | batch_1 |
| IMG_4460.jpeg | Ollie Gordon II | rush_yards | batch_1 |
| IMG_4461.jpeg | Jaxon Smith-Njigba | rec_yards | batch_1 |
| IMG_4462.jpeg | Nico Collins | rec_yards | batch_1 |
| IMG_4463.jpeg | Tyler Shough | pass_yards | batch_2 |
| IMG_4464.jpeg | Greg Dulcich | rec_yards | batch_2 |
| IMG_4465.jpeg | Cooper Kupp | rec_yards | batch_2 |
| IMG_4466.jpeg | AJ Barner | rec_yards | batch_2 |
| IMG_4467.jpeg | Kyler Murray | pass_yards | batch_2 |
| IMG_4468.jpeg | George Kittle | rec_yards | batch_2 |

## Files in this evidence package

- `WEEK7_SUNDAY_SNF_TICKET_CREATION_LEDGER_2026-10-25.md` — this human-readable ledger.
- `ticket_legs.csv` — one row per ticket leg for grading/exposure analysis.
- `pricing_transcription.csv` — exact visible FanDuel alt prices transcribed from user screenshots.
- `pricing_screenshot_manifest.csv` — screenshot mapping + SHA-256.
- `SCREENSHOT_BINARY_ARCHIVE_NOTE.md` — exact archive counts, sizes, SHA-256s, and the binary-transfer limitation.
- Raw JPEG binaries: **not committed by this connector**; use the archive hashes to verify any later binary import.

## Postmortem use

When grading this record, calculate at minimum:
- unique-leg hit rate;
- ticket hit rate;
- repeated-player exposure;
- repeated-game exposure;
- leg kill count;
- counterfactual ticket survival if repeated anchors are removed;
- separate ATD grading from yardage/reception science.

Do not use postgame outcomes to rewrite these cards.
