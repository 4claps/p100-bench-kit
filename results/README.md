# Shared results

Each file here is one run's result file, downloaded from the kit's run page
(**Download result file**). `TABLES.md` is generated from them.

## Adding yours

1. Run the kit with a **standard model** (Models page) and verify its checksum.
   Custom models are fine for your own testing but are left out of the tables.
2. On the run page, download the result file.
3. Put it in a folder named after your GPU and count, for example
   `results/p100x2/` or `results/v100x4/`, keeping the file name.
4. Open a pull request. The tables are rebuilt with
   `python3 tools/build_tables.py`.

The result file holds the run settings, the numbers, a telemetry summary
(temperature, clocks, power) and a hardware summary: GPU model and count,
driver, PCIe link, CPU model, RAM size, type and speed, and the motherboard
model. It does not hold hostnames, IP addresses, user names, GPU serials or
UUIDs. Open it in a text editor before sharing if you want to check.

Runs you marked invalid (for example a throttling card) are skipped.
