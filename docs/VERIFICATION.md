# Verification Record

**Checked: November 3, 2026**

## Completed checks

| Check | Result |
| --- | --- |
| Python source compilation | All portfolio Python files passed |
| Command-line loading | All 23 command-line entry points returned help successfully |
| Synthetic workflow | Completed with 480 retained records, 80 source records, and 400 target records across five target hospitals |
| Eligibility and deduplication | Removed deliberately ineligible records, repeat stays, and cross-hospital patients; retained six censored-age records as age 90 |
| Label agreement | Mortality labels matched cohort labels after independent row shuffling, including a discharge-status case/whitespace variation |
| Event preparation | Retained static records and in-window events; removed duplicate events, out-of-window events, and identifier-like variables |
| ID alignment | FIDDLE-format feature IDs and metadata were independently reordered; the analysis completed with matched rows |
| Model-to-similarity integration | All target records received source-model predictions and class-specific similarities |
| Performance differences | Source-minus-target differences matched the saved CV and external metrics |
| Numerical checks | Chunked cosine means, self-exclusion, BH adjustment, and class-balanced aggregation matched independent calculations |
| Preparation wrapper | Cohort, labels, population, and input-preparation stages completed on synthetic inputs |
| Figures | Both retained figure outputs were generated; the class-balanced panel was visually inspected |
| Package review | English-only text, valid local documentation links, no original machine paths, and no stored research records or results |
| Original files | Hash checks confirmed that all 21 source files used in curation remained unchanged |

The demonstration is executable through `examples/synthetic_demo.py`. Generated files and local verification logs are outside the submitted source package.

## Limits

Raw clinical-table extraction and the external FIDDLE transformation were not run on the original eICU tables. The synthetic example constructs FIDDLE-format arrays directly. The supplied directory did not contain the original feature arrays, complete research outputs, or an exact historical FIDDLE revision. Numerical reproduction of the poster therefore remains unverified.

The versions in `requirements.txt` describe the local synthetic verification environment, not a recovered lockfile from the original study. Exact installation of that environment on a clean operating system was not separately tested. Synthetic associations and figure statistics have no clinical interpretation.
