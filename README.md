# Sustainable Web Assessment Framework

This repository contains the source code used in the study:

**Development of a Carbon-Oriented Web Assessment Framework and Composite Sustainable Web Scores: A Türkiye University Case Study**

The implementation supports automated homepage crawling, calculation of a four-component Sustainable Web Score (SWS), correlation and sensitivity analyses, entropy-based weighting, Monte Carlo weight-uncertainty analysis, and temporal-consistency assessment.

## Overview

The study evaluates university homepages using an automated Playwright-based measurement framework. The analysis consists of four main stages:

1. Automated crawling and network measurement
2. Calculation of normalized SWS components
3. Statistical, sensitivity, and Monte Carlo analyses
4. Temporal-consistency assessment across multiple crawling sessions

The final SWS contains four components:

- standardized data-transfer-based CO2-emission proxy
- media-intensity score
- request-count score
- third-party dependency score

The HTTP/2 component is not included in the final SWS because the HTTP/2 detection field was constant across all observations and therefore provided no discriminatory information.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── crawler.py
│   ├── sws_analysis.py
│   ├── statistical_analysis.py
│   └── temporal_consistency.py
├── data/
│   └── README.md
└── outputs/
    └── README.md
