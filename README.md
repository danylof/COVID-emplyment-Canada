# COVID and Unemployment in Canada: Data Analysis and Insights

## Overview

This repository contains an analysis of labor force statistics in Canada, focusing on the impact of COVID-19 on employment trends across all provinces from January 1976 to January 2023. The dataset used in this analysis is available on [Kaggle](https://www.kaggle.com/datasets/pienik/unemployment-in-canada-by-province-1976-present?resource=download).

## Dataset

- The dataset provides monthly labor statistics for all provinces of Canada from January 1976 to January 2023.
- It includes employment details such as:
  - Employment rates
  - Full-time and part-time employment
  - Labor force participation
  - Unemployment rates
  - Breakdown by age groups and sexes.

## Data Preparation and Cleaning

- Loaded the dataset and explored its structure to understand the available data.
- Converted the date column to `datetime` format for easier manipulation.
- Checked for duplicate rows and missing values:
  - Imputed missing values for `Full-time employment` and `Part-time employment` for Canada by summing values from all other provinces for the same date.
  - Imputed missing values for `Unemployment` and `Unemployment rate` using the labor force and employment data.

## Data Analysis

### 1. **Trends Overview**
- **Unemployment Rate Trends**:
  - General decrease in unemployment rates across all provinces over time.
  - Notable peak in unemployment during the COVID-19 period (beginning in early 2020).

### 2. **Effect of COVID-19 on Employment**
- Analyzed the effect of COVID-19 on different age groups and employment types using statistical tests.
- Conducted the **Wilcoxon signed-rank test** to compare employment metrics (Full-time employment, Part-time employment, Unemployment rate) before and during COVID-19.

### 3. **Post-COVID Changes**
- By the end of 2021, unemployment reached pre-COVID levels in Canada overall and in most provinces.
- Observed that post-COVID, the unemployment rates were generally lower than pre-COVID levels.
- Changes in employment structure: notable decrease in the number of part-time workers.

### 4. **Effect on Different Age Groups**
- Younger populations (15 to 24 years) were more affected by COVID-19, with higher unemployment rates during the pandemic.

## Visualization and Statistical Analysis

- **Data Visualization**:
  - Created line plots to visualize unemployment rate trends by province and age group.
  - Focused on the period from January 2018 to January 2023 to highlight the impact of COVID-19.

- **Statistical Tests**:
  - Employed the Wilcoxon signed-rank test to compare pre-COVID and post-COVID employment metrics.
  - Determined the significance of changes in employment data across different provinces and age groups.
