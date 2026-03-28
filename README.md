# FitIQ

**FitIQ** is a production-style machine learning ranking system for apparel products. Its goal is to help an e-commerce platform surface items that users are more likely to be satisfied with, particularly on **fit** and **overall product quality**, using noisy public signals such as review text, ratings, and product metadata.

Rather than treating raw star ratings as ground truth, FitIQ frames apparel satisfaction as a **ranking problem under weak supervision**. It combines product-level review evidence, fit-related signals, and metadata to rank items within comparable apparel groups such as **T-shirts, jackets, pants, dresses, and jewelry**.

This project is designed to mirror the work of real algorithm and software teams building ranking systems: it emphasizes **clean data foundations, reproducibility, honest assumptions, offline evaluation, and modular engineering**.

---

## Problem

Apparel reviews are noisy.

A 5-star rating may reflect style, price, shipping, or expectations rather than actual fit. Review text can be contradictory, subjective, and biased by differences in body shape, sizing expectations, or intended style. A product with a high average rating may still be a poor recommendation if review evidence is sparse, inconsistent, or full of fit complaints.

Naive ranking methods such as sorting by average rating often promote products that:
- have high variance in user experience
- have too little evidence
- look good on the surface but fit poorly
- mix together fundamentally different product types

FitIQ addresses this by ranking products **within relevant apparel subcategories** and explicitly modelling both **satisfaction** and **reliability of evidence**.

---

## Project Goal

The goal of FitIQ is to build an end-to-end ranking pipeline that:

- ingests and cleans large-scale review and product metadata
- derives product-level fit and quality signals from weak labels
- ranks products within comparable subcategories
- supports rigorous offline evaluation
- is structured like a deployable production system

The long-term product idea is simple:

> Reduce the chance that a user’s first few viewed products lead to a poor fit experience.

---

## Current Scope

FitIQ is currently focused on building the **data foundation correctly before modelling**.

### In scope now
- Canonical review-level data pipeline
- Product metadata integration
- Apparel subcategory mapping
- Product-level feature generation
- Preparation for category-level ranking

### Explicitly out of scope for the current stage
- Personalization
- Online learning
- A/B testing
- Body measurement prediction
- Computer vision features
- Neural ranking models
- API and Docker deployment

Those may be added later, but they are not part of the current milestone.

---

## Dataset

FitIQ currently uses **Amazon Reviews 2023** as the primary dataset.

### Why this dataset
- large-scale and reproducible
- contains review text, ratings, and product identifiers
- includes metadata needed to derive product taxonomy
- suitable for building a realistic ranking pipeline under weak supervision

### Key identifiers
The dataset contains both:
- `asin`
- `parent_asin`

For product-level aggregation and metadata joins, **`parent_asin` is the key product identifier**. This matters because multiple review-level variants can belong to the same parent product.

---

## Core Data Model

FitIQ separates the pipeline into two main tables.

### 1. Canonical reviews table
This is the first clean review-level artifact.

**Row = one review**

Typical columns:
- `asin`
- `parent_asin`
- `rating`
- `review_text`
- `timestamp`
- `verified_purchase`
- `helpful_votes`
- `title`

Purpose:
- preserve a clean and auditable review-level dataset
- avoid repeatedly parsing raw source files
- provide the foundation for feature engineering later

### 2. Product features table
This is the later model-ready table.

**Row = one product**

Typical columns:
- `parent_asin`
- `subcategory`
- `mean_rating`
- `rating_variance`
- `review_count`
- `log_review_count`
- `fit_complaint_rate`

Optional later columns:
- size descriptor aggregates
- metadata-derived features
- text embedding components
- price and brand features

Purpose:
- provide a compact product-level feature table for ranking
- support offline evaluation within query groups

---

## Ranking Setup

FitIQ is a **within-category ranking system**.

Products should only compete against other products of the same broad type. For example:
- T-shirts compete with T-shirts
- jackets compete with jackets
- pants compete with pants
- jewelry competes with jewelry

This is handled by constructing a **subcategory label** from metadata and using it as the **query group** for ranking.

### Example query groups
- `t_shirt`
- `shirt_top`
- `dress`
- `jacket`
- `pants`
- `skirt`
- `shoes`
- `jewelry`
- `accessories`
- `unknown`

This prevents nonsensical comparisons across very different product types.

---

## Subcategory Mapping

The raw review data is not enough to determine whether a product is a T-shirt, jacket, or ring. To support within-category ranking, FitIQ joins review data with the product metadata and derives a controlled apparel taxonomy.

### Subcategory derivation strategy
1. Load metadata for the matching Amazon category split
2. Use `parent_asin` to join metadata back to reviews
3. Parse fields such as:
   - `categories`
   - `title`
   - other useful metadata
4. Map products into a small controlled set of apparel subcategories

This gives FitIQ a stable `subcategory` field that later becomes the ranking query group.

---

## Fit Signal and Weak Labels

FitIQ treats fit satisfaction as a **latent variable** inferred from noisy public evidence.

Because the dataset does not contain true body measurements or return reasons, the system uses weak signals such as:
- star ratings
- fit-related complaint phrases in review text
- review consistency
- review volume

Examples of fit-related phrases:
- `too small`
- `too big`
- `runs small`
- `runs large`
- `tight`
- `loose`
- `didn't fit`
- `not true to size`

These signals are not perfect. That limitation is part of the design and is documented explicitly.

---

## Planned Pipeline

### Stage 1: Build canonical review table
- load raw review data
- select relevant columns
- clean and normalize schema
- preserve `parent_asin`
- write a canonical review-level artifact

### Stage 2: Build product taxonomy table
- load raw metadata
- derive apparel subcategory labels
- create a product lookup keyed by `parent_asin`

### Stage 3: Join taxonomy into reviews
- merge subcategory labels into the canonical reviews table
- ensure each review is linked to its product type

### Stage 4: Aggregate product-level features
- group by `parent_asin`
- compute product-level summary features
- output one row per product

### Stage 5: Ranking model and evaluation
- rank products within subcategory
- evaluate using ranking metrics such as NDCG@K and Recall@K
- compare against simple baselines

---

## Repository Structure

```text
FitIQ/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   │   ├── build_reviews_table.py
│   │   ├── build_product_taxonomy.py
│   │   └── merge_reviews_taxonomy.py
│   ├── features/
│   │   └── build_product_features.py
│   ├── modeling/
│   └── evaluation/
├── README.md
└── requirements.txt
