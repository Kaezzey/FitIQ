# FitIQ

**FitIQ** is a machine learning ranking system for fashion e-commerce. It ranks apparel products by likely user satisfaction, with a focus on **fit** and **overall product quality**, using review text, ratings, and product metadata. A great fashion product looks great but also has to **fit well** on your customers, and therefore should be promoted more on the store page.

Fit is hard to infer from public marketplace data. Star ratings are noisy, review text is subjective, and products with strong headline ratings can still generate repeated sizing complaints. FitIQ handles this as a **within-category ranking problem** and builds product-level signals from weak public evidence.

The project is scoped as a production-style ML pipeline: data ingestion, taxonomy construction, product-level feature generation, and offline ranking evaluation.

---

## Problem

Apparel reviews are noisy.

A 5-star rating may reflect style, price, shipping, or brand preference, with little connection to whether the item actually fits well. Review text has its own issues. It can be sparse, contradictory, and shaped by different sizing expectations.

That makes naive ranking methods unreliable. Sorting by average rating often promotes products that:
- have little evidence behind them
- show inconsistent review patterns
- contain recurring fit complaints
- sit in the wrong comparison set

FitIQ deals with this by:
- ranking products **within comparable apparel groups**
- extracting **fit-related and quality-related signals** at product level
- accounting for **evidence strength**, not only average sentiment

---

## Goal

The goal of FitIQ is to build an end-to-end ranking pipeline that:
- ingests and cleans large-scale review and metadata sources
- derives product-level signals from weak labels
- ranks products within relevant subcategories
- supports offline evaluation with ranking metrics
- is structured like a deployable ML system

In practical terms, the system is intended to reduce the chance that a user’s first few viewed products lead to a poor fit experience.

---

## Current Scope

The current milestone is focused on the data and ranking foundation.

### In scope
- canonical review-level data pipeline
- product metadata integration
- apparel subcategory mapping
- product-level feature generation
- preparation for category-level ranking

### Out of scope for now
- personalization
- online learning
- A/B testing
- body measurement prediction
- computer vision features
- neural ranking models
- API and Docker deployment

These may come later, but they are outside the current build.

---

## Dataset

FitIQ currently uses **Amazon Reviews 2023**.

### Why this dataset
- large-scale and reproducible
- includes review text, ratings, and product identifiers
- includes metadata needed to derive product type
- suitable for weakly supervised product ranking

### Key identifiers
The dataset includes both:
- `asin`
- `parent_asin`

For metadata joins and product-level aggregation, **`parent_asin` is the main product identifier**. Multiple review-level variants can map to the same parent product, so using `parent_asin` keeps the product view consistent.

---

## Core Data Model

FitIQ is built around two main tables.

### 1. Canonical reviews table
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
- preserve a clean review-level dataset
- avoid repeated parsing of raw source files
- provide an auditable base for later feature engineering

### 2. Product features table
**Row = one product**

Typical columns:
- `parent_asin`
- `subcategory`
- `mean_rating`
- `rating_variance`
- `review_count`
- `log_review_count`
- `fit_complaint_rate`

Possible later columns:
- size descriptor aggregates
- metadata-derived features
- text embedding features
- price and brand features

Purpose:
- provide a compact product-level table for ranking
- support evaluation within query groups
- keep feature generation separate from raw data handling

---

## Ranking Setup

FitIQ is a **within-category ranking system**.

Products compete only against items of the same broad type. For example:
- T-shirts compete with T-shirts
- jackets compete with jackets
- pants compete with pants
- dresses compete with dresses
- jewelry competes with jewelry

This is handled by deriving a controlled `subcategory` field from metadata and using it as the ranking query group.

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

This keeps comparisons aligned with how users actually browse products.

---

## Subcategory Mapping

The review dataset alone does not reliably tell us whether a product is a T-shirt, jacket, or ring. FitIQ solves this by joining reviews with product metadata and mapping each product into a controlled apparel taxonomy.

### Strategy
1. Load metadata for the relevant Amazon category split
2. Join metadata back to reviews using `parent_asin`
3. Parse useful fields such as:
   - `categories`
   - `title`
   - other metadata fields where useful
4. Map each product into a controlled set of apparel subcategories

This produces a stable `subcategory` field for within-category ranking.

---

## Fit Signal and Weak Labels

FitIQ treats fit satisfaction as a hidden variable inferred from noisy public evidence.

The dataset does not contain body measurements, return reasons, or direct fit labels. The system uses weak signals such as:
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

These signals are imperfect. That limitation is part of the project and is handled explicitly in the pipeline design and evaluation.

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
- link each review to a product type

### Stage 4: Aggregate product-level features
- group by `parent_asin`
- compute product-level summary features
- output one row per product

### Stage 5: Ranking model and evaluation
- rank products within subcategory
- evaluate with metrics such as NDCG@K and Recall@K
- compare against simple baselines

---

## Engineering Principles

FitIQ is being built around a few clear principles:
- reproducible data artifacts
- clear separation between raw data, features, and modeling
- honest treatment of weak labels
- modular scripts that can be extended later
- evaluation against sensible baselines

The goal is to build a ranking pipeline that is clean, defensible, and easy to extend.

---
