# Product Requirements Document (PRD)
## FreshTrack AI: Fruit Freshness, Quality Grading & Shelf-Life Prediction System
**Version**: 1.0  
**Last Updated**: February 11, 2026  
**Owner**: Yashash Mathur  
**Stakeholders**: ML Engineering Team, Retail Pilot Users, Placement Recruiters  
**Status**: Finalized

---

## 1. Executive Summary
FreshTrack AI is an end-to-end computer vision system designed to solve food waste in agriculture and retail by delivering accurate, explainable predictions of fruit freshness, quality grade, and remaining shelf-life. The system combines multi-task learning, production-grade MLOps, and a user-friendly interface to address both industry pain points and student project gaps (lack of production-ready ML skills in typical academic projects).

Key differentiators:
- Multi-task CNN for concurrent freshness classification, quality grading, and shelf-life regression.
- Explainable predictions via Grad-CAM to build user trust.
- Full MLOps pipeline (data versioning, experiment tracking, CI/CD, drift monitoring).
- A demo tailored to showcase industry-ready skills for ML placement roles.

---

## 2. Problem Statement
### 2.1 Industry Pain Points
- **Food Waste**: 30% of global food waste stems from inaccurate shelf-life predictions and inconsistent quality grading (FAO, 2025). Retailers discard $16 billion worth of edible produce annually due to lack of objective quality metrics.
- **Lack of Transparency**: Retailers and consumers have no way to verify hidden defects (e.g., internal bruising) that accelerate spoilage.

### 2.2 Student Project Gaps
- Most academic CV projects focus on single-task classification without production MLOps, limiting their relevance for industry roles.
- Few projects address real-world use cases with measurable impact, making them less attractive to recruiters.

FreshTrack AI solves both gaps by building a scalable, explainable system that delivers tangible business value while demonstrating advanced ML engineering skills.

---

## 3. Goals & Objectives
### 3.1 Primary Goals
1. Build a multi-task model with:
   - Freshness classification (4 classes): F1-score ≥ 0.9.
   - Quality grading (3 classes): F1-score ≥ 0.85.
   - Shelf-life regression: MAE ≤ 0.9 days.
2. Deploy the system as a production-ready service with end-to-end MLOps.
3. Create a placement-ready project that impresses recruiters with production ML skills.

### 3.2 Secondary Objectives
1. Reduce food waste by 15% for pilot retail users.
2. Collect 500+ user feedback entries to improve model performance.
3. Achieve a 90% user satisfaction rate with predictions.

---

## 4. Target Audience
| Segment | Use Cases & Needs |
|---------|-------------------|
| **Primary: Retailers/Farmers** | Optimize inventory levels, reduce food waste, standardize quality grading for consistent pricing. |
| **Secondary: ML Recruiters** | Evaluate production ML skills (MLOps, multi-task learning, deployment, explainability). |
| **Tertiary: Home Consumers** | Identify hidden defects and plan fruit consumption to reduce household food waste. |

---

## 5. Functional Requirements
### 5.1 Core Model Features
| ID | Requirement | Priority | Testable Criteria |
|----|-------------|----------|-------------------|
| FR-001 | Classify fruit freshness into 4 classes: Fresh/Semi-ripe/Overripe/Rotten. | High | F1-score ≥0.9 on test dataset. |
| FR-002 | Grade fruit quality into 3 classes: A (Premium)/B (Standard)/C (Poor). | High | F1-score ≥0.85 on test dataset. |
| FR-003 | Predict remaining shelf-life in days with decimal precision. | High | MAE ≤0.9 days on test dataset. |
| FR-004 | Generate Grad-CAM heatmaps to highlight spoilage regions (bruises, mold, overripe patches). | High | Heatmaps correctly identify defect areas for 90% of test images. |

### 5.2 MLOps Pipeline Features
| ID | Requirement | Priority | Testable Criteria |
|----|-------------|----------|-------------------|
| FR-005 | Track dataset versions using DVC with remote storage (S3/GCS). | High | Dataset rollbacks to previous versions are possible. |
| FR-006 | Log experiments and register models using MLflow. | High | All experiments include metrics, hyperparameters, and model artifacts. |
| FR-007 | Containerize the API and model using Docker. | High | Docker image runs on any environment without dependency conflicts. |
| FR-008 | Implement CI/CD via GitHub Actions to auto-test and deploy on main branch merges. | High | PRs trigger unit tests; merging to main deploys the system to Render/AWS. |
| FR-009 | Monitor data/prediction drift using Evidently AI. | High | Alerts are sent when drift exceeds a 20% threshold. |
| FR-010 | Add a feedback endpoint to collect corrected predictions for retraining. | Medium | Users can submit feedback via the frontend; feedback is stored in a database. |

### 5.3 Frontend/Backend Features
| ID | Requirement | Priority | Testable Criteria |
|----|-------------|----------|-------------------|
| FR-011 | Expose a FastAPI endpoint `/predict` to accept images and storage parameters (temp/h
\<Streaming stoppped because the conversation grew too long for this model\>