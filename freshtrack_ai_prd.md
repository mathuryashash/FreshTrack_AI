# Product Requirements Document (PRD)
## FreshTrack AI: Intelligent Fruit Quality Assessment System

**Document Version**: 2.0  
**Last Updated**: February 11, 2026  
**Project Owner**: ML Engineering Team  
**Status**: Ready for Development  

---

## 📋 Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Jan 2026 | Initial Team | Initial draft |
| 2.0 | Feb 2026 | ML Team | Added MLOps, monitoring, deployment specs |

---

## 📑 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Business Context](#2-business-context)
3. [Problem Statement](#3-problem-statement)
4. [Goals & Objectives](#4-goals--objectives)
5. [Success Metrics](#5-success-metrics)
6. [User Personas & Use Cases](#6-user-personas--use-cases)
7. [Functional Requirements](#7-functional-requirements)
8. [Non-Functional Requirements](#8-non-functional-requirements)
9. [Technical Architecture](#9-technical-architecture)
10. [Data Requirements](#10-data-requirements)
11. [Machine Learning Specifications](#11-machine-learning-specifications)
12. [API Specifications](#12-api-specifications)
13. [User Interface Requirements](#13-user-interface-requirements)
14. [MLOps Pipeline](#14-mlops-pipeline)
15. [Testing Strategy](#15-testing-strategy)
16. [Security & Compliance](#16-security--compliance)
17. [Deployment Plan](#17-deployment-plan)
18. [Monitoring & Observability](#18-monitoring--observability)
19. [Timeline & Milestones](#19-timeline--milestones)
20. [Risk Analysis](#20-risk-analysis)
21. [Dependencies & Assumptions](#21-dependencies--assumptions)
22. [Future Roadmap](#22-future-roadmap)
23. [Appendices](#23-appendices)

---

## 1. Executive Summary

### 1.1 Project Overview

**FreshTrack AI** is an end-to-end computer vision system that automates fruit quality assessment through real-time image analysis. The system performs three critical tasks:

1. **Freshness Classification**: Categorizes fruits into 4 freshness stages
2. **Quality Grading**: Assigns quality grades (A/B/C) based on visual defects
3. **Shelf-Life Prediction**: Estimates remaining days until spoilage

The system is designed as a production-ready ML service with complete MLOps infrastructure, suitable for deployment in retail environments, warehouses, or as an API service.

### 1.2 Business Value

- **Reduce Food Waste**: 30-40% of produce is discarded due to poor quality assessment
- **Optimize Pricing**: Dynamic pricing based on remaining shelf life
- **Improve Customer Satisfaction**: Ensure consistent quality standards
- **Operational Efficiency**: Automate manual inspection (80% time reduction)

### 1.3 Key Differentiators

| Feature | FreshTrack AI | Competitors |
|---------|---------------|-------------|
| Multi-Task Learning | ✅ 3 simultaneous predictions | ❌ Single task only |
| Explainable AI | ✅ Grad-CAM + LIME | ⚠️ Black box |
| Custom Dataset | ✅ Temporal decay data | ❌ Static images only |
| Production MLOps | ✅ Full pipeline | ⚠️ Research-grade only |
| Deployment Ready | ✅ Docker + Cloud | ❌ Local only |

---

## 2. Business Context

### 2.1 Market Opportunity

**Global Food Waste Problem**:
- $1 trillion worth of food wasted annually
- 30% waste occurs at retail/distribution stage
- AI-based quality control market: $2.1B by 2027 (CAGR 23%)

**Target Markets**:
1. **Primary**: Retail grocery chains (Walmart, Whole Foods)
2. **Secondary**: Fruit distributors and wholesalers
3. **Tertiary**: Smart farming operations

### 2.2 Current Pain Points

| Stakeholder | Current Process | Pain Points |
|-------------|-----------------|-------------|
| Retail Manager | Manual visual inspection | Inconsistent, time-consuming, subjective |
| Supply Chain | Quality checks at each stage | Delayed decisions, lack of data |
| Consumer | Trust-based purchasing | Uncertainty about freshness |

### 2.3 Competitive Landscape

**Existing Solutions**:
- **Manual Inspection**: Subjective, slow, costly
- **Basic CV Systems**: Single-task, no shelf-life prediction
- **Sensor-Based**: Expensive hardware ($5k+ per unit)

**Our Advantage**: Software-only solution with multi-dimensional insights at 10x lower cost.

---

## 3. Problem Statement

### 3.1 Core Problem

**"How can we accurately and efficiently assess fruit quality and predict shelf life using only visual input, in a way that is scalable, explainable, and deployable in real-world settings?"**

### 3.2 Problem Breakdown

1. **Freshness Detection**
   - **Challenge**: Subtle visual changes between stages
   - **Requirement**: 90%+ accuracy across all classes

2. **Quality Grading**
   - **Challenge**: Subjective definitions of "defect"
   - **Requirement**: Consistent grading aligned with industry standards

3. **Shelf-Life Prediction**
   - **Challenge**: No ground truth for Kaggle datasets
   - **Requirement**: Predictions within ±1 day of actual spoilage

4. **Explainability**
   - **Challenge**: Black-box models lack trust
   - **Requirement**: Visual attribution of predictions

5. **Production Deployment**
   - **Challenge**: Research models often fail in production
   - **Requirement**: <500ms latency, 99.9% uptime

---

## 4. Goals & Objectives

### 4.1 Primary Goals

| Goal | Description | Priority |
|------|-------------|----------|
| **G1** | Build multi-task CNN with 90%+ accuracy on freshness | P0 |
| **G2** | Achieve <1 day MAE on shelf-life prediction | P0 |
| **G3** | Deploy production-ready API with <500ms latency | P0 |
| **G4** | Implement complete MLOps pipeline | P0 |
| **G5** | Create explainable AI visualizations | P1 |

### 4.2 Secondary Goals

- **G6**: Mobile app deployment (TensorFlow Lite) - P2
- **G7**: Real-time video stream analysis - P2
- **G8**: Multi-fruit batch processing - P1

### 4.3 Non-Goals (Out of Scope for v1.0)

- ❌ 3D imaging or depth sensors
- ❌ Multi-modal inputs (smell, touch sensors)
- ❌ Fruit type classification (assume known fruit)
- ❌ Integration with existing POS systems
- ❌ Blockchain-based traceability

---

## 5. Success Metrics

### 5.1 Model Performance Metrics

#### Primary Metrics

| Metric | Baseline | Target (v1.0) | Stretch Goal |
|--------|----------|---------------|--------------|
| **Freshness Accuracy** | 75% | 90% | 95% |
| **Quality F1-Score** | 0.70 | 0.85 | 0.92 |
| **Shelf-Life MAE** | 2.5 days | 1.0 day | 0.6 days |
| **Shelf-Life R²** | 0.60 | 0.85 | 0.92 |

#### Secondary Metrics

- **Per-Class Precision/Recall**: >85% for all classes
- **Confusion Matrix**: Off-diagonal <5% for adjacent classes
- **Calibration Error (ECE)**: <0.10 (well-calibrated probabilities)

### 5.2 System Performance Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **API Latency (P50)** | <200ms | Prometheus histogram |
| **API Latency (P99)** | <500ms | Prometheus histogram |
| **Throughput** | 100 req/min | Load testing |
| **Uptime** | 99.5% | Health check monitoring |
| **Model Size** | <20MB | Model registry |

### 5.3 Business Metrics

- **Accuracy vs Human**: Within 5% of expert inspector
- **Inspection Time**: 80% reduction vs manual (30s → 6s)
- **Cost per Prediction**: <$0.001 (cloud inference)

### 5.4 User Satisfaction Metrics

- **Trust Score**: >4.0/5.0 (user survey)
- **Explainability Rating**: >3.5/5.0 (clarity of Grad-CAM)
- **API Integration Ease**: <2 hours for developer onboarding

---

## 6. User Personas & Use Cases

### 6.1 Primary Personas

#### Persona 1: Retail Quality Inspector
**Name**: Sarah Johnson  
**Role**: Quality Control Manager at grocery chain  
**Goals**:
- Process 500+ fruits per day
- Maintain consistent quality standards
- Reduce waste by identifying near-expiry items

**Pain Points**:
- Manual inspection is tiring and subjective
- Difficulty training new staff
- No historical data for trend analysis

**How FreshTrack Helps**:
- Instant quality assessment
- Objective, repeatable decisions
- Dashboard showing quality trends over time

---

#### Persona 2: Supply Chain Manager
**Name**: Michael Chen  
**Role**: Distribution Center Operations  
**Goals**:
- Optimize inventory routing
- Predict demand based on shelf life
- Minimize spoilage losses

**Pain Points**:
- No visibility into remaining shelf life
- Delayed quality issues discovered at retail
- Inefficient pricing strategies

**How FreshTrack Helps**:
- Batch processing of shipments
- API integration with inventory systems
- Predictive alerts for near-expiry stock

---

#### Persona 3: ML Engineer (Internal User)
**Name**: Priya Patel  
**Role**: AI/ML Engineer maintaining the system  
**Goals**:
- Monitor model performance
- Retrain models when drift detected
- Debug prediction errors

**Pain Points**:
- Lack of observability in production
- Manual retraining process
- No A/B testing framework

**How FreshTrack Helps**:
- Complete MLOps pipeline
- Automated drift detection
- Experiment tracking with W&B

---

### 6.2 Use Cases

#### Use Case 1: Single Fruit Quality Check

**Actor**: Retail Inspector  
**Precondition**: Fruit placed in imaging area  
**Flow**:
1. Inspector opens FreshTrack app
2. Captures image of fruit
3. System processes image (2-3 seconds)
4. Display shows:
   - Freshness: "Fresh"
   - Quality: "Grade A"
   - Shelf Life: "6.2 days"
   - Heatmap highlighting any defects
5. Inspector accepts/rejects based on store policy

**Success Criteria**: <5 second end-to-end process

---

#### Use Case 2: Batch Shipment Analysis

**Actor**: Supply Chain Manager  
**Precondition**: 100 fruit images from incoming shipment  
**Flow**:
1. Upload batch via API or bulk upload
2. System processes all images in parallel
3. Generate summary report:
   - Distribution of quality grades
   - Average shelf life
   - Flagged items (rotten or damaged)
4. Export results as CSV for inventory system
5. Manager routes stock based on shelf life (long → distant stores, short → local)

**Success Criteria**: Process 100 images in <30 seconds

---

#### Use Case 3: Model Retraining (Internal)

**Actor**: ML Engineer  
**Precondition**: Drift alert triggered (>30% distribution shift)  
**Flow**:
1. Engineer reviews drift report in monitoring dashboard
2. Downloads recent production data
3. Labels 200-300 new images with ground truth
4. Triggers retraining pipeline via Git commit
5. CI/CD automatically:
   - Trains new model
   - Validates on holdout set
   - Compares to current production model
   - Deploys if metrics improve
6. New model live in <4 hours

**Success Criteria**: Zero-downtime deployment, rollback capability

---

#### Use Case 4: Explainability Review

**Actor**: Retail Manager questioning a prediction  
**Precondition**: Fruit marked as "Overripe" but appears fresh to human eye  
**Flow**:
1. Manager clicks on prediction in history
2. System shows:
   - Original image
   - Grad-CAM heatmap (highlights brown spot on back)
   - LIME explanation (top 3 regions influencing decision)
   - Confidence scores per class
3. Manager rotates fruit, confirms hidden defect
4. Trust in system increases

**Success Criteria**: Non-technical user can interpret visualization

---

## 7. Functional Requirements

### 7.1 Core Functionality

| ID | Requirement | Priority | Acceptance Criteria |
|----|-------------|----------|---------------------|
| **FR-1** | System shall classify fruit into 4 freshness stages | P0 | 90%+ accuracy on test set |
| **FR-2** | System shall assign quality grades (A/B/C) | P0 | F1-score >0.85 |
| **FR-3** | System shall predict shelf life in days | P0 | MAE <1 day |
| **FR-4** | System shall generate Grad-CAM heatmaps | P0 | Heatmap overlays defect regions |
| **FR-5** | System shall support batch processing | P1 | 100 images in <30s |
| **FR-6** | System shall provide confidence scores | P0 | Calibrated probabilities (ECE <0.10) |
| **FR-7** | System shall handle multiple fruit types | P1 | 5+ fruit categories |
| **FR-8** | System shall log all predictions | P0 | Persistent storage with metadata |

### 7.2 Input Requirements

| ID | Requirement | Specification |
|----|-------------|---------------|
| **FR-IN-1** | Accept JPEG/PNG images | Size: 224x224 to 1024x1024 |
| **FR-IN-2** | Support RGB images only | 3 channels required |
| **FR-IN-3** | Handle various lighting conditions | Model trained on augmented data |
| **FR-IN-4** | Accept images via API upload | multipart/form-data |
| **FR-IN-5** | Accept images via webcam capture | Base64 encoded |
| **FR-IN-6** | Validate file size | Max 5MB per image |
| **FR-IN-7** | Reject invalid formats | Return 400 error with message |

### 7.3 Output Requirements

| ID | Requirement | Format |
|----|-------------|--------|
| **FR-OUT-1** | Return freshness class | String: "Fresh" / "Semi-ripe" / "Overripe" / "Rotten" |
| **FR-OUT-2** | Return quality grade | String: "Grade A" / "Grade B" / "Grade C" |
| **FR-OUT-3** | Return shelf life | Float: days (e.g., 5.7) |
| **FR-OUT-4** | Return confidence scores | Dict: {class: probability} |
| **FR-OUT-5** | Return heatmap URL | String: URL to visualization |
| **FR-OUT-6** | Return processing time | Float: milliseconds |
| **FR-OUT-7** | Return model version | String: e.g., "v2.1.3" |

### 7.4 API Endpoints

#### Endpoint 1: Single Prediction
```
POST /api/v1/predict
Content-Type: multipart/form-data

Request:
- file: image file
- return_heatmap: boolean (optional, default: true)
- fruit_type: string (optional, for multi-fruit models)

Response (200):
{
  "prediction_id": "uuid",
  "timestamp": "2026-02-11T14:30:00Z",
  "freshness": {
    "class": "Fresh",
    "confidence": 0.92,
    "probabilities": {
      "Fresh": 0.92,
      "Semi-ripe": 0.06,
      "Overripe": 0.01,
      "Rotten": 0.01
    }
  },
  "quality": {
    "grade": "A",
    "confidence": 0.88
  },
  "shelf_life": {
    "days": 6.2,
    "confidence_interval": [5.5, 6.9]
  },
  "heatmap_url": "https://cdn.../heatmap_abc123.jpg",
  "metadata": {
    "model_version": "v2.1.3",
    "inference_time_ms": 287
  }
}

Error Responses:
- 400: Invalid image format
- 413: File too large
- 500: Model inference failed
```

#### Endpoint 2: Batch Prediction
```
POST /api/v1/predict/batch
Content-Type: multipart/form-data

Request:
- files: array of image files (max 100)

Response (200):
{
  "batch_id": "uuid",
  "timestamp": "2026-02-11T14:30:00Z",
  "total_images": 50,
  "successful": 48,
  "failed": 2,
  "predictions": [ /* array of prediction objects */ ],
  "summary": {
    "freshness_distribution": {
      "Fresh": 20,
      "Semi-ripe": 15,
      "Overripe": 10,
      "Rotten": 3
    },
    "average_shelf_life": 4.3,
    "quality_distribution": {
      "Grade A": 25,
      "Grade B": 18,
      "Grade C": 5
    }
  },
  "download_url": "https://api.../batch_results_xyz.csv"
}
```

#### Endpoint 3: Health Check
```
GET /health

Response (200):
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v2.1.3",
  "uptime_seconds": 86400,
  "last_prediction": "2026-02-11T14:29:45Z"
}
```

#### Endpoint 4: Model Metadata
```
GET /api/v1/model/info

Response (200):
{
  "version": "v2.1.3",
  "training_date": "2026-02-01",
  "architecture": "EfficientNet-B0",
  "parameters": 5300000,
  "training_samples": 18500,
  "validation_metrics": {
    "freshness_accuracy": 0.912,
    "quality_f1": 0.876,
    "shelf_life_mae": 0.87
  }
}
```

### 7.5 UI Functionality

| ID | Feature | Description |
|----|---------|-------------|
| **FR-UI-1** | Image Upload | Drag-and-drop or file picker |
| **FR-UI-2** | Live Webcam Capture | Stream from device camera |
| **FR-UI-3** | Result Display | Card layout with metrics |
| **FR-UI-4** | Heatmap Overlay | Toggle between original and heatmap |
| **FR-UI-5** | History View | Last 20 predictions |
| **FR-UI-6** | Export Results | Download as PDF/CSV |
| **FR-UI-7** | Comparison Mode | Side-by-side comparison of 2 fruits |
| **FR-UI-8** | Batch Upload | Upload multiple images (max 50) |

---

## 8. Non-Functional Requirements

### 8.1 Performance Requirements

| ID | Requirement | Target | Measurement |
|----|-------------|--------|-------------|
| **NFR-P1** | API response time (P50) | <200ms | Prometheus |
| **NFR-P2** | API response time (P99) | <500ms | Prometheus |
| **NFR-P3** | Model inference time | <150ms | GPU profiling |
| **NFR-P4** | Batch processing throughput | 100 images/min | Load test |
| **NFR-P5** | Concurrent users | 50+ simultaneous | Load test |
| **NFR-P6** | UI load time | <2s | Lighthouse |

### 8.2 Scalability Requirements

| ID | Requirement | Specification |
|----|-------------|---------------|
| **NFR-S1** | Horizontal scaling | Auto-scale to 5 instances under load |
| **NFR-S2** | Database capacity | Support 1M+ predictions logged |
| **NFR-S3** | Model serving | Handle 10k requests/day |
| **NFR-S4** | Storage | 100GB for models + artifacts |

### 8.3 Reliability Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| **NFR-R1** | System uptime | 99.5% (43 hours downtime/year) |
| **NFR-R2** | Error rate | <1% of predictions |
| **NFR-R3** | Data durability | 99.999% (S3 standard) |
| **NFR-R4** | Disaster recovery | RTO <4 hours, RPO <1 hour |

### 8.4 Usability Requirements

| ID | Requirement | Measurement |
|----|-------------|-------------|
| **NFR-U1** | API documentation | OpenAPI 3.0 spec, interactive docs |
| **NFR-U2** | Learning curve | New developer productive in <2 hours |
| **NFR-U3** | Error messages | Clear, actionable messages |
| **NFR-U4** | UI accessibility | WCAG 2.1 Level AA compliance |

### 8.5 Maintainability Requirements

| ID | Requirement | Specification |
|----|-------------|---------------|
| **NFR-M1** | Code coverage | >80% unit test coverage |
| **NFR-M2** | Documentation | All functions documented (docstrings) |
| **NFR-M3** | Logging | Structured JSON logs (info/warn/error) |
| **NFR-M4** | Dependency management | Poetry/pip-tools for reproducibility |
| **NFR-M5** | Model versioning | All models tagged in registry |

### 8.6 Portability Requirements

| ID | Requirement | Specification |
|----|-------------|---------------|
| **NFR-PT1** | Containerization | Docker images <1GB |
| **NFR-PT2** | Cloud-agnostic | Works on AWS, GCP, Azure |
| **NFR-PT3** | Model formats | PyTorch, ONNX, TFLite |
| **NFR-PT4** | Database | PostgreSQL or MySQL |

---

## 9. Technical Architecture

### 9.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Streamlit   │  │   Gradio     │  │  Mobile App  │      │
│  │     UI       │  │     UI       │  │   (Future)   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Load Balancer │
                    │     (Nginx)     │
                    └────────┬────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │                                     │
┌─────────▼─────────┐               ┌──────────▼──────────┐
│   FastAPI Service │               │  Prometheus/Grafana │
│   (Main API)      │               │   (Monitoring)      │
│                   │               └─────────────────────┘
│  ┌─────────────┐  │
│  │  Prediction │  │
│  │   Service   │  │
│  └──────┬──────┘  │
│         │         │
│  ┌──────▼──────┐  │
│  │   Model     │  │
│  │  Inference  │  │
│  │   Engine    │  │
│  └──────┬──────┘  │
└─────────┼─────────┘
          │
    ┌─────▼─────┐
    │   Model   │
    │  Storage  │
    │    (S3)   │
    └─────┬─────┘
          │
┌─────────▼─────────┐
│   PostgreSQL DB   │
│  (Predictions +   │
│   Metadata)       │
└───────────────────┘

MLOps Pipeline (Offline):
┌──────────┐    ┌────────────┐    ┌────────────┐
│   DVC    │───▶│  Training  │───▶│  W&B/MLflow│
│ (Data)   │    │  Pipeline  │    │ (Tracking) │
└──────────┘    └─────┬──────┘    └────────────┘
                      │
                ┌─────▼──────┐
                │   Model    │
                │  Registry  │
                └─────┬──────┘
                      │
                ┌─────▼──────┐
                │   CI/CD    │
                │  Pipeline  │
                └─────┬──────┘
                      │
                  [Deploy]
```

### 9.2 Component Breakdown

#### 9.2.1 Inference Service
**Technology**: FastAPI + Uvicorn  
**Responsibilities**:
- HTTP request handling
- Input validation
- Model loading and caching
- Response formatting
- Error handling

**Scaling Strategy**: Horizontal (multiple replicas behind load balancer)

---

#### 9.2.2 Model Inference Engine
**Technology**: PyTorch + TorchServe (optional)  
**Responsibilities**:
- Image preprocessing
- Model forward pass
- Post-processing (softmax, denormalization)
- Heatmap generation (Grad-CAM)

**Optimization**:
- Batch inference for throughput
- Model quantization (INT8)
- GPU acceleration (CUDA)

---

#### 9.2.3 Data Layer
**Technology**: PostgreSQL + S3  
**Schema**:
```sql
CREATE TABLE predictions (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    image_url TEXT NOT NULL,
    freshness_class VARCHAR(20),
    freshness_confidence FLOAT,
    quality_grade VARCHAR(10),
    shelf_life_days FLOAT,
    model_version VARCHAR(20),
    inference_time_ms INT,
    user_id UUID,
    feedback_rating INT
);

CREATE INDEX idx_timestamp ON predictions(timestamp);
CREATE INDEX idx_model_version ON predictions(model_version);
```

---

#### 9.2.4 Monitoring Stack
**Components**:
- **Prometheus**: Metrics collection
- **Grafana**: Dashboards
- **Evidently**: ML-specific monitoring (drift, quality)
- **Sentry**: Error tracking

**Key Dashboards**:
1. System health (CPU, memory, latency)
2. Model performance (accuracy, confidence distribution)
3. Data drift alerts
4. User engagement

---

### 9.3 Technology Stack

| Layer | Technology | Justification |
|-------|------------|---------------|
| **Deep Learning** | PyTorch 2.0 | Dynamic graphs, extensive ecosystem |
| **Training Framework** | PyTorch Lightning | Reduces boilerplate, built-in logging |
| **Model Backbone** | EfficientNet-B0 | Best accuracy/efficiency tradeoff |
| **Computer Vision** | OpenCV, Albumentations | Industry standard, fast augmentations |
| **API Framework** | FastAPI | Async support, auto documentation |
| **Web Server** | Uvicorn | ASGI server, production-ready |
| **Database** | PostgreSQL 14+ | Robust, JSON support, full-text search |
| **Object Storage** | AWS S3 / MinIO | Scalable, durable |
| **Experiment Tracking** | Weights & Biases | Better UI than MLflow, free tier |
| **Data Versioning** | DVC | Git-like for datasets |
| **Containerization** | Docker + docker-compose | Reproducibility, portability |
| **Orchestration** | Docker Swarm / K8s (future) | Auto-scaling, self-healing |
| **CI/CD** | GitHub Actions | Free for public repos, easy setup |
| **Monitoring** | Prometheus + Grafana | Open source, widely adopted |
| **Model Serving** | TorchServe (optional) | Production-grade serving |
| **Frontend** | Streamlit + Gradio | Rapid prototyping, Python-native |

---

### 9.4 Model Architecture Details

```python
class FreshTrackModel(nn.Module):
    """
    Multi-task CNN for fruit quality assessment
    
    Architecture:
        Input (224x224x3)
        ↓
        EfficientNet-B0 Backbone (pretrained)
        ↓
        Global Average Pooling (1280 features)
        ↓
        ┌─────────────┬───────────────┬──────────────┐
        │             │               │              │
    Freshness    Quality      Shelf-Life      Aux Task
    Head (4)     Head (3)     Head (1)        (rotation)
        │             │               │              │
    Softmax      Softmax       ReLU           Softmax
    """
    
    def __init__(self, num_freshness=4, num_quality=3):
        super().__init__()
        
        # Backbone
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,  # Remove head
            global_pool='avg'
        )
        
        in_features = 1280
        
        # Task-specific heads
        self.freshness_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_freshness)
        )
        
        self.quality_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_quality)
        )
        
        self.shelf_life_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU()  # Ensure positive days
        )
        
        # Auxiliary task (helps regularization)
        self.rotation_head = nn.Linear(in_features, 4)
    
    def forward(self, x, return_features=False):
        # Extract features
        features = self.backbone(x)
        
        # Multi-task predictions
        freshness_logits = self.freshness_head(features)
        quality_logits = self.quality_head(features)
        shelf_life = self.shelf_life_head(features)
        rotation_logits = self.rotation_head(features)
        
        if return_features:
            return {
                'freshness': freshness_logits,
                'quality': quality_logits,
                'shelf_life': shelf_life,
                'rotation': rotation_logits,
                'features': features
            }
        
        return freshness_logits, quality_logits, shelf_life, rotation_logits
```

**Model Statistics**:
- **Parameters**: 5.3M (backbone) + 0.8M (heads) = 6.1M total
- **FLOPs**: ~390M (EfficientNet-B0)
- **Memory**: ~25MB (FP32), ~6MB (INT8 quantized)
- **Inference Time**: ~80ms (CPU), ~15ms (GPU)

---

### 9.5 Data Flow Diagram

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ 1. Upload image (JPEG/PNG)
     ▼
┌─────────────┐
│  FastAPI    │
│  Endpoint   │
└────┬────────┘
     │ 2. Validate format/size
     ▼
┌─────────────┐
│ Preprocessing│ 3. Resize to 224x224
│   Pipeline  │    Normalize RGB values
└────┬────────┘    Convert to tensor
     │
     ▼
┌─────────────┐
│   Model     │ 4. Forward pass
│  Inference  │    Multi-task prediction
└────┬────────┘
     │
     ├─────────────┐
     │             │ 5a. Generate Grad-CAM
     ▼             ▼
┌─────────────┐ ┌──────────────┐
│ Predictions │ │  Explainability│
│  (logits)   │ │   (heatmap)    │
└────┬────────┘ └───┬──────────┘
     │              │
     │ 6. Post-process (softmax, scaling)
     │              │
     └──────┬───────┘
            ▼
     ┌──────────────┐
     │   Database   │ 7. Log prediction
     │  (Postgres)  │    + metadata
     └──────────────┘
            │
            ▼
     ┌──────────────┐
     │  S3 Storage  │ 8. Store heatmap image
     └──────────────┘
            │
            ▼
     ┌──────────────┐
     │   Response   │ 9. Return JSON
     │   to Client  │    + heatmap URL
     └──────────────┘
```

---

## 10. Data Requirements

### 10.1 Dataset Composition

#### 10.1.1 Training Data Sources

| Source | Size | Quality | Usage |
|--------|------|---------|-------|
| **Kaggle Fruit Dataset** | 13,599 images | High quality, labeled | 70% training |
| **Fruit Quality Detection** | 6,033 images | Medium quality | 15% validation |
| **Custom Timelapse** | 500-1000 images | High quality, temporal | 15% test + fine-tuning |
| **Web Scraping** (optional) | 2,000 images | Variable | Augmentation only |

**Total Dataset**: ~20,000 images (after cleaning)

---

#### 10.1.2 Data Distribution

**Freshness Classes**:
```
Fresh:      35% (7,000 images)
Semi-ripe:  30% (6,000 images)
Overripe:   20% (4,000 images)
Rotten:     15% (3,000 images)
```

**Quality Grades**:
```
Grade A:    40% (8,000 images)
Grade B:    35% (7,000 images)
Grade C:    25% (5,000 images)
```

**Fruit Types** (multi-fruit extension):
```
Banana:     25%
Apple:      20%
Orange:     20%
Strawberry: 15%
Tomato:     10%
Others:     10%
```

---

### 10.2 Data Collection Specifications

#### 10.2.1 Custom Timelapse Protocol

**Equipment**:
- Smartphone camera (12MP+)
- Fixed tripod
- Consistent LED lighting (5000K daylight)
- White background

**Procedure**:
1. **Day 0**: Purchase fresh fruits (5 samples per type)
2. **Daily Capture**:
   - Same time (10 AM ± 30 min)
   - Same angle (45° from horizontal)
   - Multiple orientations (front, back, top)
   - Total: 15 photos/day (5 fruits × 3 angles)
3. **Labeling**:
   - Visual inspection → freshness stage
   - Measure actual spoilage date
   - Calculate days_remaining = spoilage_date - current_date
4. **Duration**: 20 days per fruit batch
5. **Repeats**: 3 batches (different seasons/sources)

**Labeling Schema**:
```json
{
  "image_id": "banana_batch1_day05_front.jpg",
  "fruit_type": "banana",
  "batch_id": "batch1",
  "day_number": 5,
  "angle": "front",
  "freshness": "Fresh",
  "quality": "A",
  "days_remaining": 5.0,
  "actual_spoilage_date": "2026-02-16",
  "visible_defects": ["minor_bruise"],
  "metadata": {
    "temperature_c": 22,
    "humidity_percent": 60
  }
}
```

---

#### 10.2.2 Data Quality Standards

| Criterion | Requirement | Rejection Threshold |
|-----------|-------------|---------------------|
| **Resolution** | ≥800×600 | <640×480 |
| **Brightness** | 80-180 (mean pixel value) | <50 or >200 |
| **Blur** | Laplacian variance >100 | <50 (too blurry) |
| **Focus** | Fruit occupies 40-80% of frame | <30% or >90% |
| **Background** | Single color preferred | Cluttered (manual review) |
| **Occlusion** | <10% of fruit hidden | >25% hidden |

**Quality Check Script**:
```python
def check_image_quality(image_path):
    img = cv2.imread(image_path)
    
    # Check resolution
    h, w = img.shape[:2]
    if h < 640 or w < 640:
        return False, "Resolution too low"
    
    # Check brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 50 or mean_brightness > 200:
        return False, "Poor lighting"
    
    # Check blur
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        return False, "Image too blurry"
    
    return True, "Pass"
```

---

### 10.3 Data Preprocessing Pipeline

#### Stage 1: Raw Data Cleaning
```python
# Remove duplicates (perceptual hashing)
# Remove corrupted files
# Fix EXIF orientation
# Standardize naming convention
```

#### Stage 2: Annotation Validation
```python
# Check label consistency
# Flag outliers (e.g., "Fresh" but shelf_life=0)
# Inter-annotator agreement (if multiple labelers)
```

#### Stage 3: Train/Val/Test Split
```python
# Stratified split (maintain class distribution)
# Temporal split for custom data (avoid data leakage)
# Test set: held-out fruits (not seen in training)
```

#### Stage 4: Augmentation (Training Only)
```python
train_transform = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
        p=0.5
    ),
    A.GaussNoise(p=0.2),
    A.GaussianBlur(p=0.2),
    A.CoarseDropout(
        max_holes=8,
        max_height=16,
        max_width=16,
        p=0.3
    ),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])
```

---

### 10.4 Data Versioning Strategy

**DVC Configuration**:
```yaml
# .dvc/config
[core]
    remote = s3storage
    autostage = true

['remote "s3storage"']
    url = s3://freshtrack-ai-data/dvc-storage
    region = us-east-1
```

**Dataset Versions**:
```
v1.0: Initial Kaggle-only dataset (15k images)
v1.1: Added custom timelapse batch 1 (500 images)
v2.0: Quality re-annotation + augmentation (20k images)
v2.1: Multi-fruit expansion (25k images)
```

**Tracking Command**:
```bash
# Add new data
dvc add data/raw/fruits
git add data/raw/fruits.dvc
git commit -m "feat: add batch 2 timelapse data"
git tag -a v2.1 -m "Dataset v2.1 - multi-fruit support"
dvc push
```

---

## 11. Machine Learning Specifications

### 11.1 Training Configuration

#### 11.1.1 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Batch Size** | 32 | Balance memory/convergence |
| **Learning Rate** | 1e-4 (backbone), 1e-3 (heads) | Differential learning rates |
| **Optimizer** | AdamW | Weight decay regularization |
| **Weight Decay** | 0.01 | Prevent overfitting |
| **LR Scheduler** | CosineAnnealingWarmRestarts | Smooth convergence |
| **Epochs** | 50 (early stopping) | Typically converges at 30-40 |
| **Gradient Clip** | 1.0 | Prevent exploding gradients |
| **Mixed Precision** | FP16 | 2x speedup on GPU |

#### 11.1.2 Loss Function

```python
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss()
    
    def forward(self, outputs, targets):
        # Unpack outputs
        fresh_logits = outputs['freshness']
        qual_logits = outputs['quality']
        shelf_pred = outputs['shelf_life']
        rot_logits = outputs['rotation']
        
        # Unpack targets
        fresh_labels = targets['freshness']
        qual_labels = targets['quality']
        shelf_true = targets['shelf_life']
        rot_labels = targets['rotation']
        
        # Classification losses
        loss_fresh = self.focal_loss(fresh_logits, fresh_labels)
        loss_qual = self.ce_loss(qual_logits, qual_labels)
        
        # Regression loss (combined MSE + Huber for robustness)
        loss_shelf = (
            0.5 * self.mse_loss(shelf_pred, shelf_true) +
            0.5 * self.huber_loss(shelf_pred, shelf_true)
        )
        
        # Auxiliary task
        loss_rot = self.ce_loss(rot_logits, rot_labels)
        
        # Weighted combination
        total_loss = (
            0.4 * loss_fresh +
            0.3 * loss_qual +
            0.25 * loss_shelf +
            0.05 * loss_rot  # Regularization only
        )
        
        return total_loss, {
            'loss_freshness': loss_fresh.item(),
            'loss_quality': loss_qual.item(),
            'loss_shelf_life': loss_shelf.item(),
            'loss_rotation': loss_rot.item()
        }
```

**Focal Loss** for class imbalance:
```
FL(p_t) = -α_t(1 - p_t)^γ log(p_t)
```
- Focuses on hard examples
- Down-weights easy examples
- γ=2.0 (standard)

---

### 11.2 Training Protocol

#### 11.2.1 Three-Stage Training

**Stage 1: Warmup (5 epochs)**
```python
# Freeze backbone, train heads only
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)
```

**Stage 2: Fine-tuning (20 epochs)**
```python
# Unfreeze last 2 blocks of backbone
for param in model.backbone.blocks[5:].parameters():
    param.requires_grad = True

optimizer = AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-4},
    {'params': model.freshness_head.parameters(), 'lr': 1e-3},
    {'params': model.quality_head.parameters(), 'lr': 1e-3},
    {'params': model.shelf_life_head.parameters(), 'lr': 1e-3}
])
```

**Stage 3: Full Fine-tuning (25 epochs)**
```python
# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Lower learning rate
optimizer = AdamW([
    {'params': model.backbone.parameters(), 'lr': 5e-5},
    {'params': model.freshness_head.parameters(), 'lr': 5e-4},
    # ...
])
```

---

#### 11.2.2 Early Stopping Criteria

```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=7,
    min_delta=0.001,
    mode='min',
    restore_best_weights=True
)
```

**Validation Strategy**:
- Evaluate every epoch
- Save top 3 models by validation loss
- Final model selection by composite metric:
  ```
  score = 0.4 * freshness_acc + 0.3 * quality_f1 - 0.3 * shelf_mae
  ```

---

### 11.3 Evaluation Metrics

#### 11.3.1 Classification Metrics

**Per-Class Metrics**:
```python
from sklearn.metrics import classification_report

# Freshness classification
report = classification_report(
    y_true,
    y_pred,
    target_names=['Fresh', 'Semi-ripe', 'Overripe', 'Rotten'],
    digits=3
)
```

**Confusion Matrix**:
```
              Fresh  Semi-ripe  Overripe  Rotten
Fresh          0.92       0.06      0.02    0.00
Semi-ripe      0.05       0.88      0.06    0.01
Overripe       0.02       0.08      0.85    0.05
Rotten         0.00       0.01      0.09    0.90
```

**Target**: Off-diagonal <0.10, especially adjacent classes

---

#### 11.3.2 Regression Metrics

**Shelf-Life Prediction**:
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"MAE: {mae:.2f} days")
print(f"RMSE: {rmse:.2f} days")
print(f"R²: {r2:.3f}")
print(f"MAPE: {mape:.1f}%")
```

**Target Values**:
- MAE < 1.0 day
- R² > 0.85
- MAPE < 15%

**Error Analysis by Freshness Stage**:
| Stage | MAE Target |
|-------|------------|
| Fresh | <0.8 days |
| Semi-ripe | <1.0 days |
| Overripe | <1.2 days |
| Rotten | <0.5 days |

---

#### 11.3.3 Calibration Metrics

**Expected Calibration Error (ECE)**:
```python
def expected_calibration_error(probs, labels, n_bins=10):
    """
    Measures how well predicted probabilities match actual outcomes
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece
```

**Target**: ECE < 0.10 (well-calibrated)

---

### 11.4 Explainability Implementation

#### 11.4.1 Grad-CAM

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def __call__(self, image, class_idx):
        # Forward pass
        output = self.model(image.unsqueeze(0))
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0][class_idx]
        class_score.backward()
        
        # Generate heatmap
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)  # ReLU to remove negative influence
        heatmap = heatmap / torch.max(heatmap)  # Normalize
        
        return heatmap.cpu().numpy()
```

**Usage**:
```python
grad_cam = GradCAM(model, model.backbone.blocks[-1])
heatmap = grad_cam(image, predicted_class)

# Overlay on original image
colored_heatmap = cv2.applyColorMap(
    np.uint8(255 * heatmap),
    cv2.COLORMAP_JET
)
overlayed = cv2.addWeighted(original_image, 0.5, colored_heatmap, 0.5, 0)
```

---

#### 11.4.2 LIME (Local Interpretable Model-agnostic Explanations)

```python
from lime import lime_image
from skimage.segmentation import mark_boundaries

explainer = lime_image.LimeImageExplainer()

def explain_prediction(image, model, top_labels=3):
    def predict_fn(images):
        # Preprocess and predict
        batch = preprocess_batch(images)
        with torch.no_grad():
            outputs = model(batch)
        return outputs[0].softmax(dim=1).cpu().numpy()
    
    explanation = explainer.explain_instance(
        image,
        predict_fn,
        top_labels=top_labels,
        hide_color=0,
        num_samples=1000
    )
    
    # Get image with highlighted regions
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    
    return mark_boundaries(temp / 255.0, mask)
```

---

### 11.5 Model Optimization

#### 11.5.1 Quantization

```python
# Post-training quantization (INT8)
quantized_model = torch.quantization.quantize_dynamic(
    model.cpu(),
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

# Size reduction: 25MB → 6MB
# Speed improvement: 2-3x on CPU
# Accuracy loss: <2%
```

#### 11.5.2 ONNX Export

```python
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "freshtrack_v2.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['freshness', 'quality', 'shelf_life'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'freshness': {0: 'batch_size'},
        'quality': {0: 'batch_size'},
        'shelf_life': {0: 'batch_size'}
    }
)
```

**Benefits**:
- Cross-platform deployment (C++, Java, web)
- Runtime flexibility (ONNX Runtime)
- Hardware acceleration (TensorRT, OpenVINO)

---

#### 11.5.3 Model Pruning (Future)

```python
import torch.nn.utils.prune as prune

# Prune 40% of connections in Linear layers
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.4)

# Remove pruning reparameterization
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        prune.remove(module, 'weight')
```

**Expected**: 30-40% size reduction with <3% accuracy loss

---

## 12. API Specifications

### 12.1 RESTful API Design

**Base URL**: `https://api.freshtrack.ai/v1`

**Authentication** (v2.0 feature):
```
Authorization: Bearer <JWT_TOKEN>
X-API-Key: <API_KEY>
```

**Rate Limiting**:
- Free tier: 100 requests/hour
- Premium: 10,000 requests/hour

---

### 12.2 Endpoint Details

#### 12.2.1 POST /predict

**Description**: Analyze a single fruit image

**Request**:
```http
POST /api/v1/predict HTTP/1.1
Host: api.freshtrack.ai
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary

------WebKitFormBoundary
Content-Disposition: form-data; name="file"; filename="apple.jpg"
Content-Type: image/jpeg

[binary image data]
------WebKitFormBoundary
Content-Disposition: form-data; name="return_heatmap"

true
------WebKitFormBoundary
Content-Disposition: form-data; name="fruit_type"

apple
------WebKitFormBoundary--
```

**Response (200 OK)**:
```json
{
  "prediction_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-02-11T14:30:00.123Z",
  "freshness": {
    "class": "Fresh",
    "confidence": 0.92,
    "probabilities": {
      "Fresh": 0.92,
      "Semi-ripe": 0.06,
      "Overripe": 0.01,
      "Rotten": 0.01
    }
  },
  "quality": {
    "grade": "A",
    "confidence": 0.88,
    "probabilities": {
      "A": 0.88,
      "B": 0.10,
      "C": 0.02
    }
  },
  "shelf_life": {
    "days": 6.2,
    "confidence_interval": [5.5, 6.9],
    "prediction_std": 0.35
  },
  "heatmap_url": "https://cdn.freshtrack.ai/heatmaps/550e8400.jpg",
  "recommendations": [
    "Store in cool, dry place",
    "Consume within 6 days for best quality"
  ],
  "metadata": {
    "model_version": "v2.1.3",
    "inference_time_ms": 287,
    "image_resolution": "1024x768",
    "fruit_type": "apple"
  }
}
```

**Error Responses**:

```json
// 400 Bad Request - Invalid image
{
  "error": "INVALID_IMAGE_FORMAT",
  "message": "Only JPEG and PNG formats are supported",
  "supported_formats": ["image/jpeg", "image/png"],
  "request_id": "abc123"
}

// 413 Payload Too Large
{
  "error": "FILE_TOO_LARGE",
  "message": "Image size exceeds 5MB limit",
  "max_size_mb": 5,
  "actual_size_mb": 7.2
}

// 429 Too Many Requests
{
  "error": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit of 100 req/hour exceeded",
  "retry_after": 3600,
  "remaining_quota": 0
}

// 500 Internal Server Error
{
  "error": "MODEL_INFERENCE_FAILED",
  "message": "Unable to process image",
  "request_id": "abc123",
  "support_email": "support@freshtrack.ai"
}
```

---

#### 12.2.2 POST /predict/batch

**Description**: Analyze multiple fruit images

**Request**:
```http
POST /api/v1/predict/batch HTTP/1.1
Content-Type: multipart/form-data

files: [image1.jpg, image2.jpg, ..., image50.jpg]
```

**Response (200 OK)**:
```json
{
  "batch_id": "batch_20260211_143000",
  "timestamp": "2026-02-11T14:30:00Z",
  "status": "completed",
  "total_images": 50,
  "successful": 48,
  "failed": 2,
  "predictions": [
    { /* individual prediction object */ },
    // ... 47 more
  ],
  "failed_images": [
    {
      "filename": "image34.jpg",
      "error": "INVALID_FORMAT"
    },
    {
      "filename": "image45.jpg",
      "error": "IMAGE_TOO_BLURRY"
    }
  ],
  "summary": {
    "freshness_distribution": {
      "Fresh": 20,
      "Semi-ripe": 15,
      "Overripe": 10,
      "Rotten": 3
    },
    "average_shelf_life": 4.3,
    "quality_distribution": {
      "Grade A": 25,
      "Grade B": 18,
      "Grade C": 5
    },
    "recommended_actions": {
      "immediate_sale": 3,
      "discount_pricing": 10,
      "standard_pricing": 35
    }
  },
  "download_url": "https://api.freshtrack.ai/downloads/batch_xyz.csv",
  "processing_time_ms": 15400
}
```

---

#### 12.2.3 GET /health

**Description**: Health check endpoint

**Response (200 OK)**:
```json
{
  "status": "healthy",
  "timestamp": "2026-02-11T14:30:00Z",
  "service": "freshtrack-api",
  "version": "2.1.3",
  "model": {
    "loaded": true,
    "version": "v2.1.3",
    "last_loaded": "2026-02-11T10:00:00Z"
  },
  "database": {
    "connected": true,
    "latency_ms": 12
  },
  "uptime_seconds": 86400,
  "last_prediction": "2026-02-11T14:29:45Z",
  "predictions_today": 1234
}
```

---

#### 12.2.4 GET /model/info

**Response**:
```json
{
  "model_version": "v2.1.3",
  "architecture": "EfficientNet-B0 Multi-Task",
  "training_date": "2026-02-01",
  "dataset_size": 20000,
  "parameters": 6100000,
  "model_size_mb": 24.5,
  "quantized_size_mb": 6.2,
  "metrics": {
    "freshness_accuracy": 0.912,
    "quality_f1": 0.876,
    "shelf_life_mae": 0.87,
    "shelf_life_r2": 0.89
  },
  "supported_fruits": ["apple", "banana", "orange", "strawberry", "tomato"],
  "inference_time": {
    "cpu_ms": 280,
    "gpu_ms": 45
  }
}
```

---

#### 12.2.5 GET /predictions/{prediction_id}

**Description**: Retrieve a past prediction

**Response (200 OK)**:
```json
{
  "prediction_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-02-11T14:30:00.123Z",
  "image_url": "https://cdn.freshtrack.ai/images/550e8400.jpg",
  "freshness": { /* same as /predict */ },
  "quality": { /* same as /predict */ },
  "shelf_life": { /* same as /predict */ },
  "heatmap_url": "https://cdn.freshtrack.ai/heatmaps/550e8400.jpg",
  "user_feedback": {
    "rating": 5,
    "comment": "Very accurate prediction",
    "actual_shelf_life": 6.5
  }
}
```

---

#### 12.2.6 POST /feedback

**Description**: Submit user feedback on prediction accuracy

**Request**:
```json
{
  "prediction_id": "550e8400-e29b-41d4-a716-446655440000",
  "rating": 5,
  "actual_shelf_life": 6.5,
  "comment": "Spot on!",
  "correctness": {
    "freshness": true,
    "quality": true
  }
}
```

**Response (200 OK)**:
```json
{
  "feedback_id": "feedback_abc123",
  "message": "Thank you for your feedback!",
  "prediction_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### 12.3 WebSocket API (Future)

**Endpoint**: `wss://api.freshtrack.ai/v1/stream`

**Use Case**: Real-time video stream analysis

```javascript
const ws = new WebSocket('wss://api.freshtrack.ai/v1/stream');

ws.onopen = () => {
  // Send frame
  ws.send(JSON.stringify({
    type: 'frame',
    data: base64EncodedImage
  }));
};

ws.onmessage = (event) => {
  const prediction = JSON.parse(event.data);
  console.log(prediction);
};
```

---

### 12.4 SDK Support

#### Python SDK
```python
from freshtrack import FreshTrackClient

client = FreshTrackClient(api_key='your_api_key')

# Single prediction
result = client.predict('apple.jpg', return_heatmap=True)
print(result.freshness.class_name)  # "Fresh"
print(result.shelf_life.days)  # 6.2

# Batch prediction
results = client.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
for r in results:
    print(f"{r.filename}: {r.freshness.class_name}")
```

#### JavaScript SDK
```javascript
import { FreshTrackClient } from 'freshtrack-sdk';

const client = new FreshTrackClient({ apiKey: 'your_api_key' });

// Single prediction
const result = await client.predict(imageFile);
console.log(result.freshness.class);  // "Fresh"
console.log(result.shelfLife.days);  // 6.2
```

---

## 13. User Interface Requirements

### 13.1 Web Application (Streamlit)

#### 13.1.1 Page Structure

**Home Page**:
```
┌──────────────────────────────────────┐
│  🍎 FreshTrack AI                    │
│  Intelligent Fruit Quality Assessment│
├──────────────────────────────────────┤
│                                      │
│  [Upload Image] [Use Camera]         │
│                                      │
│  ┌────────────────────────────────┐ │
│  │                                │ │
│  │     [Drag & Drop Zone]         │ │
│  │                                │ │
│  └────────────────────────────────┘ │
│                                      │
│  [Analyze]                           │
│                                      │
└──────────────────────────────────────┘
```

**Results Page**:
```
┌──────────────────────────────────────┐
│  🍎 FreshTrack AI - Results          │
├────────────┬─────────────────────────┤
│            │  Freshness: Fresh       │
│            │  Quality: Grade A       │
│   Image    │  Shelf Life: 6.2 days   │
│            │                         │
│  [Photo]   │  Confidence: 92%        │
│            │                         │
│            │  [View Heatmap]         │
├────────────┴─────────────────────────┤
│  Recommendations:                    │
│  • Store in cool, dry place          │
│  • Consume within 6 days             │
│                                      │
│  [New Analysis] [Export PDF]         │
└──────────────────────────────────────┘
```

---

#### 13.1.2 UI Components

**Component 1: Upload Widget**
```python
import streamlit as st

uploaded_file = st.file_uploader(
    "Upload fruit image",
    type=['jpg', 'jpeg', 'png'],
    help="Max file size: 5MB"
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
```

**Component 2: Camera Capture**
```python
camera_image = st.camera_input("Take a picture")

if camera_image:
    # Process camera image
    pass
```

**Component 3: Results Display**
```python
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Freshness",
        value=result['freshness']['class'],
        delta=f"{result['freshness']['confidence']*100:.1f}% confidence"
    )

with col2:
    st.metric(
        label="Quality",
        value=result['quality']['grade']
    )

with col3:
    st.metric(
        label="Shelf Life",
        value=f"{result['shelf_life']['days']:.1f} days",
        delta=f"±{result['shelf_life']['prediction_std']:.1f}"
    )
```

**Component 4: Heatmap Visualization**
```python
st.subheader("Explainability - Grad-CAM")

tab1, tab2, tab3 = st.tabs(["Original", "Heatmap", "Overlay"])

with tab1:
    st.image(original_image, use_column_width=True)

with tab2:
    st.image(heatmap_image, use_column_width=True)

with tab3:
    st.image(overlay_image, use_column_width=True)
    st.caption("Red regions indicate areas influencing the prediction")
```

**Component 5: History**
```python
st.subheader("Recent Predictions")

history_df = pd.DataFrame(prediction_history)
st.dataframe(
    history_df[['timestamp', 'freshness', 'quality', 'shelf_life']],
    use_container_width=True
)
```

---

### 13.2 Mobile App (Future - React Native)

**Screens**:
1. **Camera Screen**: Live viewfinder with capture button
2. **Processing Screen**: Loading animation
3. **Results Screen**: Card-based results with swipe gestures
4. **History Screen**: Scrollable list of past predictions
5. **Settings Screen**: Model selection, quality settings

**Offline Mode**:
- Download TFLite model (~6MB)
- Local inference (no internet required)
- Sync results when online

---

### 13.3 Dashboard (Grafana)

**Page 1: System Health**
- Request rate (req/min)
- Latency percentiles (P50, P95, P99)
- Error rate
- CPU/Memory usage

**Page 2: Model Performance**
- Predictions per day
- Confidence distribution
- Freshness class distribution
- Average shelf-life predictions

**Page 3: Data Quality**
- Input image quality scores
- Rejected images (too blurry, wrong format)
- User feedback ratings

**Page 4: Business Metrics**
- Daily active users
- API usage by client
- Cost per prediction

---

## 14. MLOps Pipeline

### 14.1 Pipeline Overview

```
┌───────────┐     ┌────────────┐     ┌───────────┐
│   Data    │────▶│  Training  │────▶│   Model   │
│ Ingestion │     │  Pipeline  │     │ Validation│
└───────────┘     └────────────┘     └─────┬─────┘
      │                  │                  │
      │                  ▼                  │
      │           ┌────────────┐            │
      │           │ Experiment │            │
      │           │  Tracking  │            │
      │           │   (W&B)    │            │
      │           └────────────┘            │
      │                                     ▼
      │                              ┌────────────┐
      │                              │   Model    │
      │                              │  Registry  │
      │                              └─────┬──────┘
      │                                    │
      │                                    ▼
      │                              ┌────────────┐
      │                              │   CI/CD    │
      │                              │  Pipeline  │
      │                              └─────┬──────┘
      │                                    │
      │                                    ▼
      │                              ┌────────────┐
      ▼                              │ Production │
┌───────────┐                        │ Deployment │
│Monitoring │◀───────────────────────┤            │
│  & Drift  │                        └────────────┘
└───────────┘
```

---

### 14.2 Data Pipeline

#### 14.2.1 Data Ingestion

```python
# scripts/ingest_data.py

import dvc.api

def ingest_new_batch(batch_path, batch_metadata):
    """
    Ingest new batch of fruit images
    """
    # Validate images
    valid_images = validate_batch(batch_path)
    
    # Upload to DVC
    with dvc.api.open(
        f'data/raw/batch_{batch_metadata["id"]}',
        mode='w',
        remote='s3storage'
    ) as f:
        upload_images(valid_images, f)
    
    # Update metadata
    update_dataset_metadata(batch_metadata)
    
    # Version control
    os.system('dvc add data/raw')
    os.system('git add data/raw.dvc')
    os.system(f'git commit -m "feat: add batch {batch_metadata["id"]}"')
    os.system('dvc push')
    
    return len(valid_images)
```

---

#### 14.2.2 Data Validation

```python
import great_expectations as ge

# Define data expectations
data_context = ge.data_context.DataContext()

# Expectation suite
suite = data_context.create_expectation_suite("fruit_images_v1")

# Add expectations
batch = data_context.get_batch(batch_kwargs, suite)

batch.expect_column_values_to_be_in_set(
    column="freshness",
    value_set=["Fresh", "Semi-ripe", "Overripe", "Rotten"]
)

batch.expect_column_values_to_be_between(
    column="shelf_life_days",
    min_value=0,
    max_value=30
)

batch.expect_column_values_to_match_regex(
    column="image_path",
    regex=r".*\.(jpg|png)$"
)

# Run validation
results = data_context.run_validation_operator(
    "action_list_operator",
    assets_to_validate=[batch]
)
```

---

### 14.3 Training Pipeline

#### 14.3.1 PyTorch Lightning Trainer

```python
# train.py

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# Initialize logger
wandb_logger = WandbLogger(
    project="freshtrack-ai",
    name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config={
        "architecture": "EfficientNet-B0",
        "batch_size": 32,
        "learning_rate": 1e-4,
        # ... other hyperparameters
    }
)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath="models/checkpoints",
    filename="freshtrack_{epoch:02d}_{val_loss:.2f}",
    save_top_k=3,
    monitor="val_loss",
    mode="min"
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=7,
    mode="min",
    verbose=True
)

# Trainer
trainer = pl.Trainer(
    max_epochs=50,
    gpus=1,
    precision=16,  # Mixed precision
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=wandb_logger,
    log_every_n_steps=10,
    val_check_interval=0.5  # Validate twice per epoch
)

# Train
model = FreshTrackModel()
datamodule = FruitDataModule(batch_size=32)

trainer.fit(model, datamodule)

# Test
trainer.test(model, datamodule)
```

---

#### 14.3.2 Hyperparameter Tuning

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    
    # Create model with suggested hyperparameters
    model = FreshTrackModel(dropout=dropout)
    
    # Train
    trainer = pl.Trainer(
        max_epochs=20,
        gpus=1,
        logger=False,
        enable_checkpointing=False
    )
    
    datamodule = FruitDataModule(batch_size=batch_size)
    trainer.fit(model, datamodule)
    
    # Return validation metric
    return trainer.callback_metrics["val_loss"].item()

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)
```

---

### 14.4 Model Registry

```python
# model_registry.py

import mlflow
import mlflow.pytorch

class ModelRegistry:
    def __init__(self, tracking_uri):
        mlflow.set_tracking_uri(tracking_uri)
    
    def register_model(self, model, model_name, metrics, artifacts):
        """
        Register model in MLflow
        """
        with mlflow.start_run():
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log artifacts
            for artifact_path in artifacts:
                mlflow.log_artifact(artifact_path)
            
            # Log model
            mlflow.pytorch.log_model(
                model,
                "model",
                registered_model_name=model_name
            )
            
            # Tag version
            run_id = mlflow.active_run().info.run_id
            mlflow.set_tag("version", model_name)
            mlflow.set_tag("stage", "staging")
            
        return run_id
    
    def promote_model(self, model_name, version, stage):
        """
        Promote model to production
        """
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage  # "Staging" or "Production"
        )
    
    def load_production_model(self, model_name):
        """
        Load current production model
        """
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pytorch.load_model(model_uri)
        return model
```

---

### 14.5 CI/CD Pipeline

#### 14.5.1 GitHub Actions Workflow

```yaml
# .github/workflows/train_and_deploy.yml

name: Train and Deploy

on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'data/**'
      - 'requirements.txt'
  workflow_dispatch:

jobs:
  # Job 1: Code Quality
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install flake8 black isort
      
      - name: Lint with flake8
        run: flake8 src/ --max-line-length=120
      
      - name: Check formatting
        run: black --check src/
      
      - name: Check import order
        run: isort --check-only src/
  
  # Job 2: Unit Tests
  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
  
  # Job 3: Train Model (if data changed)
  train:
    runs-on: ubuntu-latest
    needs: test
    if: contains(github.event.head_commit.message, '[train]')
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Pull data from DVC
        run: |
          dvc remote add -d storage s3://freshtrack-data
          dvc pull
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      
      - name: Train model
        run: python src/train.py --config configs/train_config.yaml
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
      
      - name: Validate model
        run: python src/validate.py --model-path models/best.pth
      
      - name: Upload model artifact
        uses: actions/upload-artifact@v3
        with:
          name: trained-model
          path: models/best.pth
  
  # Job 4: Build Docker Image
  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v3
      
      - name: Log in to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            ${{ secrets.DOCKER_USERNAME }}/freshtrack-ai:latest
            ${{ secrets.DOCKER_USERNAME }}/freshtrack-ai:${{ github.sha }}
  
  # Job 5: Deploy to Staging
  deploy-staging:
    runs-on: ubuntu-latest
    needs: build
    environment:
      name: staging
      url: https://staging.freshtrack.ai
    steps:
      - name: Deploy to Render
        run: |
          curl -X POST https://api.render.com/deploy/srv-xxx \
            -H "Authorization: Bearer ${{ secrets.RENDER_API_KEY }}"
  
  # Job 6: Integration Tests
  integration-test:
    runs-on: ubuntu-latest
    needs: deploy-staging
    steps:
      - uses: actions/checkout@v3
      
      - name: Run API tests
        run: |
          pip install pytest requests
          pytest tests/integration/ --base-url=https://staging.freshtrack.ai
  
  # Job 7: Deploy to Production (manual approval)
  deploy-production:
    runs-on: ubuntu-latest
    needs: integration-test
    environment:
      name: production
      url: https://api.freshtrack.ai
    steps:
      - name: Deploy to production
        run: |
          curl -X POST https://api.render.com/deploy/srv-yyy \
            -H "Authorization: Bearer ${{ secrets.RENDER_API_KEY }}"
      
      - name: Notify Slack
        run: |
          curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
            -H 'Content-Type: application/json' \
            -d '{"text":"🚀 FreshTrack AI v${{ github.sha }} deployed to production!"}'
```

---

### 14.6 Monitoring & Alerts

#### 14.6.1 Evidently Monitoring

```python
# monitoring/drift_detection.py

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns

class ModelMonitor:
    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.column_mapping = ColumnMapping(
            prediction='prediction',
            target='target'
        )
    
    def detect_drift(self, current_data):
        """
        Detect data and prediction drift
        """
        # Create report
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset()
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Extract drift metrics
        result = report.as_dict()
        drift_share = result['metrics'][0]['result']['drift_share']
        
        # Alert if drift > threshold
        if drift_share > 0.3:
            self.send_alert(
                f"Data drift detected: {drift_share*100:.1f}% of features drifted"
            )
        
        # Save report
        report.save_html("reports/drift_report.html")
        
        return drift_share
    
    def send_alert(self, message):
        """
        Send alert via Slack/email
        """
        # Slack webhook
        import requests
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        requests.post(webhook_url, json={"text": f"⚠️ {message}"})
```

---

#### 14.6.2 Prometheus Metrics

```python
# api/metrics.py

from prometheus_client import Counter, Histogram, Gauge

# Define metrics
prediction_counter = Counter(
    'freshtrack_predictions_total',
    'Total predictions made',
    ['freshness_class', 'quality_grade']
)

prediction_latency = Histogram(
    'freshtrack_prediction_latency_seconds',
    'Prediction latency',
    buckets=[0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
)

model_confidence = Histogram(
    'freshtrack_prediction_confidence',
    'Model confidence scores',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

active_requests = Gauge(
    'freshtrack_active_requests',
    'Number of active requests'
)

# Usage in FastAPI
@app.post("/predict")
async def predict(file: UploadFile):
    active_requests.inc()
    
    with prediction_latency.time():
        # Inference
        result = model.predict(image)
    
    prediction_counter.labels(
        freshness_class=result['freshness']['class'],
        quality_grade=result['quality']['grade']
    ).inc()
    
    model_confidence.observe(result['freshness']['confidence'])
    
    active_requests.dec()
    
    return result
```

---

#### 14.6.3 Grafana Dashboards

**Dashboard 1: API Performance**
```yaml
# grafana/dashboards/api_performance.json
{
  "dashboard": {
    "title": "FreshTrack API Performance",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(freshtrack_predictions_total[5m])"
          }
        ]
      },
      {
        "title": "Latency Percentiles",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, freshtrack_prediction_latency_seconds_bucket)",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, freshtrack_prediction_latency_seconds_bucket)",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, freshtrack_prediction_latency_seconds_bucket)",
            "legendFormat": "P99"
          }
        ]
      }
    ]
  }
}
```

---

## 15. Testing Strategy

### 15.1 Unit Tests

```python
# tests/test_model.py

import pytest
import torch
from src.models.freshtrack_model import FreshTrackModel

def test_model_forward_pass():
    model = FreshTrackModel()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    freshness, quality, shelf_life, rotation = model(dummy_input)
    
    assert freshness.shape == (1, 4)
    assert quality.shape == (1, 3)
    assert shelf_life.shape == (1, 1)
    assert rotation.shape == (1, 4)

def test_model_output_range():
    model = FreshTrackModel()
    dummy_input = torch.randn(1, 3, 224, 224)
    
    _, _, shelf_life, _ = model(dummy_input)
    
    # Shelf life should be positive
    assert torch.all(shelf_life >= 0)

# tests/test_preprocessing.py

def test_image_preprocessing():
    from src.data.preprocessing import preprocess_image
    
    # Load test image
    image = cv2.imread("tests/fixtures/apple.jpg")
    
    # Preprocess
    tensor = preprocess_image(image)
    
    # Check output shape and dtype
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    
    # Check normalization
    assert -3 < tensor.min() < 0
    assert 0 < tensor.max() < 3
```

---

### 15.2 Integration Tests

```python
# tests/integration/test_api.py

import pytest
import requests

BASE_URL = "https://staging.freshtrack.ai"

def test_predict_endpoint():
    # Upload test image
    with open("tests/fixtures/apple.jpg", "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/api/v1/predict", files=files)
    
    assert response.status_code == 200
    
    result = response.json()
    assert "prediction_id" in result
    assert "freshness" in result
    assert "quality" in result
    assert "shelf_life" in result

def test_invalid_image_format():
    with open("tests/fixtures/invalid.txt", "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/api/v1/predict", files=files)
    
    assert response.status_code == 400
    assert "INVALID_IMAGE_FORMAT" in response.json()["error"]

def test_rate_limiting():
    # Send 101 requests quickly
    responses = []
    for i in range(101):
        with open("tests/fixtures/apple.jpg", "rb") as f:
            files = {"file": f}
            response = requests.post(f"{BASE_URL}/api/v1/predict", files=files)
            responses.append(response)
    
    # Should get rate limited
    assert any(r.status_code == 429 for r in responses)
```

---

### 15.3 Model Tests

```python
# tests/test_model_performance.py

def test_model_accuracy():
    model = load_model("models/best.pth")
    test_loader = create_test_loader()
    
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        outputs = model(images)
        freshness_pred = outputs[0].argmax(dim=1)
        
        correct += (freshness_pred == labels['freshness']).sum().item()
        total += labels['freshness'].size(0)
    
    accuracy = correct / total
    
    # Assert minimum accuracy threshold
    assert accuracy >= 0.90, f"Accuracy {accuracy:.2%} below threshold"

def test_shelf_life_mae():
    model = load_model("models/best.pth")
    test_loader = create_test_loader()
    
    predictions = []
    actuals = []
    
    for images, labels in test_loader:
        outputs = model(images)
        shelf_life_pred = outputs[2]
        
        predictions.extend(shelf_life_pred.squeeze().tolist())
        actuals.extend(labels['shelf_life'].tolist())
    
    mae = mean_absolute_error(actuals, predictions)
    
    # Assert MAE threshold
    assert mae <= 1.0, f"MAE {mae:.2f} days exceeds threshold"
```

---

### 15.4 Load Testing

```python
# tests/load/locustfile.py

from locust import HttpUser, task, between
import random

class FreshTrackUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def predict_fruit(self):
        # Select random test image
        image_path = random.choice([
            "tests/fixtures/apple.jpg",
            "tests/fixtures/banana.jpg",
            "tests/fixtures/orange.jpg"
        ])
        
        with open(image_path, "rb") as f:
            files = {"file": f}
            self.client.post("/api/v1/predict", files=files)
    
    @task(2)
    def health_check(self):
        self.client.get("/health")
```

**Run load test**:
```bash
locust -f tests/load/locustfile.py --host=https://api.freshtrack.ai
```

**Acceptance Criteria**:
- Support 100 concurrent users
- P95 latency <500ms under load
- 0% error rate under normal conditions
- Graceful degradation under 2x expected load

---

## 16. Security & Compliance

### 16.1 Security Requirements

| Requirement | Implementation |
|-------------|----------------|
| **Authentication** | JWT tokens for API access |
| **Rate Limiting** | 100 req/hour (free), 10k/hour (premium) |
| **Input Validation** | File type, size, content validation |
| **Data Encryption** | TLS 1.3 for data in transit, AES-256 for data at rest |
| **API Keys** | Hashed storage, rotation every 90 days |
| **Secrets Management** | AWS Secrets Manager / HashiCorp Vault |
| **CORS** | Whitelist specific domains |
| **Logging** | No PII in logs, centralized log management |

---

### 16.2 Privacy & Compliance

**Data Handling**:
- User-uploaded images stored for 30 days (configurable)
- Option to delete images immediately after processing
- No facial recognition or person detection
- GDPR-compliant data deletion requests

**Compliance Standards**:
- GDPR (General Data Protection Regulation)
- SOC 2 Type II (future)
- ISO 27001 (future)

---

### 16.3 Security Implementation

```python
# api/security.py

from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader
import jwt

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify API key
    """
    # Check against database (hashed)
    if not is_valid_api_key(api_key):
        raise HTTPException(
            status_code=403,
            detail="Invalid or expired API key"
        )
    
    return api_key

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/predict")
@limiter.limit("100/hour")
async def predict(
    file: UploadFile,
    api_key: str = Security(verify_api_key)
):
    # Process request
    pass
```

---

## 17. Deployment Plan

### 17.1 Deployment Architecture

**Option 1: Render (Simple)**
```
User → Render Load Balancer → FastAPI Container → PostgreSQL (Render)
                                      ↓
                                   S3 (AWS)
```

**Option 2: AWS (Production-grade)**
```
User → CloudFront (CDN) → ALB → ECS Fargate (Multi-AZ)
                                      ↓
                              RDS PostgreSQL (Multi-AZ)
                                      ↓
                              S3 (Models + Images)
                                      ↓
                         CloudWatch (Monitoring)
```

---

### 17.2 Deployment Steps

#### Phase 1: Local Development
```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run locally
uvicorn api.main:app --reload

# 3. Test endpoints
curl -X POST http://localhost:8000/predict -F "file=@test.jpg"
```

---

#### Phase 2: Staging Deployment (Render)
```bash
# 1. Create Dockerfile
docker build -t freshtrack-api .

# 2. Test locally
docker run -p 8000:8000 freshtrack-api

# 3. Push to Render
git push origin main  # Auto-deploys on Render

# 4. Run smoke tests
pytest tests/integration/ --base-url=https://staging.freshtrack.ai
```

---

#### Phase 3: Production Deployment (AWS)
```bash
# 1. Build and tag image
docker build -t freshtrack-api:v2.1.3 .

# 2. Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag freshtrack-api:v2.1.3 <account-id>.dkr.ecr.us-east-1.amazonaws.com/freshtrack-api:v2.1.3
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/freshtrack-api:v2.1.3

# 3. Update ECS task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# 4. Update service (blue-green deployment)
aws ecs update-service \
  --cluster freshtrack-cluster \
  --service freshtrack-service \
  --task-definition freshtrack-api:v2.1.3

# 5. Monitor deployment
aws ecs wait services-stable \
  --cluster freshtrack-cluster \
  --services freshtrack-service
```

---

### 17.3 Rollback Strategy

```bash
# If deployment fails, rollback to previous version
aws ecs update-service \
  --cluster freshtrack-cluster \
  --service freshtrack-service \
  --task-definition freshtrack-api:v2.1.2 \
  --force-new-deployment
```

---

### 17.4 Zero-Downtime Deployment

**Blue-Green Deployment**:
1. Deploy new version (Green) alongside old (Blue)
2. Run health checks on Green
3. Route 10% traffic to Green (canary)
4. Monitor metrics for 30 minutes
5. If stable, route 100% to Green
6. Decommission Blue after 24 hours

---

## 18. Monitoring & Observability

### 18.1 Metrics to Track

#### System Metrics
- **CPU Usage**: Target <70% average
- **Memory Usage**: Target <80%
- **Disk I/O**: Monitor for bottlenecks
- **Network Traffic**: Inbound/outbound bandwidth

#### Application Metrics
- **Request Rate**: Requests per minute
- **Latency**: P50, P95, P99 percentiles
- **Error Rate**: 4xx and 5xx errors
- **Availability**: Uptime percentage

#### ML Metrics
- **Prediction Distribution**: Freshness classes, quality grades
- **Confidence Scores**: Average confidence over time
- **Model Version**: Which model is serving requests
- **Inference Time**: Model forward pass duration

#### Business Metrics
- **Daily Active Users (DAU)**
- **Predictions per User**
- **API Usage by Client**
- **User Feedback Ratings**

---

### 18.2 Alerting Rules

```yaml
# prometheus/alerts.yml

groups:
  - name: freshtrack_alerts
    rules:
      # High latency
      - alert: HighLatency
        expr: histogram_quantile(0.95, freshtrack_prediction_latency_seconds_bucket) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency detected"
          description: "P95 latency is {{ $value }}s (threshold: 0.5s)"
      
      # High error rate
      - alert: HighErrorRate
        expr: rate(freshtrack_predictions_failed_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} (threshold: 1%)"
      
      # Low confidence
      - alert: LowModelConfidence
        expr: avg_over_time(freshtrack_prediction_confidence[1h]) < 0.7
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Model confidence degrading"
          description: "Average confidence is {{ $value }} (threshold: 0.7)"
      
      # Data drift
      - alert: DataDrift
        expr: freshtrack_data_drift_detected == 1
        labels:
          severity: warning
        annotations:
          summary: "Data drift detected"
          description: "Input distribution has shifted significantly"
```

---

### 18.3 Logging Strategy

```python
# utils/logging.py

import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'prediction_id'):
            log_data['prediction_id'] = record.prediction_id
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        
        return json.dumps(log_data)

# Configure logger
logger = logging.getLogger("freshtrack")
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info(
    "Prediction completed",
    extra={
        'prediction_id': result['prediction_id'],
        'freshness': result['freshness']['class'],
        'inference_time_ms': result['metadata']['inference_time_ms']
    }
)
```

---

## 19. Timeline & Milestones

### Week 1: Foundation (Days 1-7)

**Days 1-2: Setup & Data**
- [ ] Initialize Git repository
- [ ] Setup Python environment
- [ ] Download Kaggle datasets
- [ ] Implement data loader
- [ ] Create train/val/test splits

**Days 3-5: Baseline Model**
- [ ] Implement FreshTrackModel architecture
- [ ] Train frozen backbone (5 epochs)
- [ ] Achieve 70%+ freshness accuracy
- [ ] Setup W&B logging

**Days 6-7: Fine-tuning**
- [ ] Unfreeze last 2 layers
- [ ] Train for 20 epochs
- [ ] Target: 85%+ accuracy, MAE <2 days
- [ ] Implement early stopping

**Deliverables**:
- ✅ Working baseline model
- ✅ Training logged in W&B
- ✅ Validation metrics documented

---

### Week 2: Advanced Features (Days 8-14)

**Days 8-9: Custom Dataset**
- [ ] Setup photo equipment
- [ ] Begin daily fruit photography
- [ ] Label with ground truth shelf life
- [ ] Add to training pipeline

**Days 10-12: Explainability**
- [ ] Implement Grad-CAM
- [ ] Generate heatmaps for test set
- [ ] Verify ROI highlighting
- [ ] Add LIME (optional)

**Days 13-14: Model Optimization**
- [ ] Quantize model (INT8)
- [ ] Export to ONNX
- [ ] Benchmark inference speed
- [ ] Compare accuracy before/after

**Deliverables**:
- ✅ Custom dataset collected
- ✅ Explainability working
- ✅ 3 model formats ready

---

### Week 3: MLOps Pipeline (Days 15-21)

**Days 15-16: API Development**
- [ ] Implement FastAPI endpoints
- [ ] Add input validation
- [ ] Add Prometheus metrics
- [ ] Write API tests

**Days 17-18: Containerization**
- [ ] Create Dockerfile
- [ ] Setup docker-compose
- [ ] Test locally
- [ ] Optimize image size

**Days 19-20: Deployment**
- [ ] Deploy to Render (staging)
- [ ] Configure environment variables
- [ ] Setup PostgreSQL
- [ ] Configure S3 storage

**Day 21: CI/CD**
- [ ] Setup GitHub Actions
- [ ] Configure automated tests
- [ ] Setup automated deployment
- [ ] Test full pipeline

**Deliverables**:
- ✅ API deployed and accessible
- ✅ CI/CD pipeline operational
- ✅ Monitoring dashboards live

---

### Week 4: Polish & Demo (Days 22-28)

**Days 22-23: Frontend**
- [ ] Build Streamlit app
- [ ] Add file upload
- [ ] Display results
- [ ] Show heatmaps

**Days 24-25: Monitoring**
- [ ] Setup Evidently
- [ ] Configure Grafana dashboards
- [ ] Setup alerts
- [ ] Test drift detection

**Days 26-28: Documentation**
- [ ] Write comprehensive README
- [ ] Create architecture diagrams
- [ ] Record demo video
- [ ] Prepare presentation

**Deliverables**:
- ✅ Full demo ready
- ✅ GitHub repo polished
- ✅ Presentation slides

---

### Milestone Checklist

| Milestone | Completion Criteria | Status |
|-----------|---------------------|--------|
| **M1: Data Ready** | 20k images, train/val/test split, DVC setup | ⬜ |
| **M2: Baseline Model** | 85%+ freshness accuracy, MAE <2 days | ⬜ |
| **M3: Custom Dataset** | 500+ timelapse images collected | ⬜ |
| **M4: Production Model** | 90%+ accuracy, MAE <1 day | ⬜ |
| **M5: API Deployed** | Live endpoint, <500ms latency | ⬜ |
| **M6: MLOps Pipeline** | CI/CD, monitoring, drift detection | ⬜ |
| **M7: Demo Ready** | Streamlit app, video, documentation | ⬜ |

---

## 20. Risk Analysis

### 20.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Insufficient training data** | Medium | High | Use aggressive data augmentation + transfer learning |
| **Poor model generalization** | Medium | High | Collect custom dataset, use domain-specific augmentation |
| **High inference latency** | Low | Medium | Model quantization, use faster backbone |
| **API scalability issues** | Medium | High | Horizontal scaling, load balancing, caching |
| **Data drift in production** | High | Medium | Continuous monitoring, automated retraining |
| **Model serving failures** | Low | Critical | Health checks, auto-restart, fallback model |

---

### 20.2 Project Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Timeline slippage** | Medium | Medium | Prioritize P0 features, cut optional features |
| **Hardware constraints** | Low | Medium | Use Colab/Kaggle GPUs, optimize batch size |
| **Dataset quality issues** | High | High | Implement rigorous validation, manual review |
| **Scope creep** | Medium | Medium | Stick to PRD, defer v2.0 features |

---

### 20.3 Mitigation Strategies

#### For Data Quality Issues
1. **Automated Validation**: Run Great Expectations on all batches
2. **Manual Review**: Sample 10% of new data for human verification
3. **Outlier Detection**: Flag suspicious labels (e.g., Fresh + shelf_life=0)

#### For Model Performance Issues
1. **Incremental Training**: Start with frozen backbone, gradually unfreeze
2. **Ensemble Models**: Train 3-5 models, average predictions
3. **Confidence Thresholding**: Flag low-confidence predictions for review

#### For Deployment Issues
1. **Staging Environment**: Test thoroughly before production
2. **Blue-Green Deployment**: Zero-downtime rollout
3. **Rollback Plan**: Keep previous version running for 24 hours

---

## 21. Dependencies & Assumptions

### 21.1 External Dependencies

| Dependency | Purpose | Fallback |
|------------|---------|----------|
| **Kaggle Datasets** | Training data | Use alternative datasets (ImageNet fruits) |
| **AWS S3** | Model and image storage | Use MinIO (self-hosted) |
| **Render/AWS** | Deployment | Deploy locally or on GCP |
| **Weights & Biases** | Experiment tracking | Use MLflow (self-hosted) |
| **PyTorch** | Deep learning framework | TensorFlow (requires refactor) |

---

### 21.2 Assumptions

1. **Data Availability**: Kaggle datasets remain accessible
2. **Compute Resources**: Access to GPU for training (Colab/Kaggle)
3. **Internet Connectivity**: Required for cloud services
4. **User Input**: Images are reasonably well-lit and focused
5. **Fruit Types**: Limited to 5-10 common fruits initially
6. **Shelf Life Definition**: "Days until inedible" (not "days until peak ripeness")
7. **Deployment Budget**: Free tiers of Render/AWS sufficient for MVP

---

### 21.3 Constraints

| Constraint | Implication |
|------------|-------------|
| **Budget: $0-50/month** | Use free tiers, optimize for cost |
| **Timeline: 4 weeks** | Focus on MVP, defer advanced features |
| **Team Size: 1 person** | Leverage AI tools (Claude, Copilot) heavily |
| **No custom hardware** | Software-only solution, no sensors |
| **No labeled shelf-life data** | Must create custom timelapse dataset |

---

## 22. Future Roadmap

### Version 2.0 (Q2 2026)

**Features**:
- [ ] Multi-fruit support (10+ fruit types)
- [ ] Mobile app (React Native + TFLite)
- [ ] Real-time video stream analysis
- [ ] Batch processing optimization (100 images in <10s)
- [ ] User accounts and history tracking

**ML Improvements**:
- [ ] Active learning pipeline
- [ ] Model distillation (smaller models)
- [ ] Self-supervised pretraining on fruit images
- [ ] Uncertainty quantification (Bayesian NN)

**MLOps Enhancements**:
- [ ] A/B testing framework
- [ ] Automated hyperparameter tuning
- [ ] Federated learning (privacy-preserving)
- [ ] Edge deployment (Raspberry Pi, NVIDIA Jetson)

---

### Version 3.0 (Q4 2026)

**Advanced Features**:
- [ ] 3D imaging support (depth cameras)
- [ ] Multi-modal inputs (combine vision + temperature sensor)
- [ ] Anomaly detection (unknown defects)
- [ ] Predictive analytics (forecast spoilage trends)

**Enterprise Features**:
- [ ] Integration with POS systems
- [ ] Automated pricing recommendations
- [ ] Supply chain optimization
- [ ] Regulatory compliance reporting

---

## 23. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|------------|
| **MAE** | Mean Absolute Error - average prediction error in days |
| **Grad-CAM** | Gradient-weighted Class Activation Mapping - visualization technique |
| **DVC** | Data Version Control - Git for datasets |
| **ECE** | Expected Calibration Error - measures probability calibration |
| **P99 Latency** | 99th percentile latency - worst case for 99% of requests |
| **ONNX** | Open Neural Network Exchange - model interchange format |

---

### Appendix B: References

**Datasets**:
1. Kaggle Fruit Recognition Dataset: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
2. Fruit Quality Detection: https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-quality-detection

**Papers**:
1. EfficientNet: Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (2019)
2. Grad-CAM: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (2017)
3. Multi-Task Learning: Ruder, "An Overview of Multi-Task Learning in Deep Neural Networks" (2017)

**Tools**:
1. PyTorch: https://pytorch.org
2. FastAPI: https://fastapi.tiangolo.com
3. Weights & Biases: https://wandb.ai
4. DVC: https://dvc.org
5. Evidently: https://evidentlyai.com

---

### Appendix C: Contact & Support

**Project Owner**: [Your Name]  
**Email**: [your-email@example.com]  
**GitHub**: https://github.com/[username]/freshtrack-ai  
**Demo**: https://freshtrack.streamlit.app  
**API**: https://api.freshtrack.ai  

---

### Appendix D: Change Log

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2026-02-11 | Complete PRD with MLOps, deployment, monitoring |
| 1.0 | 2026-01-15 | Initial draft with core features |

---

## 🎯 Next Steps

**To get started**:

1. **Review this PRD** with stakeholders (professor, advisor)
2. **Set up GitHub repo** using the file structure in Section 9.3
3. **Start with Week 1 tasks** (data preparation)
4. **Schedule weekly check-ins** to track progress
5. **Use this PRD as your single source of truth**

**Questions? Let's build this! 🚀**
