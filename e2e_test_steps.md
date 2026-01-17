# End-to-End Test Steps

## Test Date: 2026-01-17

## Prerequisites
- [x] Environment set up with `uv sync`
- [x] Authenticated to target host (localhost)

## Test Steps

### 1. Create New CIFAR-10 Catalog
```bash
uv run load-cifar10 --hostname localhost --create-catalog cifar-10 --num-images 10000
```

**Result:**
- Catalog ID: **62**
- Schema: **cifar-10**
- Images loaded: 10,000 (5,000 training + 5,000 testing)

**Datasets created:**
| Dataset | RID | Description |
|---------|-----|-------------|
| Complete | 28CT | All 10,000 images |
| Split | 28D4 | Parent of Training + Testing |
| Training | 28DC | 5,000 training images |
| Testing | 28DP | 5,000 testing images (unlabeled) |
| Small_Split | 28EA | Parent of small datasets |
| Small_Training | 28EJ | 500 training images |
| Small_Testing | 28EW | 500 testing images |
| Labeled_Split | 28FG | 5,000 images with labels |
| Labeled_Training | 28FT | 4,000 labeled training images |
| Labeled_Testing | 28G4 | 1,000 labeled test images |
| Small_Labeled_Split | 28GR | 500 labeled images |
| Small_Labeled_Training | 28H2 | 400 labeled training images |
| Small_Labeled_Testing | 28HC | 100 labeled test images |

### 2. Update Configuration Files
Updated `src/configs/deriva.py`:
- Changed `catalog_id` from 45 to **62**

Updated `src/configs/datasets.py`:
- Replaced all old RIDs with new catalog 62 RIDs
- Added small labeled dataset configurations

### 3. Run Model Training
```bash
# Commands to run:
uv run deriva-ml-run +experiment=cifar10_quick
```

### 4. Verification
```bash
# Commands to run:
```

## Results
- Status: In Progress
- Catalog created: ✅
- Configuration updated: ✅
- Model training: Pending
- Verification: Pending

