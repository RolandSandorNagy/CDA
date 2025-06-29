# 🚀 API Guide: Collateral Description Agent

This document explains how to build and use the containerized API.

---

## 🐳 Build the Docker Image

```bash
docker build -t collateral-agent .
```

---

## ▶️ Run the API Container

Make sure your dataset is in `./datasets/`.

```bash
docker run -p 8000:8000 -v "$(pwd)/datasets:/app/datasets" collateral-agent
```

The API will be available at `http://localhost:8000`.

---

## 🧪 Test the API

Use `curl` or Postman to call the API:

```bash
curl -X POST http://localhost:8000/generate_report \
  -H "Content-Type: application/json" \
  -d '{"asset_id": "157515_v2"}'
```

If successful, you'll receive:
- `status: success`
- `output`: generated markdown report text
- The report is also saved to the appropriate asset folder

---

## 📂 Mounting Local Data

The container expects your dataset to be mounted under `/app/datasets`.  
For example:
```bash
-v "$(pwd)/datasets:/app/datasets"
```

---

## 📎 Included Commands

You can also use the provided helper files:

### run.sh

```bash
chmod +x run.sh
./run.sh
```

### Makefile

```bash
make build
make run
make test
```