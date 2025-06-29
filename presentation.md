# 📊 Collateral Description Agent – Project Presentation

---

## 🎯 Project Goal

- **Objective**: Automate the process of describing physical collateral (vehicles, machines, buildings) from photos.
- **Use Case**: Helps banks and evaluators generate structured markdown appraisal reports from visual evidence.
- **Key Requirement**: Extract value-relevant visual features and generate detailed natural language reports.

---

## ⚙️ System Architecture

**Processing Pipeline**:

1. 📷 **Image Input**
   - Asset photo sets (interior, exterior, docs)

2. 🧠 **Caption Generation**
   - BLIP-base model describes image contents

3. 🔍 **OCR Extraction**
   - Tesseract identifies text in documents

4. 🏷️ **Domain Classification**
   - Rule-based categorization (Vehicle, Machine, etc.)

5. 📝 **Prompt Generation**
   - Qwen-compatible markdown template with image/OCR context

6. 🤖 **Description Generation**
   - Qwen-32B-AWQ endpoint returns full appraisal report

---

## 🧰 Models & Tools

| Component           | Choice                                | Justification                        |
|--------------------|----------------------------------------|--------------------------------------|
| Captioning         | `blip-image-captioning-base`           | Lightweight, accurate, CPU-friendly |
| OCR                | `pytesseract`                          | Local, fast, supports Hungarian text |
| Classification     | Keyword-based heuristic rules          | Reliable for asset types            |
| LLM                | `Qwen-32B-AWQ` (ELTE-hosted)           | Follows markdown instruction well    |

> ⚠️ InstructBLIP was replaced due to hardware limits.

---

## 🖼️ Sample Output

- **Asset**: `157515_v2` (DAF XF 480 FT tractor unit)
- **Image Caption**: "A white DAF truck photographed from the front"
- **OCR**: "439014 km, DAF VIN, Euro 6"
- **Classification**: `Truck`
- **Generated Report**:
  - Identification
  - Inspection methods
  - Valuation
  - Accessories
  - Estimated values (HUF)

✅ Output: Professional-grade markdown report

---

## ✅ Highlights & Next Steps

- Modular, inspectable pipeline
- Accurate OCR + caption integration
- Runs fully offline (no 3rd-party uploads)
- Easy to extend with manual overrides or pricing API

🛠️ Next Steps:
- Add containerized REST API (WIP)
- Optimize report filtering for length/quality
- Prepare evaluation interface

---

## 🙌 Thank You!

**Author**: Roland Sándor Nagy
**Institution**: ELTE MSc in AI  
**Model Host**: ELTE GenAI endpoint (Qwen-32B)  
**Contact**: roland.s.nagy@gmail.com | https://github.com/RolandSandorNagy