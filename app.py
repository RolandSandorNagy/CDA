def main(assetID=False, manualDescription=False):
    if assetID:
        pass
    else:
        assetID = '157515_v2'
        #assetID = '158220'
        #assetID = '159347'
        #assetID = '161552_v2'

        #assetID = '162390'
        #assetID = '162749'
        #assetID = '165484'
        #assetID = '165671'
        #assetID = '165720'
        #assetID = '165968'
        #assetID = '169883'

        #assetID = 'CZ LOKO'


    # STEP 1: Install dependencies
    #!pip install transformers accelerate timm bitsandbytes pytesseract --quiet
    #!apt-get install tesseract-ocr -y > /dev/null


    # STEP 2: Import libraries
    #from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    from transformers import BlipProcessor, BlipForConditionalGeneration

    from PIL import Image
    import pytesseract
    import torch
    import os
    import json
    from dotenv import load_dotenv
    #from google.colab import drive

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    load_dotenv()

    # STEP 3: Mount Google Drive
    #drive.mount('/content/drive')

    # STEP 4: Set image folder path (CHANGE THIS!)
    #image_folder = "/content/drive/MyDrive/nlp/project/datasets/" + assetID
    image_folder = "./datasets/" + assetID

    # STEP 5: Load model

    #InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", cache_dir="./.hf_cache")
    #model_name = "./.hf_cache"

    #model_name = "Salesforce/instructblip-flan-t5-xl"
    model_name = "Salesforce/blip-image-captioning-base"


    #processor = InstructBlipProcessor.from_pretrained(model_name)
    processor = BlipProcessor.from_pretrained(model_name)

    print("Loading model...")
    #model = InstructBlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    #model = InstructBlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32, device_map="auto")
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    print("Model loaded!")
    model.to("cuda" if torch.cuda.is_available() else "cpu")


    # STEP 6: Set domain-specific prompt
    prompt = (
        "You're a collateral appraisal assistant.\n"
        "Describe what is visible in the image that would help estimate the value of a vehicle or machine.\n"
        "Mention visible make, model, year, license plate, condition, documents, or dashboard info."
    )


    out_path = os.path.join(image_folder, "instructblip_ocr_results.json")
    if os.path.exists(out_path):
        print(f"✅ Skipping captioning+OCR – file already exists: {out_path}")
        with open(out_path, "r") as f:
            results = json.load(f)
    else:
    # STEP 7: Run captioning + OCR
        results = []
        image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])#[:10]  # Limit to 10 for testing

        for fname in image_files:
            fpath = os.path.join(image_folder, fname)
            try:
                image = Image.open(fpath).convert("RGB")

                # InstructBLIP caption
                #inputs = processor(image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
                inputs = processor(image, return_tensors="pt").to(model.device)


                output = model.generate(**inputs, max_new_tokens=100)
                caption = processor.decode(output[0], skip_special_tokens=True)

                # OCR text
                ocr_text = pytesseract.image_to_string(image)

                results.append({
                    "image": fname,
                    "caption": caption,
                    "ocr": ocr_text.strip()
                })

                print(f"✓ {fname}\n  Caption: {caption}\n  OCR: {ocr_text.strip()[:100]}...\n")

            except Exception as e:
                print(f"⚠️ Failed to process {fname}: {e}")

    # STEP 8: Save results
    out_path = os.path.join(image_folder, "instructblip_ocr_results.json")
    if os.path.exists(out_path):
        pass
    else:
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\n✅ Combined results saved to: {out_path}")

    from typing import List

    # Re-defining classify_domain with List now imported
    def classify_domain(captions: List[str], ocr_lines: List[str]) -> str:
        text = " ".join(captions + ocr_lines).lower()

        VEHICLE_KEYWORDS = [
            "car", "truck", "vehicle", "license plate", "dashboard", "bmw", "chevrolet", "mileage", "odometer",
            "steering wheel", "interior", "gear", "sedan", "hatchback", "station wagon", "diesel", "petrol",
            "speedometer", "tachometer", "airbag", "engine"
        ]

        BUILDING_KEYWORDS = [
            "building", "apartment", "facade", "roof", "floor plan", "real estate", "entrance",
            "kitchen", "bathroom", "room", "window", "door", "garage", "balcony", "house", "villa",
            "elevator", "corridor", "stairs", "property", "unit"
        ]

        MACHINE_KEYWORDS = [
            "machine", "motor", "gear", "compressor", "engine", "tractor", "pump", "hydraulic",
            "industrial", "valve", "cylinder", "bearing", "drive", "mechanism", "machinery",
            "excavator", "bulldozer", "press", "robot arm", "generator", "lathe"
        ]

        scores = {
            "Vehicle": sum(kw in text for kw in VEHICLE_KEYWORDS),
            "Building": sum(kw in text for kw in BUILDING_KEYWORDS),
            "Machine": sum(kw in text for kw in MACHINE_KEYWORDS),
        }
        return max(scores, key=scores.get)

    # Adding the vehicle subtype classifier
    def classify_vehicle_subtype(text: str) -> str:
        text = text.lower()

        if any(kw in text for kw in [
            "tractor unit", "trailer", "semi-trailer", "actros", "daf", "scania", "man", "volvo fh",
            "iveco", "renault truck", "tgs", "tgx", "fh", "xf", "cargo"
        ]):
            return "Truck"

        elif any(kw in text for kw in [
            "motorbike", "motorcycle", "bike", "2 wheels", "harley", "yamaha", "suzuki", "kawasaki", "ktm"
        ]):
            return "Motorcycle"

        elif any(kw in text for kw in [
            "bus", "coach", "passenger capacity", "school bus", "minibus"
        ]):
            return "Bus"

        elif any(kw in text for kw in [
            "suv", "crossover", "4x4", "all-terrain"
        ]):
            return "SUV"

        elif any(kw in text for kw in [
            "van", "transporter", "panel van", "cargo van", "vw t5", "sprinter", "transit"
        ]):
            return "Van"

        elif any(kw in text for kw in [
            "locomotive", "train", "rail", "cz loko", "freight rail", "diesel-electric", "shunting", "engineer cabin"
        ]):
            return "Locomotive"

        elif any(kw in text for kw in [
            "bmw", "audi", "mercedes", "volkswagen", "ford", "sedan", "hatchback", "wagon",
            "estate", "coupe", "compact", "passenger car", "private car", "peugeot", "citroën", "renault"
        ]):
            return "Car"

        else:
            return "Vehicle"


    # STEP 8: Extract caption and OCR lists for classification
    captions = [r["caption"] for r in results]
    ocr_lines = [line for r in results for line in r["ocr"].splitlines() if line.strip()]


    domain = classify_domain(captions, ocr_lines)
    text = " ".join(captions + ocr_lines)
    subtype = classify_vehicle_subtype(text) if domain == "Vehicle" else domain


    domain_templates = {
        "Vehicle": """### Human:

    Here is an example appraisal report for a vehicle asset:

    ## [insert assetID here]
    ### Identification & General Data
    - Asset Type: Vehicle
    - Manufacturer: Unknown
    - Model: Unknown
    - Year of Manufacture: Unknown
    - Odometer Reading: Unknown

    ### Inspection Methods
    - Visual exterior inspection and basic document verification

    ### Condition Assessment
    - Basic inspection; no visible damage
    - Limited data available

    ### Valuation Principles
    - Estimate based on general vehicle market for unspecified type

    ### Determined Values
    - Market Sales Value: 3,000,000 HUF
    - Liquidation Value: 2,400,000 HUF

    ### Documentation & Accessories
    - Few documents available; limited supporting materials

    Now please generate a similar report based on the following extracted information:
    The asset id is: """,

        "Truck": """### Human:

    Here is an example appraisal report for a vehicle asset:

    ## [insert assetID here]
    ### Identification & General Data
    - Asset Type: Vehicle (Tractor unit)
    - Manufacturer: DAF
    - Model: XF 480FT
    - Year of Manufacture: 2018
    - Odometer Reading: 439,014 km

    ### Inspection Methods
    - Visual inspection, test drive, digital photos
    - Basic paint thickness and structural testing

    ### Condition Assessment
    - Age-appropriate wear
    - No major damage; engine and systems in working condition

    ### Valuation Principles
    - Based on market price, age, mileage
    - Adjusted for average condition

    ### Determined Values
    - Market Sales Value: 10,500,000 HUF
    - Liquidation Value: 9,000,000 HUF

    ### Documentation & Accessories
    - 2 keys, service book, digital photos, valuation software used

    Now please generate a similar report based on the following extracted information:
    The asset id is: """,

        "Motorcycle": """### Human:

    Here is an example appraisal report for a vehicle asset:

    ## [insert assetID here]
    ### Identification & General Data
    - Asset Type: Vehicle (Motorcycle)
    - Manufacturer: Yamaha
    - Model: MT-07
    - Year of Manufacture: 2020
    - Odometer Reading: 12,500 km

    ### Inspection Methods
    - Visual inspection, engine sound check, brake response
    - Review of maintenance records

    ### Condition Assessment
    - Minimal wear, no accident history
    - Tires and chain in good condition

    ### Valuation Principles
    - Based on brand, mileage, and market comparables

    ### Determined Values
    - Market Sales Value: 2,100,000 HUF
    - Liquidation Value: 1,700,000 HUF

    ### Documentation & Accessories
    - Service book, keys, maintenance log

    Now please generate a similar report based on the following extracted information:
    The asset id is: """,

        "Bus": """### Human:

    Here is an example appraisal report for a vehicle asset:

    ## [insert assetID here]
    ### Identification & General Data
    - Asset Type: Vehicle (Bus)
    - Manufacturer: Mercedes-Benz
    - Model: Citaro
    - Year of Manufacture: 2015
    - Odometer Reading: 620,000 km

    ### Inspection Methods
    - Exterior/interior visual check, test drive
    - Seat count and accessibility audit

    ### Condition Assessment
    - High mileage, minor wear on seating
    - Operational, recently serviced

    ### Valuation Principles
    - Market comps for city buses, adjusted for age and mileage

    ### Determined Values
    - Market Sales Value: 9,000,000 HUF
    - Liquidation Value: 7,000,000 HUF

    ### Documentation & Accessories
    - Registration, maintenance records, seating plan

    Now please generate a similar report based on the following extracted information:
    The asset id is: """,

        "SUV": """### Human:

    Here is an example appraisal report for a vehicle asset:

    ## [insert assetID here]
    ### Identification & General Data
    - Asset Type: Vehicle (SUV)
    - Manufacturer: Toyota
    - Model: RAV4 Hybrid
    - Year of Manufacture: 2021
    - Odometer Reading: 35,000 km

    ### Inspection Methods
    - Visual inspection, documentation review, dashboard photos

    ### Condition Assessment
    - Excellent condition, low wear
    - Full hybrid system functional

    ### Valuation Principles
    - High residual value due to fuel efficiency and demand

    ### Determined Values
    - Market Sales Value: 11,000,000 HUF
    - Liquidation Value: 9,500,000 HUF

    ### Documentation & Accessories
    - EU registration, full maintenance log, dashboard image

    Now please generate a similar report based on the following extracted information:
    The asset id is: """,

        "Van": """### Human:

    Here is an example appraisal report for a vehicle asset:

    ## [insert assetID here]
    ### Identification & General Data
    - Asset Type: Vehicle (Van)
    - Manufacturer: Ford
    - Model: Transit
    - Year of Manufacture: 2019
    - Odometer Reading: 210,000 km

    ### Inspection Methods
    - Visual inspection, cargo space review
    - Documentation and accessory check

    ### Condition Assessment
    - Heavy wear on cargo area
    - Engine recently serviced, tires good

    ### Valuation Principles
    - Adjusted for mileage and commercial use

    ### Determined Values
    - Market Sales Value: 6,000,000 HUF
    - Liquidation Value: 4,800,000 HUF

    ### Documentation & Accessories
    - Cargo divider, service book, insurance card

    Now please generate a similar report based on the following extracted information:
    The asset id is: """,

        "car": """### Human:

    Here is an example appraisal report for a vehicle asset:

    ## [insert assetID here]
    ### Identification & General Data
    - Asset Type: Vehicle (Passenger car)
    - Manufacturer: DAF
    - Model: XF 480FT
    - Year of Manufacture: 2018
    - Odometer Reading: 439,014 km

    ### Inspection Methods
    - Visual inspection, digital photos, interior dashboard capture
    - Documentation review (license plate, inspection logs)

    ### Condition Assessment
    - Age-appropriate wear and tear
    - Interior intact, electronics functional
    - No major external damage

    ### Valuation Principles
    - Market value estimation based on year, mileage, and condition

    ### Determined Values
    - Market Sales Value: 4,500,000 HUF
    - Liquidation Value: 3,800,000 HUF

    ### Documentation & Accessories
    - Photos of license plate, dashboard, screens
    - Screenshots showing system info and inspection validity

    Now please generate a similar report based on the following extracted information:
    The asset id is: """,

        "Building": """### Human:

    Here is an example appraisal report for a real estate asset:

    ## [insert assetID here]
    ### Identification & General Data
    - Asset Type: Building (Residential)
    - Location: Pest County, Hungary
    - Built Year: 2006
    - Floor Area: 120 m²

    ### Inspection Methods
    - On-site inspection
    - Photographic documentation
    - Utility and ownership document verification

    ### Condition Assessment
    - Good structural integrity
    - Minor cracks on exterior wall

    ### Valuation Principles
    - Market analysis of residential buildings in the region

    ### Determined Values
    - Market Sales Value: 38,000,000 HUF
    - Liquidation Value: 30,000,000 HUF

    ### Documentation & Accessories
    - Utility bills, property deed, floor plans

    Now please generate a similar report based on the following extracted information:
    The asset id is: """,

        "Machine": """### Human:

    Here is an example appraisal report for a machine asset:

    ## [insert assetID here]
    ### Identification & General Data
    - Asset Type: Machine
    - Type: Industrial compressor
    - Manufacturer: Atlas Copco
    - Model: GA90VSD
    - Year of Manufacture: 2016
    - Hours Operated: 8,300 hrs

    ### Inspection Methods
    - Visual inspection, control panel photo, service history review

    ### Condition Assessment
    - Minor oil residue; normal usage wear
    - Operates within specification

    ### Valuation Principles
    - Based on industrial equipment sales benchmarks

    ### Determined Values
    - Market Sales Value: 5,200,000 HUF
    - Liquidation Value: 4,000,000 HUF

    ### Documentation & Accessories
    - Operator manual, service record, certification sticker

    Now please generate a similar report based on the following extracted information:
    The asset id is: """
    }

    # Add the new "Locomotive" subtype template
    domain_templates["Locomotive"] = """### Human:

    Here is an example appraisal report for a vehicle asset:

    ## [insert assetID here]
    ### Identification & General Data
    - Asset Type: Vehicle (Locomotive)
    - Manufacturer: CZ LOKO
    - Model: Unknown
    - Year of Manufacture: 2018
    - Engine Type: Diesel-electric

    ### Inspection Methods
    - Visual inspection of the exterior and control cabin
    - Undercarriage and suspension system check

    ### Condition Assessment
    - Appears structurally sound
    - Cabin controls intact, exterior clean, suspension robust

    ### Valuation Principles
    - Based on model year, manufacturer, and operational features

    ### Determined Values
    - Market Sales Value: 70,000,000 HUF
    - Liquidation Value: 55,000,000 HUF

    ### Documentation & Accessories
    - Inspection photos of cabin, serial plates, undercarriage

    Now please generate a similar report based on the following extracted information:
    The asset id is: """


    def build_qwen_prompt(
        domain_prompt: str,
        object_description: str,
        inspection_notes: str,
        captions: List[str],
        ocr_header: List[str],
        ocr_lines: List[str]
    ) -> str:
        prompt = domain_prompt.strip() + "\n\n"

        # Inject object-specific observations
        if object_description:
            prompt += object_description.strip() + "\n"
        if inspection_notes:
            prompt += inspection_notes.strip() + "\n"

        # Captions
        prompt += "\n**Captions:**\n"
        prompt += ''.join(f"- {cap.strip()}\n" for cap in captions)

        # OCR
        prompt += "\n**OCR:**\n"
        if ocr_header:
            prompt += ''.join(f"- {line.strip()}\n" for line in ocr_header)
        if ocr_lines:
            prompt += ''.join(f"- {line.strip()}\n" for line in ocr_lines)

        # Close prompt
        prompt += "\n\n### Assistant:\n"
        return prompt


    # --- Generate Qwen Prompt from InstructBLIP + OCR Results ---
    # Loads instructblip_ocr_results.json, selects top captions + OCR lines, and generates a markdown-ready prompt

    import json
    import os
    import re



    # STEP 1: Load results (CHANGE path as needed)
    folder_path = "./datasets/" + assetID
    json_path = os.path.join(folder_path, "instructblip_ocr_results.json")

    with open(json_path, 'r') as f:
        results = json.load(f)


    # STEP 2: Score and filter captions
    used = set()
    def score_caption(text):
        score = 0
        for keyword in ["dashboard", "document", "mileage", "year", "model", "bill", "registration"]:
            if keyword in text.lower():
                score += 1
        return score

    ranked = sorted(results, key=lambda x: score_caption(x["caption"]), reverse=True)
    top_captions = []
    for item in ranked:
        if item['caption'] not in used and len(top_captions) < 5:
            top_captions.append(item['caption'].strip())
            used.add(item['caption'])

    # STEP 3: Extract useful OCR lines
    ocr_lines = []
    for item in results:
        for line in item["ocr"].splitlines():
            if re.search(r"(\d{4}|\dkm|[A-Z]{2,}\d{2,}|KG|Actros|Mercedes|VIN|km)", line, re.IGNORECASE):
                clean = line.strip()
                if clean and clean not in ocr_lines:
                    ocr_lines.append(clean)
            if len(ocr_lines) >= 5:
                break

    #object_description = "This asset is a Mercedes-Benz Actros 1851 tractor unit, manufactured in 2022 with approx. 233,000 km mileage."
    manual_descriptions = {
    #    "157515_v2": "",
        "158220": "This asset is a Mercedes-Benz Actros 1851 tractor unit, manufactured in 2022 with approx. 233,000 km mileage.",
    #    "159347": "",
    #    "161552_v2": "",

        "162390": "This asset is a BMW passenger vehicle, possibly manufactured around 2022–2023, with approx. 366,000 km mileage.",
        "162749": "This asset is a BMW passenger vehicle, estimated to be manufactured around 2022–2023, with documented mileage near 60,000 km.",
        "165484": "This asset is a Ford Kuga passenger vehicle, plug-in hybrid variant, manufactured around 2021–2022. It features a red exterior finish, modern dashboard with digital instrumentation, and dual-zone climate control. The vehicle is equipped with two key fobs, navigation system, and appears to be in well-maintained condition. Photos also confirm charging accessories and intact interior features. The odometer shows approximately 56,000 km.",

        "165968": "This asset is a BMW X5 SUV, diesel-powered, manufactured around 2021–2022, with approx. 63,833 km mileage.",
        "169883": "This asset is a Krone refrigerated semi-trailer (Typ SD), manufactured in Germany, with a cargo length of 14,040 mm and width of 2,600 mm. It is equipped with Thermo King SLXi Spectrum cooling system.",
        "CZ LOKO": "This asset is a CZ LOKO diesel-electric locomotive manufactured in 2018, designed for heavy freight rail operations. It features a modern control cabin and robust suspension system.",

        # Add more entries here as needed
    }

    manual_descriptions["161552_v2"] = "This asset is a Mercedes-Benz V-Class passenger van, equipped with a premium interior layout including captain’s seats and navigation system. It was manufactured in 2019 and currently has approximately 68,000 km mileage. The vehicle has an automatic transmission and a diesel engine. It shows signs of careful maintenance and is in excellent cosmetic and technical condition, with high-end infotainment and safety features visible from the dashboard and central console. It is likely a V 250 d or similar variant, used for executive transport or private luxury travel."

    manual_descriptions["159347"] = "This asset is an industrial metal processing and separation line, manufactured in 2024 by Henan Recycling Technology Co., Ltd. It includes a vibration feeder (Model FYZW-960) and a pulse bag dust collector (Model DMC-72), both powered by 380V/50Hz electric systems. The machinery is designed for the separation and collection of processed scrap materials, possibly involving shredding and vibration-based separation stages. The system is installed indoors and connected to conveyor belts and ventilation systems."

    manual_descriptions["157515_v2"] = "This asset is a DAF XF 480 FT tractor unit, manufactured in 2018 with approximately 439,000 km mileage. It is designed for long-haul freight transport and is equipped with advanced dashboard instrumentation, aerodynamic cabin, and robust chassis. The vehicle shows age-appropriate wear and was last technically inspected in 2023."

    manual_descriptions["165671"] = (
        "This asset is a black KIA Cee'd SW 1.6 GDI EX Prémium station wagon passenger car, "
        "manufactured in early 2018, with approximately 118,500 km mileage. "
        "It features a gasoline engine, automatic transmission, and multiple comfort features "
        "including navigation system, dual-zone climate control, parking sensors, and camera support. "
        "The vehicle's condition appears consistent with age and usage based on the dashboard reading, "
        "engine bay cleanliness, trunk layout, and exterior images. Documentation includes the original "
        "vehicle data sheet, service booklet, and registration card."
    )


    manual_descriptions["165720"] = (
        "This asset is an Audi Q3 40 TFSI S line quattro passenger vehicle with S-tronic transmission, "
        "first registered in 2023 and showing approximately 25,600 km on the odometer. "
        "It features a dark grey exterior in excellent condition and a modern, premium interior equipped with a digital dashboard, "
        "central infotainment screen, and high-end driving assistance features. "
        "The vehicle is all-wheel drive (quattro), fitted with sport alloy wheels and rear parking camera. "
        "The inspection included dashboard diagnostics, manufacturer documentation (Audi AG type plate), registration certificate, "
        "and accessory review (2 keys present)."
    )


    object_description = manual_descriptions[assetID]
    if manualDescription:
        object_description = manualDescription
    inspection_notes = "The inspection included visual checks and a review of the attached documents."
    domain = domain
    #example_prompt = domain_templates[domain] + assetID
    example_prompt = domain_templates[subtype] + assetID

    final_prompt = build_qwen_prompt(
        domain_prompt=example_prompt,
        object_description=object_description,
        inspection_notes=inspection_notes,
        captions=top_captions,
        ocr_header=["Mercedes-Benz Actros 1851", "2022", "233014 km"],
        ocr_lines=ocr_lines
    )


    # STEP 5: Print + save prompt
    print("\n✅ Qwen prompt:\n")
    print(final_prompt)

    with open(os.path.join(folder_path, "qwen_prompt_" + assetID + ".md"), "w") as f:
        f.write(final_prompt)
    print("\n📄 Prompt saved to: qwen_prompt_" + assetID + ".md")















    import requests

    # STEP 1: Fill in your endpoint and key
    API_URL = os.getenv("QWEN_API_URL")
    API_KEY = os.getenv("QWEN_API_KEY")


    # STEP 2: Load your prompt text
    with open("./datasets/" + assetID + "/qwen_prompt_" + assetID + ".md", "r") as f:
        prompt_text = f.read()

    # STEP 3: Create messages list
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates markdown-style appraisal reports."},
        {"role": "user", "content": prompt_text}
    ]


    # STEP 4: Compose payload for OpenAI-style chat model
    payload = {
        #"model": "qwen-chat",  # or whatever model name you're using
        "messages": messages,
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 4096
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # STEP 5: Make the request
    response = requests.post(API_URL, headers=headers, json=payload)

    # STEP 6: Handle response
    if response.status_code == 200:
        result = response.json()
        output = result["choices"][0]["message"]["content"]
        print("✅ Qwen Response:\n")
        print(output)
        out_path = os.path.join(image_folder, "qwen_output_" + assetID + ".md")
        with open(out_path, "w") as out:
            out.write(output)
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

main()