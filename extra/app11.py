import base64
import json
import os
from io import BytesIO
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")


# -------------------------------------------------------------------
# Convert PDF → images in memory (NO TEMP FILES)
# -------------------------------------------------------------------
def convert_pdf_to_images_in_memory(pdf_path):
    try:
        from pdf2image import convert_from_path
        print("Converting PDF to images in memory...")

        pil_images = convert_from_path(pdf_path, dpi=300)
        image_bytes_list = []

        for img in pil_images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            image_bytes_list.append(buf.read())

        print(f"Loaded {len(image_bytes_list)} page(s) into memory")
        return image_bytes_list

    except Exception as e:
        print(f"Error converting PDF: {e}")
        return None


# -------------------------------------------------------------------
# Extract restaurant info from first page only
# -------------------------------------------------------------------
def extract_restaurant_info(image_bytes, groq_api_key):
    client = Groq(api_key=groq_api_key)
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    prompt = """
Extract restaurant name and phone number. Return ONLY JSON:

{
  "restaurant_name": "string or null",
  "phone": "string or null"
}
"""

    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.1,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                        }
                    ]
                }
            ]
        )

        response = completion.choices[0].message.content.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        return json.loads(response.strip())
    except:
        return {"restaurant_name": None, "phone": None}


# -------------------------------------------------------------------
# Extract menu items from a single page
# -------------------------------------------------------------------
def extract_menu_to_json(image_bytes, groq_api_key, retry_with_shorter_prompt=False):
    client = Groq(api_key=groq_api_key)
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    if retry_with_shorter_prompt:
        prompt = """
Extract menu items as JSON:
{"categories": [{"category": "string", "items": [{"name": "string", "price": number}]}]}

Rules:
- Split items with multiple prices into separate entries
- Add variant/size to item name
- Price = number only
- Output ONLY JSON
"""
    else:
        prompt = """
Extract menu items. Return ONLY JSON:

{
  "categories": [
    {
      "category": "string",
      "items": [{"name": "string", "price": number}]
    }
  ]
}

CRITICAL RULES:
1. Extract ALL text exactly as shown - full item descriptions, ingredients, serving details
2. Create NEW category when you see a category header (usually bold/larger text)
3. If one item name has multiple price options (like "180/190" or "Veg/Paneer 180/190"):
   - Split into separate items
   - Add the variant/option to each item name
4. Price must be a single number only
5. ONLY output JSON, no explanations
"""

    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.1,
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                        }
                    ]
                }
            ]
        )

        response = completion.choices[0].message.content.strip()

        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        response = response.strip()
        return json.loads(response)

    except json.JSONDecodeError as e:
        print(f"⚠ JSON Decode Error: {e}")
        print(f"⚠ Attempting to recover partial data...")
        
        try:
            last_brace = response.rfind('}')
            if last_brace > 0:
                test_json = response[:last_brace+1]
                open_braces = test_json.count('{')
                close_braces = test_json.count('}')
                test_json += '}' * (open_braces - close_braces)
                
                menu_data = json.loads(test_json)
                print("✓ Recovered partial data")
                return menu_data
        except:
            pass
        
        print("✗ Could not recover data")
        
        if not retry_with_shorter_prompt:
            print("⚠ Retrying with minimal prompt...")
            return extract_menu_to_json(image_bytes, groq_api_key, retry_with_shorter_prompt=True)
        
        return None

    except Exception as e:
        print(f"API Error: {e}")
        return None


# -------------------------------------------------------------------
# Save individual page JSON
# -------------------------------------------------------------------
def save_page_json(page_data, output_dir, page_num):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"page_{page_num}.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(page_data, f, indent=2, ensure_ascii=False)
    
    return output_path


# -------------------------------------------------------------------
# Merge all page JSONs into final output
# -------------------------------------------------------------------
def merge_page_jsons(output_dir, restaurant_info, final_output_path):
    all_categories = []
    
    page_files = sorted([f for f in os.listdir(output_dir) if f.startswith("page_") and f.endswith(".json")])
    
    for page_file in page_files:
        page_path = os.path.join(output_dir, page_file)
        with open(page_path, "r", encoding="utf-8") as f:
            page_data = json.load(f)
            all_categories.extend(page_data.get("categories", []))
    
    combined = {
        "restaurant_name": restaurant_info.get("restaurant_name"),
        "phone": restaurant_info.get("phone"),
        "categories": all_categories
    }
    
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    print(f"Saved final JSON to {final_output_path}")
    return combined


# -------------------------------------------------------------------
# MAIN SCRIPT
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import shutil

    if len(sys.argv) < 2:
        print("Usage: python3 script.py <menu_file>")
        print("Supported: PDF, JPG, JPEG, PNG")
        exit(1)

    menu_file = sys.argv[1]

    if not os.path.exists(menu_file):
        print(f"File not found: {menu_file}")
        exit(1)

    print(f"Found: {menu_file}")

    # Create temp directory for page JSONs
    menu_dir = os.path.dirname(menu_file) or "."
    menu_name = os.path.splitext(os.path.basename(menu_file))[0]
    temp_dir = os.path.join(menu_dir, f".{menu_name}_pages")
    
    # Load images into memory
    if menu_file.lower().endswith(".pdf"):
        IMAGE_BYTES_LIST = convert_pdf_to_images_in_memory(menu_file)
        if not IMAGE_BYTES_LIST:
            exit(1)
    else:
        with open(menu_file, "rb") as f:
            IMAGE_BYTES_LIST = [f.read()]

    print("Extracting menu data...")

    # Extract restaurant info from first page
    print("\nExtracting restaurant info from first page...")
    restaurant_info = extract_restaurant_info(IMAGE_BYTES_LIST[0], GROQ_API_KEY)
    print(f"Restaurant: {restaurant_info.get('restaurant_name')}")
    print(f"Phone: {restaurant_info.get('phone')}")

    # Process each page and save individually
    successful_pages = 0
    for i, img_bytes in enumerate(IMAGE_BYTES_LIST, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing page {i}/{len(IMAGE_BYTES_LIST)}...")
        print(f"{'=' * 60}")

        page_data = extract_menu_to_json(img_bytes, GROQ_API_KEY)

        if page_data:
            save_page_json(page_data, temp_dir, i)
            print(f"✓ Extracted {len(page_data.get('categories', []))} categories")
            print(f"✓ Saved page {i} JSON")
            successful_pages += 1
        else:
            print(f"✗ Failed to extract page {i}")

    # Merge all page JSONs into final output
    if successful_pages > 0:
        print(f"\n{'=' * 60}")
        print("MERGING ALL PAGES")
        print(f"{'=' * 60}")
        
        final_output = os.path.join(menu_dir, f"{menu_name}_extracted.json")
        combined = merge_page_jsons(temp_dir, restaurant_info, final_output)
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        print(f"✓ Cleaned up temporary files")
        
        total = sum(len(c["items"]) for c in combined.get("categories", []))
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"Pages Processed: {successful_pages}/{len(IMAGE_BYTES_LIST)}")
        print(f"Categories Extracted: {len(combined['categories'])}")
        print(f"Total Items: {total}")
    else:
        print("\n✗ No data extracted from any page!")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)