import base64
import json
import os
import shutil
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
 
# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
 
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")
 
def convert_pdf_to_image(pdf_path):
    try:
        from pdf2image import convert_from_path
        print("Converting PDF to images...")
        images = convert_from_path(pdf_path, dpi=300)
 
        pdf_dir = os.path.dirname(pdf_path) or "."
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        image_paths = []
        for i, img in enumerate(images, 1):
            image_path = os.path.join(pdf_dir, f"{pdf_name}_page{i}.png")
            img.save(image_path, 'PNG')
            image_paths.append(image_path)
        
        print(f"Saved {len(image_paths)} page(s)")
        return image_paths
 
    except ImportError:
        print("pdf2image not installed. Run: pip install pdf2image")
        return None
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return None


def extract_restaurant_info(file_path, groq_api_key):
    """Extract restaurant name and phone from first page only"""
    client = Groq(api_key=groq_api_key)
    
    with open(file_path, "rb") as file:
        file_data = base64.b64encode(file.read()).decode("utf-8")
    
    prompt = """Extract restaurant name and phone number. Return ONLY JSON:

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
                            "image_url": {"url": f"data:image/png;base64,{file_data}"}
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
 
 
def extract_menu_to_json(file_path, groq_api_key, retry_with_shorter_prompt=False):
    """Extract menu items from a single page with retry logic"""
    client = Groq(api_key=groq_api_key)
 
    with open(file_path, "rb") as file:
        file_data = base64.b64encode(file.read()).decode("utf-8")

    if retry_with_shorter_prompt:
        prompt = """Extract menu items as JSON:
{"categories": [{"category": "string", "items": [{"name": "string", "price": number}]}]}

Rules:
- Split items with multiple prices into separate entries
- Add variant/size to item name
- Price = number only
- Output ONLY JSON
"""
    else:
        prompt = """Extract all menu items from this restaurant menu image and return ONLY a valid JSON object.
 
Structure the JSON like this:
{
  "categories": [
    {
      "category": "string",
      "items": [
        {"name": "string", "price": number}
      ]
    }
  ]
}
 
Rules:
1. Extract all items with exact names and prices.
2. Group items by category headers.
3. If an item has multiple prices (different sizes, variants, or options), create separate entries for each.
4. Include size/variant information in the item name to distinguish them.
5. Prices must be numbers only (no strings like "180/190").
6. No explanations. Only JSON.
"""
 
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{file_data}"}
                        }
                    ]
                }
            ],
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.1,
            max_tokens=8192
        )
 
        response_text = chat_completion.choices[0].message.content.strip()
 
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
 
        menu_data = json.loads(response_text)
        return menu_data
 
    except json.JSONDecodeError as e:
        print(f"⚠ JSON Parse Error: {e}")
        print(f"⚠ Attempting to fix malformed JSON...")
        
        # Try to salvage partial JSON
        try:
            last_brace = response_text.rfind('}')
            if last_brace > 0:
                test_json = response_text[:last_brace+1]
                open_braces = test_json.count('{')
                close_braces = test_json.count('}')
                test_json += '}' * (open_braces - close_braces)
                
                menu_data = json.loads(test_json)
                print("✓ Successfully recovered partial data")
                return menu_data
        except:
            pass
        
        print("✗ Could not recover data from this page")
        
        # Retry with shorter prompt if first attempt
        if not retry_with_shorter_prompt:
            print("⚠ Retrying with minimal prompt...")
            return extract_menu_to_json(file_path, groq_api_key, retry_with_shorter_prompt=True)
        
        return None
        
    except Exception as e:
        print(f"API Error: {e}")
        return None


def save_page_json(page_data, output_dir, page_num):
    """Save individual page JSON to temp directory"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"page_{page_num}.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(page_data, f, indent=2, ensure_ascii=False)
    
    return output_path


def merge_page_jsons(output_dir, restaurant_info, final_output_path):
    """Merge all page JSONs into final output"""
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
 

def cleanup_images(image_paths):
    """Delete temporary image files created from PDF conversion"""
    for path in image_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Warning: Could not delete {path}: {e}")
 
 
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 app.py <menu_file>")
        print("Supported formats: PDF, JPG, JPEG, PNG")
        exit(1)
    
    menu_file = sys.argv[1]
 
    if not os.path.exists(menu_file):
        print(f"File not found: {menu_file}")
        exit(1)
 
    print(f"Found: {menu_file}")
 
    is_pdf = menu_file.lower().endswith(".pdf")
    
    if is_pdf:
        FILE_PATHS = convert_pdf_to_image(menu_file)
        if not FILE_PATHS:
            exit(1)
    else:
        FILE_PATHS = [menu_file]
 
    # Create temp directory for page JSONs
    menu_dir = os.path.dirname(menu_file) or "."
    menu_name = os.path.splitext(os.path.basename(menu_file))[0]
    temp_dir = os.path.join(menu_dir, f".{menu_name}_pages")
    
    print("Extracting menu data...")
    
    # Extract restaurant info from first page only
    print("\nExtracting restaurant info from first page...")
    restaurant_info = extract_restaurant_info(FILE_PATHS[0], GROQ_API_KEY)
    print(f"Restaurant: {restaurant_info.get('restaurant_name')}")
    print(f"Phone: {restaurant_info.get('phone')}")
 
    # Process each page and save individually
    successful_pages = 0
    for i, path in enumerate(FILE_PATHS, 1):
        print(f"\n{'='*60}")
        print(f"Processing page {i}/{len(FILE_PATHS)}...")
        print(f"{'='*60}")
        
        menu_data = extract_menu_to_json(path, GROQ_API_KEY)
        
        if menu_data:
            save_page_json(menu_data, temp_dir, i)
            print(f"✓ Successfully extracted {len(menu_data.get('categories', []))} categories from page {i}")
            print(f"✓ Saved page {i} JSON")
            successful_pages += 1
        else:
            print(f"✗ Failed to extract data from page {i}")
    
    # Clean up temporary PDF page images
    if is_pdf:
        print("\nCleaning up temporary image files...")
        cleanup_images(FILE_PATHS)
        print("✓ Temporary image files removed")
    
    # Merge all page JSONs into final output
    if successful_pages > 0:
        print(f"\n{'='*60}")
        print("MERGING ALL PAGES")
        print(f"{'='*60}")
        
        final_output = os.path.join(menu_dir, f"{menu_name}_extracted.json")
        combined = merge_page_jsons(temp_dir, restaurant_info, final_output)
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        print(f"✓ Cleaned up temporary page JSONs")
        
        total_items = sum(len(cat["items"]) for cat in combined.get("categories", []))
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Pages Processed: {successful_pages}/{len(FILE_PATHS)}")
        print(f"Categories Extracted: {len(combined['categories'])}")
        print(f"Total Items: {total_items}")
    else:
        print("\n✗ No data extracted from any page!")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)