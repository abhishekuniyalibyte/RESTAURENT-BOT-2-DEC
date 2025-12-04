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
 
 
def extract_menu_to_json(file_path, groq_api_key, retry_with_shorter_prompt=False):
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
- Use only plain ASCII characters in item names (replace special characters with regular letters)
- Output ONLY valid JSON, no extra text
"""
    else:
        prompt = """Extract all menu items from this restaurant menu image and return ONLY a valid JSON object.
 
Structure the JSON like this:
{
  "restaurant_name": "string or null",
  "phone": "string or null",
  "categories": [
    {
      "category": "string",
      "items": [
        {"name": "string", "price": number}
      ]
    }
  ]
}
 
CRITICAL RULES:
1. Extract ALL items with exact names and prices.
2. IMPORTANT: Identify category headers by these characteristics:
   - Text that is visually distinct (bold, larger font, different style, or underlined)
   - Text that appears alone on a line or section without a price immediately after it
   - Text that logically groups the items below it
3. Create a NEW category for each distinct section heading you identify.
4. If you see text without a price that appears to introduce a new section of items, treat it as a category header.
5. If an item has multiple prices (different sizes/variants), create separate entries for each.
6. Include size/variant information in the item name to distinguish them.
7. Prices must be numbers only (no strings like "140/150"). If no price is visible, use null.
8. IMPORTANT: Ensure all text uses standard characters. Replace special characters (é, â, etc.) with regular letters.
9. No explanations. Only valid JSON.
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
        
        # Try multiple recovery strategies
        recovery_attempts = [
            # Strategy 1: Remove control characters
            lambda txt: txt.encode('utf-8', 'ignore').decode('utf-8', 'ignore'),
            # Strategy 2: Replace problematic characters
            lambda txt: txt.replace('\n', ' ').replace('\r', ' ').replace('\t', ' '),
            # Strategy 3: Fix truncated JSON
            lambda txt: txt[:txt.rfind('}')+1] if txt.rfind('}') > 0 else txt,
        ]
        
        for idx, strategy in enumerate(recovery_attempts, 1):
            try:
                cleaned_text = strategy(response_text)
                
                # Balance braces if needed
                open_braces = cleaned_text.count('{')
                close_braces = cleaned_text.count('}')
                if open_braces > close_braces:
                    cleaned_text += '}' * (open_braces - close_braces)
                
                menu_data = json.loads(cleaned_text)
                print(f"✓ Successfully recovered data using strategy {idx}")
                return menu_data
            except:
                continue
        
        print("✗ Could not recover data from this page")
        
        # Retry with shorter prompt if first attempt
        if not retry_with_shorter_prompt:
            print("⚠ Retrying with minimal prompt...")
            return extract_menu_to_json(file_path, groq_api_key, retry_with_shorter_prompt=True)
        
        return None
        
    except Exception as e:
        print(f"API Error: {e}")
        return None
 
 
def save_menu_json(menu_data, input_path, output_filename=None):
    input_dir = os.path.dirname(input_path) or "."
 
    if output_filename is None:
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{input_name}_extracted.json"
 
    output_path = os.path.join(input_dir, output_filename)
 
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(menu_data, f, indent=2, ensure_ascii=False)
 
    print(f"Saved JSON to {output_path}")
    return output_path


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
        print("Usage: python3 script.py <menu_file>")
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
 
    # Create temp directory for individual page JSONs
    menu_dir = os.path.dirname(menu_file) or "."
    menu_name = os.path.splitext(os.path.basename(menu_file))[0]
    temp_dir = os.path.join(menu_dir, f".{menu_name}_pages")
    os.makedirs(temp_dir, exist_ok=True)
    
    print("Extracting menu data...")
    
    restaurant_name = None
    phone = None
    successful_pages = 0
 
    for i, path in enumerate(FILE_PATHS, 1):
        print(f"\n{'='*50}")
        print(f"Processing page {i}/{len(FILE_PATHS)}...")
        print(f"{'='*50}")
        menu_data = extract_menu_to_json(path, GROQ_API_KEY)
        
        if menu_data:
            # Save individual page JSON immediately
            page_json_path = os.path.join(temp_dir, f"page_{i}.json")
            with open(page_json_path, "w", encoding="utf-8") as f:
                json.dump(menu_data, f, indent=2, ensure_ascii=False)
            
            if not restaurant_name:
                restaurant_name = menu_data.get("restaurant_name")
            if not phone:
                phone = menu_data.get("phone")
            
            print(f"✓ Successfully extracted {len(menu_data.get('categories', []))} categories from page {i}")
            successful_pages += 1
        else:
            print(f"✗ Failed to extract data from page {i}")
    
    # Clean up temporary images if PDF was converted
    if is_pdf:
        print("\nCleaning up temporary image files...")
        cleanup_images(FILE_PATHS)
        print("✓ Temporary files removed")
    
    # Merge all page JSONs into final output
    if successful_pages > 0:
        all_categories = []
        page_files = sorted([f for f in os.listdir(temp_dir) if f.startswith("page_") and f.endswith(".json")])
        
        for page_file in page_files:
            with open(os.path.join(temp_dir, page_file), "r", encoding="utf-8") as f:
                page_data = json.load(f)
                all_categories.extend(page_data.get("categories", []))
        
        combined_menu = {
            "restaurant_name": restaurant_name,
            "phone": phone,
            "categories": all_categories
        }
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        print("\n" + "="*50)
        print("EXTRACTION COMPLETE")
        print("="*50)
 
        save_menu_json(combined_menu, menu_file)
 
        total_items = sum(len(cat["items"]) for cat in combined_menu.get("categories", []))
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Pages Processed: {successful_pages}/{len(FILE_PATHS)}")
        print(f"Categories Extracted: {len(combined_menu.get('categories', []))}")
        print(f"Total Items: {total_items}")
    else:
        print("\n✗ No data extracted from any page!")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)