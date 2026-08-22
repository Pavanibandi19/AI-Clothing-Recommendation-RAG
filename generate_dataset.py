import random
import pandas as pd
from pathlib import Path

# Seed for reproducibility
random.seed(42)

OUTPUT_FILE = Path(__file__).parent / "data" / "clothing_products.csv"

BRANDS = [
    "Levi's", "Zara", "H&M", "Nike", "Roadster", "FabIndia", 
    "Allen Solly", "Puma", "Adidas", "US Polo Assn", "Tommy Hilfiger", 
    "W", "Biba", "HRX", "Jack & Jones", "Mango", "Manyavar", "Aurelia", "Under Armour"
]

CATEGORIES_DATA = {
    "shirts": {
        "subcategories": ["Casual Button-Down", "Formal Dress Shirt", "Linen Resort Shirt", "Checked Flannel Shirt", "Oxford Cotton Shirt"],
        "genders": ["Men", "Women", "Unisex"],
        "fits": ["Slim Fit", "Regular Fit", "Relaxed Fit"],
        "materials": ["Cotton", "Linen", "Rayon", "Polyester"],
        "seasons": ["Summer", "Spring", "All-Season"],
        "base_price": (699, 3499)
    },
    "t-shirts": {
        "subcategories": ["Crewneck Graphic Tee", "Polo Collar T-Shirt", "Oversized Cotton Tee", "V-Neck Casual T-Shirt", "Henley Collar Tee"],
        "genders": ["Men", "Women", "Unisex"],
        "fits": ["Slim Fit", "Regular Fit", "Oversized"],
        "materials": ["Cotton", "Polyester", "Rayon"],
        "seasons": ["Summer", "All-Season"],
        "base_price": (399, 1999)
    },
    "jeans": {
        "subcategories": ["Slim Fit Denim Jeans", "Straight Leg Jeans", "Skinny Fit Stretch Jeans", "Relaxed Tapered Jeans", "High-Waist Mom Jeans"],
        "genders": ["Men", "Women", "Unisex"],
        "fits": ["Slim Fit", "Regular Fit", "Skinny Fit", "Relaxed Fit"],
        "materials": ["Denim", "Cotton"],
        "seasons": ["All-Season", "Winter", "Autumn"],
        "base_price": (1199, 4499)
    },
    "trousers": {
        "subcategories": ["Chino Trousers", "Formal Dress Trousers", "Cargo Pants", "Pleated Linen Trousers", "Slim Fit Slacks"],
        "genders": ["Men", "Women", "Unisex"],
        "fits": ["Slim Fit", "Regular Fit", "Relaxed Fit"],
        "materials": ["Cotton", "Linen", "Polyester", "Wool"],
        "seasons": ["All-Season", "Spring", "Autumn"],
        "base_price": (999, 3999)
    },
    "dresses": {
        "subcategories": ["Floral Summer Maxi Dress", "A-Line Cotton Dress", "Bodycon Evening Dress", "Casual Denim Shirt Dress", "Tiered Sun Dress"],
        "genders": ["Women"],
        "fits": ["Slim Fit", "Regular Fit", "Relaxed Fit"],
        "materials": ["Cotton", "Silk", "Rayon", "Linen"],
        "seasons": ["Summer", "Spring", "All-Season"],
        "base_price": (1299, 4999)
    },
    "skirts": {
        "subcategories": ["Pleated Midi Skirt", "Denim Mini Skirt", "A-Line High Waist Skirt", "Pencil Office Skirt", "Tiered Boho Skirt"],
        "genders": ["Women"],
        "fits": ["Slim Fit", "Regular Fit", "Relaxed Fit"],
        "materials": ["Cotton", "Denim", "Silk", "Rayon"],
        "seasons": ["Summer", "Spring", "All-Season"],
        "base_price": (799, 2999)
    },
    "jackets": {
        "subcategories": ["Classic Denim Jacket", "Biker Leather Jacket", "Puffer Winter Jacket", "Tailored Blazer", "Windbreaker Jacket"],
        "genders": ["Men", "Women", "Unisex"],
        "fits": ["Slim Fit", "Regular Fit", "Oversized"],
        "materials": ["Denim", "Leather", "Polyester", "Wool"],
        "seasons": ["Winter", "Autumn", "All-Season"],
        "base_price": (1999, 5999)
    },
    "hoodies": {
        "subcategories": ["Fleece Pullover Hoodie", "Zip-Up Hooded Sweatshirt", "Oversized Streetwear Hoodie", "Graphic Print Hoodie", "Heavyweight Warm Hoodie"],
        "genders": ["Men", "Women", "Unisex"],
        "fits": ["Regular Fit", "Oversized", "Relaxed Fit"],
        "materials": ["Cotton", "Polyester", "Wool"],
        "seasons": ["Winter", "Autumn"],
        "base_price": (1099, 3499)
    },
    "sweaters": {
        "subcategories": ["Cable Knit Pullover", "V-Neck Wool Sweater", "Cardigan Button Sweater", "Turtleneck Fine Sweater", "Ribbed Knit Top"],
        "genders": ["Men", "Women", "Unisex"],
        "fits": ["Slim Fit", "Regular Fit", "Relaxed Fit"],
        "materials": ["Wool", "Cotton"],
        "seasons": ["Winter", "Autumn"],
        "base_price": (1299, 3999)
    },
    "tops": {
        "subcategories": ["Peplum Casual Top", "Crop Ribbed Top", "Off-Shoulder Summer Top", "Satin V-Neck Blouse", "Printed Tunic Top"],
        "genders": ["Women"],
        "fits": ["Slim Fit", "Regular Fit", "Relaxed Fit"],
        "materials": ["Cotton", "Silk", "Rayon", "Polyester"],
        "seasons": ["Summer", "Spring", "All-Season"],
        "base_price": (499, 2499)
    },
    "shorts": {
        "subcategories": ["Casual Denim Shorts", "Cotton Chino Shorts", "Cargo Pocket Shorts", "Athletic Running Shorts", "Linen Summer Shorts"],
        "genders": ["Men", "Women", "Unisex"],
        "fits": ["Slim Fit", "Regular Fit", "Relaxed Fit"],
        "materials": ["Cotton", "Denim", "Polyester", "Linen"],
        "seasons": ["Summer", "Spring"],
        "base_price": (599, 2199)
    },
    "ethnic wear": {
        "subcategories": ["Straight Printed Kurta", "Anarkali Kurta Set", "Festive Nehru Jacket", "Embroidery Silk Kurta", "Ethnic Bandhgala Suit"],
        "genders": ["Men", "Women"],
        "fits": ["Regular Fit", "Slim Fit", "Relaxed Fit"],
        "materials": ["Cotton", "Silk", "Rayon"],
        "seasons": ["All-Season", "Spring", "Summer"],
        "base_price": (899, 4999)
    },
    "activewear": {
        "subcategories": ["Dry-Fit Workout Tee", "Running Shorts with Liner", "Compression Leggings", "Athletic Track Pants", "Performance Training Vest"],
        "genders": ["Men", "Women", "Unisex"],
        "fits": ["Slim Fit", "Regular Fit", "Skinny Fit"],
        "materials": ["Polyester", "Rayon", "Cotton"],
        "seasons": ["All-Season", "Summer"],
        "base_price": (699, 2999)
    }
}

COLORS = ["Black", "Blue", "White", "Red", "Green", "Navy", "Grey", "Beige", "Yellow", "Pink", "Olive", "Maroon", "Brown", "Purple"]
SIZES = ["S", "M", "L", "XL", "XXL"]

DESCRIPTIONS_TEMPLATES = [
    "Crafted from premium {material}, this {brand} {color} {subcategory} offers exceptional comfort and durability for {season} wardrobe styling. Features a sleek {fit} design.",
    "Elevate your look with this stylish {color} {subcategory} by {brand}. Designed in a comfortable {fit} using high-grade {material}, perfect for casual or semi-formal outings.",
    "Essential {brand} {gender}'s {subcategory} tailored with a modern {fit}. Made of soft, breathable {material} in rich {color}. Ideal choice for {season}.",
    "Stay fashion-forward with this versatile {color} {subcategory} from {brand}. Includes a refined {fit} profile and high quality {material} fabric.",
    "Top-rated {brand} {subcategory} featuring premium {color} {material}. Tailored to a comfortable {fit}, ideal for everyday wear during {season}."
]

def generate_products(total_count=3000):
    products = []
    category_keys = list(CATEGORIES_DATA.keys())
    
    # Calculate count per category to maintain balanced dataset
    per_category = total_count // len(category_keys)
    extra = total_count % len(category_keys)
    
    product_id_counter = 1001

    for cat_idx, cat in enumerate(category_keys):
        count = per_category + (1 if cat_idx < extra else 0)
        cat_info = CATEGORIES_DATA[cat]
        
        for _ in range(count):
            brand = random.choice(BRANDS)
            subcat = random.choice(cat_info["subcategories"])
            gender = random.choice(cat_info["genders"])
            color = random.choice(COLORS)
            fit = random.choice(cat_info["fits"])
            material = random.choice(cat_info["materials"])
            season = random.choice(cat_info["seasons"])
            size = random.choice(SIZES)
            
            min_p, max_p = cat_info["base_price"]
            # Round price to clean INR ends (e.g. 99, 49, 00)
            raw_price = random.randint(min_p, max_p)
            price_inr = (raw_price // 50) * 50 + 99
            
            rating = round(random.uniform(3.5, 5.0), 1)
            
            # Product Name format: Brand Gender Fit Color Material Subcategory
            product_name = f"{brand} {gender}'s {color} {fit} {subcat}"
            
            desc_template = random.choice(DESCRIPTIONS_TEMPLATES)
            description = desc_template.format(
                brand=brand,
                color=color,
                subcategory=subcat.lower(),
                fit=fit,
                material=material,
                season=season,
                gender=gender
            )
            
            products.append({
                "product_id": f"PRD{product_id_counter}",
                "product_name": product_name,
                "category": cat,
                "subcategory": subcat,
                "gender": gender,
                "color": color,
                "size": size,
                "fit": fit,
                "material": material,
                "brand": brand,
                "price_inr": price_inr,
                "rating": rating,
                "season": season,
                "description": description
            })
            
            product_id_counter += 1

    df = pd.DataFrame(products)
    
    # Create directory if it doesn't exist
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[SUCCESS] Generated dataset with {len(df)} products saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_products(3000)
