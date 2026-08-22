import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
from src.config import DATASET_PATH

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else Path(DATASET_PATH)
        self.df = self.load_data()

    def load_data(self) -> pd.DataFrame:
        """Load clothing products CSV dataset and clean missing or invalid values."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}. Please run generate_dataset.py first.")

        df = pd.read_csv(self.dataset_path)

        # Standardize column names
        df.columns = [col.strip().lower() for col in df.columns]

        # Ensure required columns exist
        required_cols = [
            "product_id", "product_name", "category", "subcategory", 
            "gender", "color", "size", "fit", "material", "brand", 
            "price_inr", "rating", "season", "description"
        ]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column in dataset: {col}")

        # Invalid prices cannot be safely recommended, so exclude them explicitly.
        df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")
        invalid_price_mask = df["price_inr"].isna() | (df["price_inr"] <= 0)
        if invalid_price_mask.any():
            logger.warning("Excluding %d products with invalid prices.", int(invalid_price_mask.sum()))
            df = df.loc[~invalid_price_mask].copy()

        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(4.0)

        string_cols = ["product_id", "product_name", "category", "subcategory", "gender", "color", "size", "fit", "material", "brand", "season", "description"]
        for col in string_cols:
            df[col] = df[col].fillna("").astype(str).str.strip()

        return df

    def get_all_products(self) -> pd.DataFrame:
        """Returns complete product dataframe."""
        return self.df

    def filter_products(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Applies hard metadata filtering on the DataFrame.
        Filters dictionary may contain:
        - max_price (float/int)
        - min_price (float/int)
        - gender (str)
        - category (str)
        - subcategory (str)
        - color (str)
        - fit (str)
        - material (str)
        - season (str)
        - brand (str)
        """
        filtered_df = self.df.copy()

        # Hard Price Filtering (CRITICAL: products > max_price MUST BE EXCLUDED)
        if "max_price" in filters and filters["max_price"] is not None:
            max_p = float(filters["max_price"])
            filtered_df = filtered_df[filtered_df["price_inr"] <= max_p]

        if "min_price" in filters and filters["min_price"] is not None:
            min_p = float(filters["min_price"])
            filtered_df = filtered_df[filtered_df["price_inr"] >= min_p]

        # Hard Gender Filtering
        if "gender" in filters and filters["gender"]:
            target_gender = str(filters["gender"]).strip().lower()
            # Allow Unisex items if requested Men or Women unless strictly specified
            filtered_df = filtered_df[
                filtered_df["gender"].str.lower().isin([target_gender, "unisex"])
            ]

        # Hard Category Filtering
        if "category" in filters and filters["category"]:
            target_cat = str(filters["category"]).strip().lower()
            filtered_df = filtered_df[filtered_df["category"].str.lower() == target_cat]

        # Subcategory Filter
        if "subcategory" in filters and filters["subcategory"]:
            target_subcat = str(filters["subcategory"]).strip().lower()
            filtered_df = filtered_df[filtered_df["subcategory"].str.lower().str.contains(target_subcat, regex=False)]

        # Color Filter
        if "color" in filters and filters["color"]:
            target_color = str(filters["color"]).strip().lower()
            filtered_df = filtered_df[filtered_df["color"].str.lower() == target_color]

        # Fit Filter
        if "fit" in filters and filters["fit"]:
            target_fit = str(filters["fit"]).strip().lower()
            filtered_df = filtered_df[filtered_df["fit"].str.lower() == target_fit]

        # Material Filter
        if "material" in filters and filters["material"]:
            target_mat = str(filters["material"]).strip().lower()
            filtered_df = filtered_df[filtered_df["material"].str.lower() == target_mat]

        # Season Filter
        if "season" in filters and filters["season"]:
            target_season = str(filters["season"]).strip().lower()
            filtered_df = filtered_df[
                filtered_df["season"].str.lower().isin([target_season, "all-season"])
            ]

        # Brand Filter
        if "brand" in filters and filters["brand"]:
            target_brand = str(filters["brand"]).strip().lower()
            filtered_df = filtered_df[filtered_df["brand"].str.lower() == target_brand]

        return filtered_df

    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Returns single product dict matching product_id."""
        matches = self.df[self.df["product_id"] == product_id]
        if not matches.empty:
            return matches.iloc[0].to_dict()
        return None
