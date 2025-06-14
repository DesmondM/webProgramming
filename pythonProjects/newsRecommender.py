import json
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime
import random

# --- 1. Define Context ---
@dataclass
class UserContext:
    user_id: str
    past_purchases: List[str]  # Product IDs
    browsing_history: List[str]
    location: str
    current_time: datetime
    loyalty_tier: str  # bronze/silver/gold

@dataclass
class Product:
    id: str
    name: str
    category: str
    price: float
    stock: int

# --- 2. Model (Recommendation Engine) ---
class RecommendationModel:
    def __init__(self):
        self.product_db = {
            "electronics": [
                Product("p1", "Wireless Earbuds", "electronics", 99.99, 50),
                Product("p2", "Smart Watch", "electronics", 199.99, 25),
                Product("p3", "Television", "electronics", 400.99, 25),
                Product("p4", "Music System", "electronics", 250.99, 25),
                Product("p5", "Mobile Phone", "electronics", 80.99, 25)
            ],
            "clothing": [
                Product("p6", "Running Shoes", "clothing", 89.99, 100),
                Product("p7", "Winter Jacket", "clothing", 129.99, 30),
                Product("p8", "Track pants", "clothing", 129.99, 30),
                Product("p9", "Pullover", "clothing", 129.99, 30),
                Product("p10", "Hoodie", "clothing", 129.99, 30)
            ]
        }
    
    def recommend(self, context: UserContext) -> List[Product]:
        """Generates raw recommendations"""
        recs = []
        
        # 1. Recommend similar to past purchases
        for purchase in context.past_purchases:
            for category, products in self.product_db.items():
                recs.extend([p for p in products if p.id == purchase])
        
        # 2. Add popular items in user's location
        location_popular = {
            "Johannesburg": ["p1", "p4"],
            "Cape Town": ["p2", "p3"]
        }
        if context.location in location_popular:
            for pid in location_popular[context.location]:
                recs.extend([p for p in self.product_db["electronics"] + self.product_db["clothing"] 
                           if p.id == pid])
        
        return recs

# --- 3. Protocol (Business Rules) ---
class BusinessProtocol:
    def apply_rules(self, products: List[Product], context: UserContext) -> List[Product]:
        """Enforces business logic"""
        filtered = []
        
        # Rule 1: Hide out-of-stock items
        filtered = [p for p in products if p.stock > 0]
        
        # Rule 2: Prioritize high-margin items for loyalty members
        if context.loyalty_tier == "gold":
            filtered.sort(key=lambda x: x.price, reverse=True)
        
        # Rule 3: Seasonal promotion (e.g., winter jackets in Dec)
        month = context.current_time.month
        if month in [11, 12, 1]:  # Winter months
            filtered = [p for p in filtered if p.category == "clothing" and "Jacket" in p.name] + \
                      [p for p in filtered if p not in filtered]
        
        return filtered[:5]  # Return top 5

# --- 4. MCP Orchestrator ---
class EcommerceRecommender:
    def __init__(self):
        self.model = RecommendationModel()
        self.protocol = BusinessProtocol()
    
    def get_recommendations(self, context: UserContext) -> List[Product]:
        raw_recs = self.model.recommend(context)
        final_recs = self.protocol.apply_rules(raw_recs, context)
        return final_recs

# --- 5. Example Usage ---
if __name__ == "__main__":
    system = EcommerceRecommender()
    
    # Simulate user context
    user_ctx = UserContext(
        user_id="desmond123",
        past_purchases=["p9"],  # Bought earbuds before
        browsing_history=["p3"],
        location="Johannesburg",
        current_time=datetime(2025, 6, 12),  # June
        loyalty_tier="gold"
    )
    
    # Get recommendations
    recommendations = system.get_recommendations(user_ctx)
    
    print("🎁 Personalized Recommendations:")
    for i, product in enumerate(recommendations, 1):
        print(f"{i}. {product.name} (${product.price})")