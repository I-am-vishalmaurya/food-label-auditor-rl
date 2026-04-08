#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Build static product dataset, adversarial cases, and ground truth table.

Generates ~500 Indian food products with realistic nutritional data,
30-40 adversarial cases with misleading marketing claims, and a
ground truth JSON for all (product, profile) pairs.

Data sources for real deployment:
  - OpenFoodFacts India (openfoodfacts.org)
  - ICMR-NIN Indian Food Composition Table
  - FSSAI Food Products Standards 2020

For the hackathon, this script generates a curated seed dataset
of well-known Indian products with realistic values, then expands
with synthetic variants to reach ~500 products.
"""

import json
import random
import copy
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ─── Known Indian products (seed data) ────────────────────────────────────────

SEED_PRODUCTS = [
    {
        "product_id": "IND_001",
        "product_name": "Maggi 2-Minute Noodles Masala",
        "brand": "Nestle",
        "category": "instant_noodles",
        "ingredients_text": "Refined wheat flour (maida), palm oil, iodised salt, wheat gluten, thickeners (508, 412), mineral (iron), acidity regulators (501(i), 500(ii)), humectant (451(i)), riboflavin, folic acid. Taste maker: mixed spices, sugar, salt, onion powder, flavour enhancer (627, 631), garlic powder, citric acid, turmeric",
        "nutrition_per_100g": {"energy_kcal": 430, "sugars_g": 2.5, "sodium_mg": 1120, "fat_g": 17.0},
        "marketing_claims": ["No added MSG"],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_002",
        "product_name": "Parle-G Gold Biscuits",
        "brand": "Parle",
        "category": "biscuits",
        "ingredients_text": "Wheat flour, sugar, edible vegetable oil (palm oil), invert syrup, milk solids, raising agents (503(ii), 500(ii)), salt, emulsifier (322(i)), dough conditioner (223)",
        "nutrition_per_100g": {"energy_kcal": 462, "sugars_g": 26.3, "sodium_mg": 281, "fat_g": 14.5},
        "marketing_claims": ["Glucose energy"],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_003",
        "product_name": "Amul Taaza Toned Milk",
        "brand": "Amul",
        "category": "dairy_milk",
        "ingredients_text": "Toned milk",
        "nutrition_per_100g": {"energy_kcal": 50, "sugars_g": 4.7, "sodium_mg": 52, "fat_g": 1.5},
        "marketing_claims": ["Source of calcium"],
        "nutri_score": "A",
        "nova_group": 1,
    },
    {
        "product_id": "IND_004",
        "product_name": "Cadbury Bournvita Health Drink",
        "brand": "Cadbury",
        "category": "health_drinks",
        "ingredients_text": "Sugar, cocoa solids, liquid glucose, malt extract, milk solids, caramel colour (150d), emulsifier (322(i)), vitamins, minerals, raising agent (500(ii)), salt",
        "nutrition_per_100g": {"energy_kcal": 377, "sugars_g": 37.4, "sodium_mg": 210, "fat_g": 2.0},
        "marketing_claims": ["Taller, Stronger, Sharper", "Scientifically proven nutrition"],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_005",
        "product_name": "Haldiram's Aloo Bhujia",
        "brand": "Haldiram's",
        "category": "namkeen_snacks",
        "ingredients_text": "Gram flour (besan), edible vegetable oil (palmolein oil), potato powder, spices & condiments (red chilli powder, black salt, dry mango powder, coriander), salt, mango powder",
        "nutrition_per_100g": {"energy_kcal": 526, "sugars_g": 1.8, "sodium_mg": 870, "fat_g": 30.0},
        "marketing_claims": [],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_006",
        "product_name": "Britannia Good Day Cashew Cookies",
        "brand": "Britannia",
        "category": "biscuits",
        "ingredients_text": "Refined wheat flour (maida), sugar, edible vegetable fat (palm oil, interesterified vegetable fat), cashew nut (5.2%), invert syrup, butter, milk solids, raising agents (500(ii), 503(ii)), iodised salt, emulsifier (322(i))",
        "nutrition_per_100g": {"energy_kcal": 484, "sugars_g": 25.0, "sodium_mg": 250, "fat_g": 20.5},
        "marketing_claims": ["Made with real cashews"],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_007",
        "product_name": "Mother Dairy Dahi (Curd)",
        "brand": "Mother Dairy",
        "category": "dairy_curd",
        "ingredients_text": "Pasteurised toned milk, lactic culture",
        "nutrition_per_100g": {"energy_kcal": 56, "sugars_g": 3.0, "sodium_mg": 40, "fat_g": 3.0},
        "marketing_claims": ["Probiotic"],
        "nutri_score": "A",
        "nova_group": 1,
    },
    {
        "product_id": "IND_008",
        "product_name": "Lays Classic Salted Chips",
        "brand": "Lay's",
        "category": "chips",
        "ingredients_text": "Potato, edible vegetable oil (palmolein), iodised salt",
        "nutrition_per_100g": {"energy_kcal": 536, "sugars_g": 0.5, "sodium_mg": 650, "fat_g": 33.0},
        "marketing_claims": [],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_009",
        "product_name": "Dabur Real Mango Juice",
        "brand": "Dabur",
        "category": "fruit_juice",
        "ingredients_text": "Water, mango pulp (25%), sugar, citric acid, mango flavour, preservative (211), antioxidant (300)",
        "nutrition_per_100g": {"energy_kcal": 60, "sugars_g": 14.0, "sodium_mg": 15, "fat_g": 0.0},
        "marketing_claims": ["No added preservatives", "Made with real fruit"],
        "nutri_score": "C",
        "nova_group": 4,
    },
    {
        "product_id": "IND_010",
        "product_name": "MTR Ready to Eat Rajma Masala",
        "brand": "MTR",
        "category": "ready_to_eat",
        "ingredients_text": "Water, kidney beans (rajma) (28%), tomato paste, onion, edible vegetable oil (sunflower), salt, garlic, ginger, spices (red chilli, turmeric, coriander, cumin, garam masala), sugar, acidity regulator (330)",
        "nutrition_per_100g": {"energy_kcal": 108, "sugars_g": 2.5, "sodium_mg": 480, "fat_g": 4.0},
        "marketing_claims": ["Ready in 3 minutes"],
        "nutri_score": "B",
        "nova_group": 4,
    },
    {
        "product_id": "IND_011",
        "product_name": "Aashirvaad Whole Wheat Atta",
        "brand": "ITC",
        "category": "flour",
        "ingredients_text": "100% whole wheat (gehun ka atta)",
        "nutrition_per_100g": {"energy_kcal": 341, "sugars_g": 1.5, "sodium_mg": 5, "fat_g": 1.7},
        "marketing_claims": ["100% whole wheat", "No maida"],
        "nutri_score": "A",
        "nova_group": 1,
    },
    {
        "product_id": "IND_012",
        "product_name": "Kurkure Masala Munch",
        "brand": "PepsiCo",
        "category": "namkeen_snacks",
        "ingredients_text": "Rice meal, edible vegetable oil (palmolein), corn meal, gram meal, spices & condiments, salt, wheat flour, sugar, black gram flour, MSG (flavour enhancer 621), citric acid, tartaric acid",
        "nutrition_per_100g": {"energy_kcal": 510, "sugars_g": 3.0, "sodium_mg": 920, "fat_g": 28.0},
        "marketing_claims": ["Tedha hai par mera hai"],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_013",
        "product_name": "Tata Salt",
        "brand": "Tata",
        "category": "condiments",
        "ingredients_text": "Iodised salt, potassium iodate",
        "nutrition_per_100g": {"energy_kcal": 0, "sugars_g": 0.0, "sodium_mg": 38758, "fat_g": 0.0},
        "marketing_claims": ["Vacuum evaporated", "Desh ka namak"],
        "nutri_score": "E",
        "nova_group": 2,
    },
    {
        "product_id": "IND_014",
        "product_name": "Coca-Cola Classic",
        "brand": "Coca-Cola",
        "category": "carbonated_drinks",
        "ingredients_text": "Carbonated water, sugar, colour (150d), acidulant (338), natural flavouring substances, caffeine",
        "nutrition_per_100g": {"energy_kcal": 42, "sugars_g": 10.6, "sodium_mg": 6, "fat_g": 0.0},
        "marketing_claims": ["Taste the feeling"],
        "nutri_score": "E",
        "nova_group": 4,
    },
    {
        "product_id": "IND_015",
        "product_name": "Paper Boat Aam Panna",
        "brand": "Paper Boat",
        "category": "fruit_drinks",
        "ingredients_text": "Water, raw mango pulp (12%), sugar, jaggery, salt, spices (cumin, black pepper, mint), preservative (211, 224)",
        "nutrition_per_100g": {"energy_kcal": 38, "sugars_g": 8.5, "sodium_mg": 180, "fat_g": 0.0},
        "marketing_claims": ["Drinks and memories"],
        "nutri_score": "C",
        "nova_group": 4,
    },
    {
        "product_id": "IND_016",
        "product_name": "Saffola Gold Refined Oil",
        "brand": "Saffola",
        "category": "cooking_oil",
        "ingredients_text": "Rice bran oil, sunflower oil, natural antioxidant (mixed tocopherols, rosemary extract)",
        "nutrition_per_100g": {"energy_kcal": 900, "sugars_g": 0.0, "sodium_mg": 0, "fat_g": 100.0},
        "marketing_claims": ["Heart healthy", "Losorb technology"],
        "nutri_score": "C",
        "nova_group": 2,
    },
    {
        "product_id": "IND_017",
        "product_name": "Amul Cheese Slices",
        "brand": "Amul",
        "category": "dairy_cheese",
        "ingredients_text": "Cheese (70%), water, sodium citrate, common salt, citric acid, permitted natural colour (annatto)",
        "nutrition_per_100g": {"energy_kcal": 310, "sugars_g": 2.0, "sodium_mg": 1100, "fat_g": 25.0},
        "marketing_claims": ["100% vegetarian"],
        "nutri_score": "D",
        "nova_group": 3,
    },
    {
        "product_id": "IND_018",
        "product_name": "Patanjali Cow Ghee",
        "brand": "Patanjali",
        "category": "dairy_ghee",
        "ingredients_text": "Cow milk fat",
        "nutrition_per_100g": {"energy_kcal": 900, "sugars_g": 0.0, "sodium_mg": 0, "fat_g": 99.7},
        "marketing_claims": ["Pure desi ghee", "Natural"],
        "nutri_score": "C",
        "nova_group": 2,
    },
    {
        "product_id": "IND_019",
        "product_name": "Kellogg's Chocos",
        "brand": "Kellogg's",
        "category": "breakfast_cereal",
        "ingredients_text": "Cereals (wheat flour, rice flour), sugar, cocoa solids (5%), edible vegetable oil (palm), malt extract, salt, minerals (iron, zinc), vitamins (niacin, B6, riboflavin, thiamin, folic acid, B12), emulsifier (322(i)), antioxidant (320)",
        "nutrition_per_100g": {"energy_kcal": 394, "sugars_g": 30.0, "sodium_mg": 350, "fat_g": 5.0},
        "marketing_claims": ["With iron", "Protein power"],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_020",
        "product_name": "Tropicana 100% Orange Juice",
        "brand": "Tropicana",
        "category": "fruit_juice",
        "ingredients_text": "Orange juice from concentrate, water, added vitamins (C)",
        "nutrition_per_100g": {"energy_kcal": 45, "sugars_g": 10.0, "sodium_mg": 5, "fat_g": 0.0},
        "marketing_claims": ["100% juice", "No added sugar"],
        "nutri_score": "C",
        "nova_group": 3,
    },
    {
        "product_id": "IND_021",
        "product_name": "Bikaji Bhujia Sev",
        "brand": "Bikaji",
        "category": "namkeen_snacks",
        "ingredients_text": "Gram flour (besan), edible vegetable oil (palmolein), moth bean flour, spices, salt, asafoetida",
        "nutrition_per_100g": {"energy_kcal": 545, "sugars_g": 2.0, "sodium_mg": 780, "fat_g": 33.0},
        "marketing_claims": [],
        "nutri_score": "D",
        "nova_group": 3,
    },
    {
        "product_id": "IND_022",
        "product_name": "Kissan Mixed Fruit Jam",
        "brand": "Kissan",
        "category": "jams_spreads",
        "ingredients_text": "Sugar, mixed fruit pulp (apple, pineapple, papaya, banana) (30%), water, citric acid, pectin, preservative (224), colour (129)",
        "nutrition_per_100g": {"energy_kcal": 268, "sugars_g": 63.5, "sodium_mg": 30, "fat_g": 0.1},
        "marketing_claims": ["Made with real fruit"],
        "nutri_score": "E",
        "nova_group": 4,
    },
    {
        "product_id": "IND_023",
        "product_name": "Tata Tea Gold",
        "brand": "Tata",
        "category": "tea",
        "ingredients_text": "Tea leaves (15% long leaf)",
        "nutrition_per_100g": {"energy_kcal": 0, "sugars_g": 0.0, "sodium_mg": 0, "fat_g": 0.0},
        "marketing_claims": ["15% long leaf blend"],
        "nutri_score": "A",
        "nova_group": 1,
    },
    {
        "product_id": "IND_024",
        "product_name": "Maggi Hot & Sweet Sauce",
        "brand": "Nestle",
        "category": "sauces",
        "ingredients_text": "Tomato paste, sugar, vinegar, salt, onion, garlic, modified starch, chilli, spices, preservative (211), acidity regulator (260), stabiliser (415)",
        "nutrition_per_100g": {"energy_kcal": 150, "sugars_g": 28.0, "sodium_mg": 1480, "fat_g": 0.5},
        "marketing_claims": ["Taste bhi, health bhi"],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_025",
        "product_name": "Sundrop Superlite Advanced Sunflower Oil",
        "brand": "Sundrop",
        "category": "cooking_oil",
        "ingredients_text": "Refined sunflower oil, natural antioxidant (tocopherol rich extract)",
        "nutrition_per_100g": {"energy_kcal": 900, "sugars_g": 0.0, "sodium_mg": 0, "fat_g": 100.0},
        "marketing_claims": ["Light on stomach", "High in vitamin E"],
        "nutri_score": "C",
        "nova_group": 2,
    },
    {
        "product_id": "IND_026",
        "product_name": "Thums Up",
        "brand": "Coca-Cola",
        "category": "carbonated_drinks",
        "ingredients_text": "Carbonated water, sugar, colour (150d), acidulant (338), caffeine, natural flavouring substances",
        "nutrition_per_100g": {"energy_kcal": 44, "sugars_g": 11.0, "sodium_mg": 18, "fat_g": 0.0},
        "marketing_claims": ["Toofani thanda"],
        "nutri_score": "E",
        "nova_group": 4,
    },
    {
        "product_id": "IND_027",
        "product_name": "Real Fruit Power Mixed Fruit",
        "brand": "Dabur",
        "category": "fruit_juice",
        "ingredients_text": "Water, sugar, mixed fruit pulp (apple, mango, pineapple, banana, grape, papaya), citric acid, preservative (211), antioxidant (300)",
        "nutrition_per_100g": {"energy_kcal": 55, "sugars_g": 12.5, "sodium_mg": 12, "fat_g": 0.0},
        "marketing_claims": ["No added preservatives"],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_028",
        "product_name": "Amul Butter",
        "brand": "Amul",
        "category": "dairy_butter",
        "ingredients_text": "Pasteurised cream, common salt, permitted natural colour (annatto)",
        "nutrition_per_100g": {"energy_kcal": 729, "sugars_g": 0.5, "sodium_mg": 700, "fat_g": 80.0},
        "marketing_claims": ["Utterly butterly delicious"],
        "nutri_score": "D",
        "nova_group": 2,
    },
    {
        "product_id": "IND_029",
        "product_name": "Britannia Marie Gold Biscuits",
        "brand": "Britannia",
        "category": "biscuits",
        "ingredients_text": "Wheat flour, sugar, edible vegetable oil (palm oil), invert syrup, milk solids, raising agents (503(ii), 500(ii)), iodised salt, emulsifier (322(i))",
        "nutrition_per_100g": {"energy_kcal": 440, "sugars_g": 20.0, "sodium_mg": 320, "fat_g": 12.0},
        "marketing_claims": ["Healthy snack"],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_030",
        "product_name": "MDH Chana Masala",
        "brand": "MDH",
        "category": "spice_mix",
        "ingredients_text": "Coriander, red chilli, turmeric, salt, black pepper, cumin, dry ginger, fenugreek, mango powder, pomegranate seed powder, black cardamom, cardamom, cinnamon, clove, bay leaf, nutmeg, mace",
        "nutrition_per_100g": {"energy_kcal": 280, "sugars_g": 5.0, "sodium_mg": 2800, "fat_g": 8.0},
        "marketing_claims": ["Asli masale sach sach"],
        "nutri_score": "C",
        "nova_group": 2,
    },
    {
        "product_id": "IND_031",
        "product_name": "Patanjali Dant Kanti Toothpaste",
        "brand": "Patanjali",
        "category": "oral_care",
        "ingredients_text": "Calcium carbonate, water, sorbitol, hydrated silica, glycerin, herbal extracts (neem, babool, pudina, tomar, akarkara), sodium lauryl sulfate, flavour, sodium carboxymethyl cellulose, sodium saccharin, preservative",
        "nutrition_per_100g": {"energy_kcal": 0, "sugars_g": 0.0, "sodium_mg": 0, "fat_g": 0.0},
        "marketing_claims": ["Herbal", "Ayurvedic"],
        "nutri_score": "NA",
        "nova_group": 0,
    },
    {
        "product_id": "IND_032",
        "product_name": "Frooti Mango Drink",
        "brand": "Parle Agro",
        "category": "fruit_drinks",
        "ingredients_text": "Water, sugar, mango pulp (13.5%), citric acid, mango flavour, preservative (211), antioxidant (300), colour (110)",
        "nutrition_per_100g": {"energy_kcal": 58, "sugars_g": 13.5, "sodium_mg": 10, "fat_g": 0.0},
        "marketing_claims": ["Fresh 'n' juicy"],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_033",
        "product_name": "Lijjat Papad",
        "brand": "Lijjat",
        "category": "papad",
        "ingredients_text": "Black gram flour (urad dal), edible common salt, black pepper, asafoetida, edible oil (groundnut), papad khar",
        "nutrition_per_100g": {"energy_kcal": 312, "sugars_g": 1.0, "sodium_mg": 1200, "fat_g": 2.0},
        "marketing_claims": [],
        "nutri_score": "C",
        "nova_group": 3,
    },
    {
        "product_id": "IND_034",
        "product_name": "Nescafe Classic Instant Coffee",
        "brand": "Nestle",
        "category": "coffee",
        "ingredients_text": "Spray dried instant coffee powder (100% arabica and robusta blend)",
        "nutrition_per_100g": {"energy_kcal": 0, "sugars_g": 0.0, "sodium_mg": 0, "fat_g": 0.0},
        "marketing_claims": ["100% pure coffee"],
        "nutri_score": "A",
        "nova_group": 3,
    },
    {
        "product_id": "IND_035",
        "product_name": "Everest Kitchen King Masala",
        "brand": "Everest",
        "category": "spice_mix",
        "ingredients_text": "Coriander, red chilli, turmeric, cumin, black pepper, dried fenugreek leaves, dry ginger, fennel, cinnamon, cardamom, clove, bay leaf, nutmeg, mace, salt",
        "nutrition_per_100g": {"energy_kcal": 270, "sugars_g": 4.0, "sodium_mg": 2500, "fat_g": 7.5},
        "marketing_claims": ["Taste ka tadka"],
        "nutri_score": "C",
        "nova_group": 2,
    },
    {
        "product_id": "IND_036",
        "product_name": "Maaza Mango Drink",
        "brand": "Coca-Cola",
        "category": "fruit_drinks",
        "ingredients_text": "Water, sugar, mango pulp (14%), citric acid, preservative (211), antioxidant (300)",
        "nutrition_per_100g": {"energy_kcal": 55, "sugars_g": 12.8, "sodium_mg": 8, "fat_g": 0.0},
        "marketing_claims": ["Har mausam aam"],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_037",
        "product_name": "Amul Kool Chocolate Milk",
        "brand": "Amul",
        "category": "flavoured_milk",
        "ingredients_text": "Toned milk, sugar, cocoa powder (1.5%), stabiliser (407), emulsifier (471)",
        "nutrition_per_100g": {"energy_kcal": 75, "sugars_g": 10.5, "sodium_mg": 60, "fat_g": 2.0},
        "marketing_claims": ["Real milk, real taste"],
        "nutri_score": "C",
        "nova_group": 4,
    },
    {
        "product_id": "IND_038",
        "product_name": "Act II Classic Salted Popcorn",
        "brand": "Conagra",
        "category": "popcorn",
        "ingredients_text": "Corn kernels, edible vegetable oil (palm), iodised salt, butter flavour, colour (160a, 100)",
        "nutrition_per_100g": {"energy_kcal": 480, "sugars_g": 1.0, "sodium_mg": 750, "fat_g": 24.0},
        "marketing_claims": ["Movie time snack"],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_039",
        "product_name": "Britannia NutriChoice Digestive Biscuits",
        "brand": "Britannia",
        "category": "biscuits",
        "ingredients_text": "Wheat flour (50%), edible vegetable fat (palm oil), sugar, wheat bran (10%), liquid glucose, invert syrup, raising agents (500(ii), 503(ii)), iodised salt, emulsifier (322(i)), malt extract",
        "nutrition_per_100g": {"energy_kcal": 460, "sugars_g": 18.0, "sodium_mg": 400, "fat_g": 18.0},
        "marketing_claims": ["Fibre rich", "NutriChoice", "Healthy biscuit"],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_040",
        "product_name": "Tata Sampann Unpolished Dal Moong",
        "brand": "Tata",
        "category": "pulses",
        "ingredients_text": "Unpolished moong dal (green gram)",
        "nutrition_per_100g": {"energy_kcal": 348, "sugars_g": 2.0, "sodium_mg": 15, "fat_g": 1.2},
        "marketing_claims": ["Unpolished", "Rich in protein"],
        "nutri_score": "A",
        "nova_group": 1,
    },
    {
        "product_id": "IND_041",
        "product_name": "Wai Wai Noodles",
        "brand": "CG Foods",
        "category": "instant_noodles",
        "ingredients_text": "Wheat flour, edible vegetable oil (palm), salt, flavour enhancer (621), sugar, onion powder, garlic powder, spices, citric acid, colour (110, 124)",
        "nutrition_per_100g": {"energy_kcal": 440, "sugars_g": 3.0, "sodium_mg": 1050, "fat_g": 18.0},
        "marketing_claims": ["Quick snack"],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_042",
        "product_name": "Horlicks Classic Malt",
        "brand": "HUL",
        "category": "health_drinks",
        "ingredients_text": "Wheat flour, malted barley, milk solids, sugar, minerals, cocoa solids, vitamins, barley malt extract, salt",
        "nutrition_per_100g": {"energy_kcal": 377, "sugars_g": 32.0, "sodium_mg": 260, "fat_g": 2.0},
        "marketing_claims": ["Taller, Stronger, Sharper", "Clinically proven"],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_043",
        "product_name": "Catch Italian Seasoning",
        "brand": "Catch",
        "category": "spice_mix",
        "ingredients_text": "Oregano, thyme, basil, rosemary, marjoram, sage",
        "nutrition_per_100g": {"energy_kcal": 265, "sugars_g": 4.0, "sodium_mg": 50, "fat_g": 4.0},
        "marketing_claims": [],
        "nutri_score": "A",
        "nova_group": 1,
    },
    {
        "product_id": "IND_044",
        "product_name": "Yakult Probiotic Drink",
        "brand": "Yakult",
        "category": "probiotic_drinks",
        "ingredients_text": "Water, skimmed milk powder, glucose fructose syrup, sugar, flavour, Lactobacillus casei strain Shirota",
        "nutrition_per_100g": {"energy_kcal": 66, "sugars_g": 14.2, "sodium_mg": 18, "fat_g": 0.0},
        "marketing_claims": ["6.5 billion Shirota strain per bottle"],
        "nutri_score": "C",
        "nova_group": 4,
    },
    {
        "product_id": "IND_045",
        "product_name": "Saffola Fittify Hi Protein Soup",
        "brand": "Saffola",
        "category": "ready_to_eat",
        "ingredients_text": "Soy protein isolate, corn starch, sugar, dehydrated vegetables (spinach, corn), iodised salt, spices, whey protein concentrate, milk solids, flavour enhancer (621, 627, 631)",
        "nutrition_per_100g": {"energy_kcal": 330, "sugars_g": 12.0, "sodium_mg": 1800, "fat_g": 3.0},
        "marketing_claims": ["Hi protein", "Low calorie", "Healthy soup"],
        "nutri_score": "C",
        "nova_group": 4,
    },
    {
        "product_id": "IND_046",
        "product_name": "Sundrop Peanut Butter Creamy",
        "brand": "Sundrop",
        "category": "nut_butter",
        "ingredients_text": "Roasted peanuts (85%), edible vegetable oil (palm), sugar, iodised salt, emulsifier (471)",
        "nutrition_per_100g": {"energy_kcal": 597, "sugars_g": 8.0, "sodium_mg": 350, "fat_g": 49.0},
        "marketing_claims": ["High protein", "No trans fat"],
        "nutri_score": "C",
        "nova_group": 3,
    },
    {
        "product_id": "IND_047",
        "product_name": "Nestle Milkmaid Condensed Milk",
        "brand": "Nestle",
        "category": "dairy_condensed",
        "ingredients_text": "Milk, sugar",
        "nutrition_per_100g": {"energy_kcal": 332, "sugars_g": 55.0, "sodium_mg": 150, "fat_g": 9.0},
        "marketing_claims": ["Original sweetened"],
        "nutri_score": "E",
        "nova_group": 3,
    },
    {
        "product_id": "IND_048",
        "product_name": "Bingo Mad Angles Achari Masti",
        "brand": "ITC",
        "category": "chips",
        "ingredients_text": "Corn grits, edible vegetable oil (palmolein), seasoning (sugar, iodised salt, citric acid, spice extracts, dried mango powder, flavour enhancer (621), maltodextrin)",
        "nutrition_per_100g": {"energy_kcal": 520, "sugars_g": 5.0, "sodium_mg": 800, "fat_g": 28.0},
        "marketing_claims": ["No artificial colours"],
        "nutri_score": "D",
        "nova_group": 4,
    },
    {
        "product_id": "IND_049",
        "product_name": "Raw Pressed Cold Press Apple Juice",
        "brand": "Raw Pressed",
        "category": "fruit_juice",
        "ingredients_text": "Cold pressed apple juice (100%)",
        "nutrition_per_100g": {"energy_kcal": 46, "sugars_g": 10.4, "sodium_mg": 3, "fat_g": 0.0},
        "marketing_claims": ["100% natural", "No preservatives", "Cold pressed"],
        "nutri_score": "B",
        "nova_group": 1,
    },
    {
        "product_id": "IND_050",
        "product_name": "Epigamia Greek Yogurt Plain",
        "brand": "Epigamia",
        "category": "dairy_yogurt",
        "ingredients_text": "Pasteurised toned milk, milk protein concentrate, live bacterial cultures",
        "nutrition_per_100g": {"energy_kcal": 59, "sugars_g": 3.5, "sodium_mg": 40, "fat_g": 2.5},
        "marketing_claims": ["High protein", "No preservatives"],
        "nutri_score": "A",
        "nova_group": 1,
    },
]

# ─── Categories for synthetic expansion ───────────────────────────────────────

CATEGORIES = {
    "instant_noodles": {"nutri_score_range": ["C", "D", "E"], "nova_range": [3, 4]},
    "biscuits": {"nutri_score_range": ["C", "D", "E"], "nova_range": [3, 4]},
    "dairy_milk": {"nutri_score_range": ["A", "B"], "nova_range": [1, 2]},
    "health_drinks": {"nutri_score_range": ["C", "D"], "nova_range": [3, 4]},
    "namkeen_snacks": {"nutri_score_range": ["C", "D", "E"], "nova_range": [3, 4]},
    "chips": {"nutri_score_range": ["D", "E"], "nova_range": [4]},
    "fruit_juice": {"nutri_score_range": ["B", "C", "D"], "nova_range": [1, 3, 4]},
    "fruit_drinks": {"nutri_score_range": ["C", "D"], "nova_range": [4]},
    "carbonated_drinks": {"nutri_score_range": ["D", "E"], "nova_range": [4]},
    "ready_to_eat": {"nutri_score_range": ["B", "C", "D"], "nova_range": [3, 4]},
    "flour": {"nutri_score_range": ["A", "B"], "nova_range": [1]},
    "spice_mix": {"nutri_score_range": ["A", "B", "C"], "nova_range": [1, 2]},
    "condiments": {"nutri_score_range": ["C", "D", "E"], "nova_range": [2, 3]},
    "sauces": {"nutri_score_range": ["C", "D"], "nova_range": [3, 4]},
    "cooking_oil": {"nutri_score_range": ["C", "D"], "nova_range": [2]},
    "dairy_cheese": {"nutri_score_range": ["C", "D"], "nova_range": [2, 3]},
    "dairy_butter": {"nutri_score_range": ["C", "D"], "nova_range": [2]},
    "dairy_ghee": {"nutri_score_range": ["C", "D"], "nova_range": [2]},
    "dairy_curd": {"nutri_score_range": ["A", "B"], "nova_range": [1]},
    "dairy_yogurt": {"nutri_score_range": ["A", "B"], "nova_range": [1, 3]},
    "breakfast_cereal": {"nutri_score_range": ["B", "C", "D"], "nova_range": [3, 4]},
    "pulses": {"nutri_score_range": ["A"], "nova_range": [1]},
    "tea": {"nutri_score_range": ["A"], "nova_range": [1]},
    "coffee": {"nutri_score_range": ["A"], "nova_range": [1, 3]},
    "jams_spreads": {"nutri_score_range": ["D", "E"], "nova_range": [4]},
    "nut_butter": {"nutri_score_range": ["B", "C"], "nova_range": [2, 3]},
    "probiotic_drinks": {"nutri_score_range": ["B", "C"], "nova_range": [3, 4]},
    "flavoured_milk": {"nutri_score_range": ["C", "D"], "nova_range": [3, 4]},
    "popcorn": {"nutri_score_range": ["C", "D"], "nova_range": [3, 4]},
    "dairy_condensed": {"nutri_score_range": ["D", "E"], "nova_range": [3]},
    "papad": {"nutri_score_range": ["B", "C"], "nova_range": [2, 3]},
}

BRAND_POOL = [
    "Nestle", "Parle", "Amul", "Britannia", "ITC", "Haldiram's", "Bikaji",
    "Mother Dairy", "Cadbury", "PepsiCo", "Coca-Cola", "Dabur", "MTR",
    "Patanjali", "Kellogg's", "HUL", "Tata", "Everest", "MDH", "Saffola",
    "Sundrop", "Paper Boat", "CG Foods", "Conagra", "Epigamia", "Yakult",
    "Kissan", "Tropicana", "Godrej", "Marico", "Heritage", "Nandini",
]

NUTRITION_RANGES = {
    "instant_noodles": {"energy_kcal": (380, 460), "sugars_g": (1, 5), "sodium_mg": (800, 1200), "fat_g": (13, 20)},
    "biscuits": {"energy_kcal": (420, 500), "sugars_g": (15, 30), "sodium_mg": (200, 450), "fat_g": (12, 22)},
    "dairy_milk": {"energy_kcal": (40, 65), "sugars_g": (3, 5), "sodium_mg": (30, 60), "fat_g": (1, 4)},
    "health_drinks": {"energy_kcal": (350, 400), "sugars_g": (25, 40), "sodium_mg": (150, 300), "fat_g": (1, 3)},
    "namkeen_snacks": {"energy_kcal": (480, 560), "sugars_g": (1, 5), "sodium_mg": (600, 1000), "fat_g": (25, 35)},
    "chips": {"energy_kcal": (500, 560), "sugars_g": (0, 5), "sodium_mg": (500, 900), "fat_g": (28, 36)},
    "fruit_juice": {"energy_kcal": (40, 65), "sugars_g": (8, 14), "sodium_mg": (2, 20), "fat_g": (0, 0.5)},
    "fruit_drinks": {"energy_kcal": (35, 60), "sugars_g": (8, 14), "sodium_mg": (5, 20), "fat_g": (0, 0.2)},
    "carbonated_drinks": {"energy_kcal": (38, 48), "sugars_g": (9, 12), "sodium_mg": (5, 25), "fat_g": (0, 0)},
    "ready_to_eat": {"energy_kcal": (80, 160), "sugars_g": (1, 5), "sodium_mg": (350, 650), "fat_g": (2, 8)},
    "flour": {"energy_kcal": (330, 350), "sugars_g": (1, 3), "sodium_mg": (3, 10), "fat_g": (1, 3)},
    "spice_mix": {"energy_kcal": (250, 300), "sugars_g": (2, 8), "sodium_mg": (1500, 3500), "fat_g": (5, 12)},
    "condiments": {"energy_kcal": (0, 10), "sugars_g": (0, 1), "sodium_mg": (30000, 40000), "fat_g": (0, 0)},
    "sauces": {"energy_kcal": (100, 200), "sugars_g": (15, 35), "sodium_mg": (800, 1800), "fat_g": (0, 2)},
    "cooking_oil": {"energy_kcal": (900, 900), "sugars_g": (0, 0), "sodium_mg": (0, 0), "fat_g": (100, 100)},
    "dairy_cheese": {"energy_kcal": (280, 350), "sugars_g": (1, 3), "sodium_mg": (800, 1200), "fat_g": (20, 28)},
    "dairy_butter": {"energy_kcal": (700, 750), "sugars_g": (0, 1), "sodium_mg": (500, 800), "fat_g": (75, 82)},
    "dairy_ghee": {"energy_kcal": (890, 910), "sugars_g": (0, 0), "sodium_mg": (0, 5), "fat_g": (98, 100)},
    "dairy_curd": {"energy_kcal": (45, 65), "sugars_g": (2, 5), "sodium_mg": (30, 50), "fat_g": (2, 4)},
    "dairy_yogurt": {"energy_kcal": (50, 100), "sugars_g": (3, 15), "sodium_mg": (30, 60), "fat_g": (1, 5)},
    "breakfast_cereal": {"energy_kcal": (360, 420), "sugars_g": (15, 35), "sodium_mg": (250, 500), "fat_g": (3, 10)},
    "pulses": {"energy_kcal": (330, 360), "sugars_g": (1, 4), "sodium_mg": (5, 20), "fat_g": (0.5, 2)},
    "tea": {"energy_kcal": (0, 5), "sugars_g": (0, 0), "sodium_mg": (0, 5), "fat_g": (0, 0)},
    "coffee": {"energy_kcal": (0, 5), "sugars_g": (0, 0), "sodium_mg": (0, 5), "fat_g": (0, 0)},
    "jams_spreads": {"energy_kcal": (240, 280), "sugars_g": (55, 70), "sodium_mg": (10, 50), "fat_g": (0, 0.5)},
    "nut_butter": {"energy_kcal": (570, 620), "sugars_g": (5, 12), "sodium_mg": (200, 400), "fat_g": (45, 55)},
    "probiotic_drinks": {"energy_kcal": (50, 75), "sugars_g": (10, 16), "sodium_mg": (10, 30), "fat_g": (0, 1)},
    "flavoured_milk": {"energy_kcal": (60, 90), "sugars_g": (8, 14), "sodium_mg": (40, 80), "fat_g": (1, 3)},
    "popcorn": {"energy_kcal": (440, 500), "sugars_g": (0, 3), "sodium_mg": (600, 900), "fat_g": (20, 28)},
    "dairy_condensed": {"energy_kcal": (310, 350), "sugars_g": (50, 58), "sodium_mg": (100, 200), "fat_g": (7, 12)},
    "papad": {"energy_kcal": (290, 330), "sugars_g": (0, 2), "sodium_mg": (900, 1400), "fat_g": (1, 4)},
}

INGREDIENT_TEMPLATES = {
    "instant_noodles": "Wheat flour, edible vegetable oil ({oil}), salt, spices, flavour enhancer ({fe}), sugar, {extras}",
    "biscuits": "Wheat flour, sugar, edible vegetable oil (palm oil), {extras}, raising agents, salt, emulsifier (322(i))",
    "chips": "Potato, edible vegetable oil ({oil}), iodised salt, {seasoning}",
    "carbonated_drinks": "Carbonated water, sugar, colour (150d), acidulant (338), natural flavouring substances, caffeine",
    "fruit_drinks": "Water, sugar, {fruit} pulp ({pct}%), citric acid, flavour, preservative (211), antioxidant (300)",
    "fruit_juice": "Water, {fruit} pulp ({pct}%), sugar, citric acid, preservative (211), antioxidant (300)",
}

SUGAR_SYNONYMS = [
    "maltodextrin", "dextrose", "corn syrup solids", "glucose syrup",
    "invert syrup", "fructose", "high fructose corn syrup",
]


def _rand_nutrition(rng: random.Random, cat: str) -> dict:
    """Generate random nutrition values within category range."""
    ranges = NUTRITION_RANGES.get(cat, NUTRITION_RANGES["biscuits"])
    return {
        "energy_kcal": round(rng.uniform(*ranges["energy_kcal"]), 1),
        "sugars_g": round(rng.uniform(*ranges["sugars_g"]), 1),
        "sodium_mg": round(rng.uniform(*ranges["sodium_mg"]), 1),
        "fat_g": round(rng.uniform(*ranges["fat_g"]), 1),
    }


def _generate_synthetic_products(rng: random.Random, count: int) -> list[dict]:
    """Generate synthetic products to expand dataset to target size."""
    products = []
    cats = list(CATEGORIES.keys())

    for i in range(count):
        cat = rng.choice(cats)
        cat_info = CATEGORIES[cat]
        nutri = _rand_nutrition(rng, cat)
        ns = rng.choice(cat_info["nutri_score_range"])
        nova = rng.choice(cat_info["nova_range"])
        brand = rng.choice(BRAND_POOL)

        product = {
            "product_id": f"SYN_{i+1:04d}",
            "product_name": f"{brand} {cat.replace('_', ' ').title()} Product #{i+1}",
            "brand": brand,
            "category": cat,
            "ingredients_text": f"Ingredients typical for {cat.replace('_', ' ')}: varied formulation #{i+1}",
            "nutrition_per_100g": nutri,
            "marketing_claims": [],
            "nutri_score": ns,
            "nova_group": nova,
            "is_adversarial": False,
            "adversarial_type": None,
        }
        products.append(product)

    return products


def _generate_adversarial_cases(rng: random.Random, base_products: list[dict]) -> list[dict]:
    """
    Generate adversarial products where marketing claims contradict actual data.

    Types:
      A - Misleading grain claims (multigrain/wholegrain but refined flour first)
      B - Hidden sugars (no added sugar claim but sugar synonyms present)
      C - Sodium masking (low salt claim but high sodium in nutrition)
      D - NOVA mismatch (natural/organic claim but NOVA 4)
    """
    adversarial = []
    adv_id = 1

    # Type A: Misleading grain claims
    type_a_bases = [p for p in base_products if "flour" in p["ingredients_text"].lower() or "maida" in p["ingredients_text"].lower()]
    for p in rng.sample(type_a_bases, min(8, len(type_a_bases))):
        ap = copy.deepcopy(p)
        ap["product_id"] = f"ADV_{adv_id:03d}"
        ap["product_name"] = f"MultiGrain {p['brand']} {p['category'].replace('_', ' ').title()}"
        ap["marketing_claims"] = ["Multigrain", "Wholesome grains"]
        if "refined" not in ap["ingredients_text"].lower():
            ap["ingredients_text"] = f"Refined wheat flour (maida), {ap['ingredients_text']}"
        ap["is_adversarial"] = True
        ap["adversarial_type"] = "A"
        adversarial.append(ap)
        adv_id += 1

    # Type B: Hidden sugars
    type_b_cats = ["health_drinks", "breakfast_cereal", "fruit_juice", "fruit_drinks", "biscuits"]
    type_b_bases = [p for p in base_products if p["category"] in type_b_cats]
    for p in rng.sample(type_b_bases, min(8, len(type_b_bases))):
        ap = copy.deepcopy(p)
        ap["product_id"] = f"ADV_{adv_id:03d}"
        ap["product_name"] = f"Sugar Free {p['product_name']}"
        ap["marketing_claims"] = ["No added sugar", "Sugar free"]
        hidden_sugar = rng.choice(SUGAR_SYNONYMS)
        if hidden_sugar not in ap["ingredients_text"].lower():
            ap["ingredients_text"] += f", {hidden_sugar}"
        ap["nutrition_per_100g"]["sugars_g"] = max(ap["nutrition_per_100g"]["sugars_g"], rng.uniform(8, 20))
        ap["is_adversarial"] = True
        ap["adversarial_type"] = "B"
        adversarial.append(ap)
        adv_id += 1

    # Type C: Sodium masking
    type_c_bases = [p for p in base_products if p["nutrition_per_100g"]["sodium_mg"] > 500]
    for p in rng.sample(type_c_bases, min(8, len(type_c_bases))):
        ap = copy.deepcopy(p)
        ap["product_id"] = f"ADV_{adv_id:03d}"
        ap["product_name"] = f"Low Salt {p['product_name']}"
        ap["marketing_claims"] = ["Low salt", "Reduced sodium"]
        ap["nutrition_per_100g"]["sodium_mg"] = max(ap["nutrition_per_100g"]["sodium_mg"], rng.uniform(700, 1400))
        ap["is_adversarial"] = True
        ap["adversarial_type"] = "C"
        adversarial.append(ap)
        adv_id += 1

    # Type D: NOVA mismatch (natural/organic claim but NOVA 4)
    type_d_bases = [p for p in base_products if p["nova_group"] == 4]
    for p in rng.sample(type_d_bases, min(8, len(type_d_bases))):
        ap = copy.deepcopy(p)
        ap["product_id"] = f"ADV_{adv_id:03d}"
        ap["product_name"] = f"All Natural {p['product_name']}"
        ap["marketing_claims"] = ["100% natural", "Organic ingredients"]
        ap["nova_group"] = 4
        ap["is_adversarial"] = True
        ap["adversarial_type"] = "D"
        adversarial.append(ap)
        adv_id += 1

    return adversarial


# ─── FSSAI violation code mappings ────────────────────────────────────────────

VIOLATION_RULES = {
    "high_sugar": {
        "code": "FSSAI-2020-HIGH-SUGAR",
        "check": lambda nut, thresh: nut.get("sugars_g", 0) > thresh.get("sugars_g_max", 25),
        "ingredient_keywords": ["sugar", "glucose", "fructose", "maltodextrin", "dextrose", "corn syrup", "invert syrup", "jaggery", "honey"],
    },
    "high_sodium": {
        "code": "FSSAI-2020-HIGH-SODIUM",
        "check": lambda nut, thresh: nut.get("sodium_mg", 0) > thresh.get("sodium_mg_max", 2000),
        "ingredient_keywords": ["salt", "sodium", "msg", "621", "flavour enhancer"],
    },
    "high_fat": {
        "code": "ICMR-NIN-HIGH-FAT",
        "check": lambda nut, thresh: nut.get("fat_g", 0) > thresh.get("fat_g_max", 35),
        "ingredient_keywords": ["palm oil", "palmolein", "vegetable oil", "hydrogenated", "margarine", "shortening", "ghee", "butter", "lard"],
    },
    "ultra_processed": {
        "code": "NOVA-4-ULTRA-PROCESSED",
        "check": lambda nut, thresh: False,  # checked via nova_group separately
        "ingredient_keywords": [],
    },
    "adversarial_misleading_grain": {
        "code": "FSSAI-2020-MISLEADING-LABEL-GRAIN",
        "check": lambda nut, thresh: False,
        "ingredient_keywords": ["refined wheat flour", "maida"],
    },
    "adversarial_hidden_sugar": {
        "code": "FSSAI-2020-MISLEADING-LABEL-SUGAR",
        "check": lambda nut, thresh: False,
        "ingredient_keywords": SUGAR_SYNONYMS,
    },
    "adversarial_sodium_masking": {
        "code": "FSSAI-2020-MISLEADING-LABEL-SODIUM",
        "check": lambda nut, thresh: False,
        "ingredient_keywords": ["salt", "sodium"],
    },
    "adversarial_nova_mismatch": {
        "code": "FSSAI-2020-MISLEADING-LABEL-NATURAL",
        "check": lambda nut, thresh: False,
        "ingredient_keywords": [],
    },
}


def _compute_ground_truth(product: dict, profile: dict) -> dict:
    """Compute deterministic ground truth for a (product, profile) pair."""
    nut = product["nutrition_per_100g"]
    thresh = profile["thresholds"]
    ingredients_lower = product["ingredients_text"].lower()

    risk_points = 0
    flagged_ingredients = []
    violation_codes = []

    # Nutritional threshold violations
    if nut.get("sugars_g", 0) > thresh.get("sugars_g_max", 25):
        risk_points += 1
        violation_codes.append("FSSAI-2020-HIGH-SUGAR")
        for kw in VIOLATION_RULES["high_sugar"]["ingredient_keywords"]:
            if kw.lower() in ingredients_lower:
                flagged_ingredients.append(kw)

    if nut.get("sodium_mg", 0) > thresh.get("sodium_mg_max", 2000):
        risk_points += 1
        violation_codes.append("FSSAI-2020-HIGH-SODIUM")
        for kw in VIOLATION_RULES["high_sodium"]["ingredient_keywords"]:
            if kw.lower() in ingredients_lower:
                flagged_ingredients.append(kw)

    if nut.get("fat_g", 0) > thresh.get("fat_g_max", 35):
        risk_points += 1
        violation_codes.append("ICMR-NIN-HIGH-FAT")
        for kw in VIOLATION_RULES["high_fat"]["ingredient_keywords"]:
            if kw.lower() in ingredients_lower:
                flagged_ingredients.append(kw)

    # NOVA 4 penalty
    if product.get("nova_group", 0) == 4:
        risk_points += 1
        violation_codes.append("NOVA-4-ULTRA-PROCESSED")

    # Nutri-Score D/E penalty
    if product.get("nutri_score", "").upper() in ("D", "E"):
        risk_points += 1

    # Adversarial contradiction penalties
    if product.get("is_adversarial"):
        adv_type = product.get("adversarial_type")
        if adv_type == "A":
            risk_points += 1
            violation_codes.append("FSSAI-2020-MISLEADING-LABEL-GRAIN")
            for kw in ["refined wheat flour", "maida"]:
                if kw in ingredients_lower:
                    flagged_ingredients.append(kw)
        elif adv_type == "B":
            risk_points += 1
            violation_codes.append("FSSAI-2020-MISLEADING-LABEL-SUGAR")
            for kw in SUGAR_SYNONYMS:
                if kw.lower() in ingredients_lower:
                    flagged_ingredients.append(kw)
        elif adv_type == "C":
            risk_points += 1
            violation_codes.append("FSSAI-2020-MISLEADING-LABEL-SODIUM")
        elif adv_type == "D":
            risk_points += 1
            violation_codes.append("FSSAI-2020-MISLEADING-LABEL-NATURAL")

    # Deduplicate
    flagged_ingredients = sorted(set(flagged_ingredients))
    violation_codes = sorted(set(violation_codes))

    return {
        "product_id": product["product_id"],
        "profile_id": profile["profile_id"],
        "expected_risk_level": min(risk_points, 4),
        "expected_flagged_ingredients": flagged_ingredients,
        "expected_violation_codes": violation_codes,
        "is_adversarial": product.get("is_adversarial", False),
        "adversarial_type": product.get("adversarial_type"),
        "category": product["category"],
    }


def main():
    rng = random.Random(2026)

    # Start with seed products
    all_products = []
    for p in SEED_PRODUCTS:
        product = copy.deepcopy(p)
        product.setdefault("is_adversarial", False)
        product.setdefault("adversarial_type", None)
        all_products.append(product)

    # Generate synthetic products to reach ~460 base products
    synthetic_count = 460 - len(SEED_PRODUCTS)
    synthetic = _generate_synthetic_products(rng, max(0, synthetic_count))
    all_products.extend(synthetic)

    # Generate adversarial cases
    adversarial = _generate_adversarial_cases(rng, all_products)
    all_products.extend(adversarial)

    # Load profiles
    profiles_path = DATA_DIR / "user_profiles.json"
    with open(profiles_path) as f:
        profiles = json.load(f)

    # Build ground truth for every (product, profile) pair
    ground_truth = []
    for product in all_products:
        for profile in profiles:
            gt = _compute_ground_truth(product, profile)
            ground_truth.append(gt)

    # Write outputs
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    products_path = DATA_DIR / "products.json"
    with open(products_path, "w") as f:
        json.dump(all_products, f, indent=2, ensure_ascii=False)

    gt_path = DATA_DIR / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    adversarial_path = DATA_DIR / "adversarial_cases.json"
    with open(adversarial_path, "w") as f:
        json.dump(adversarial, f, indent=2, ensure_ascii=False)

    # Stats
    print(f"Products:          {len(all_products)}")
    print(f"  Seed:            {len(SEED_PRODUCTS)}")
    print(f"  Synthetic:       {len(synthetic)}")
    print(f"  Adversarial:     {len(adversarial)}")
    print(f"Ground truth:      {len(ground_truth)} entries ({len(all_products)} products x {len(profiles)} profiles)")
    print(f"Adversarial types: {dict((t, sum(1 for a in adversarial if a['adversarial_type'] == t)) for t in ['A', 'B', 'C', 'D'])}")
    print(f"\nFiles written to: {DATA_DIR}")


if __name__ == "__main__":
    main()
