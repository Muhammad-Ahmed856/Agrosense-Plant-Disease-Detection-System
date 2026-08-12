"""
Disease information database for all 38 PlantVillage classes.
Each entry has: description, symptoms, remedy, prevention
"""

DISEASE_INFO = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "severity": "medium",
        "description": "A fungal disease caused by Venturia inaequalis. Common in cool, wet climates.",
        "symptoms": "Dark, scabby lesions on leaves and fruit. Yellowing around spots.",
        "remedy": "Apply fungicides like Captan or Mancozeb. Remove infected leaves. Ensure good air circulation.",
        "prevention": "Use resistant apple varieties. Apply preventive sprays in spring."
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "severity": "high",
        "description": "Caused by fungus Botryosphaeria obtusa. Affects fruit, leaves, and bark.",
        "symptoms": "Brown lesions on leaves with purple borders. Rotting fruit with black rings.",
        "remedy": "Prune infected branches. Apply copper-based fungicides. Remove mummified fruit.",
        "prevention": "Maintain tree health. Avoid wounding bark during pruning."
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "severity": "medium",
        "description": "Fungal disease requiring both cedar/juniper and apple trees to complete its life cycle.",
        "symptoms": "Bright orange-yellow spots on upper leaf surfaces. Tube-like growths under leaves.",
        "remedy": "Apply myclobutanil or propiconazole fungicides. Remove nearby cedar trees if possible.",
        "prevention": "Plant resistant apple varieties. Remove juniper galls in winter."
    },
    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "severity": "none",
        "description": "The plant appears healthy with no visible disease symptoms.",
        "symptoms": "No symptoms detected.",
        "remedy": "Continue regular care: proper watering, fertilization, and monitoring.",
        "prevention": "Maintain good agricultural practices to keep plant healthy."
    },
    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "severity": "none",
        "description": "The blueberry plant appears healthy.",
        "symptoms": "No symptoms detected.",
        "remedy": "Maintain regular watering and acidic soil conditions.",
        "prevention": "Ensure proper drainage and nutrient management."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "severity": "medium",
        "description": "Caused by Podosphaera clandestina. Thrives in warm days and cool nights.",
        "symptoms": "White powdery coating on leaves, shoots, and fruit. Distorted young leaves.",
        "remedy": "Apply sulfur or potassium bicarbonate sprays. Improve air circulation.",
        "prevention": "Avoid overhead irrigation. Plant in sunny, well-ventilated areas."
    },
    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "severity": "none",
        "description": "The cherry plant appears healthy.",
        "symptoms": "No symptoms detected.",
        "remedy": "Continue good care practices.",
        "prevention": "Regular monitoring and balanced fertilization."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn (Maize)",
        "disease": "Gray Leaf Spot",
        "severity": "high",
        "description": "Caused by Cercospora zeae-maydis. Favored by warm, humid conditions.",
        "symptoms": "Rectangular gray to tan lesions on leaves running parallel to veins.",
        "remedy": "Apply strobilurin or triazole fungicides. Rotate crops. Use resistant hybrids.",
        "prevention": "Crop rotation with non-host crops. Reduce leaf wetness duration."
    },
    "Corn_(maize)___Common_rust_": {
        "plant": "Corn (Maize)",
        "disease": "Common Rust",
        "severity": "medium",
        "description": "Caused by Puccinia sorghi. Spreads rapidly in cool, moist conditions.",
        "symptoms": "Small, oval, brick-red pustules on both leaf surfaces.",
        "remedy": "Apply fungicides early. Plant resistant corn hybrids.",
        "prevention": "Use rust-resistant varieties. Early planting to avoid peak rust season."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn (Maize)",
        "disease": "Northern Leaf Blight",
        "severity": "high",
        "description": "Caused by Exserohilum turcicum. Major disease in humid regions.",
        "symptoms": "Long, elliptical, grayish-green to tan lesions (1-6 inches) on leaves.",
        "remedy": "Apply fungicides at tasseling. Use resistant hybrids. Rotate crops.",
        "prevention": "Crop rotation. Residue management. Resistant varieties."
    },
    "Corn_(maize)___healthy": {
        "plant": "Corn (Maize)",
        "disease": "Healthy",
        "severity": "none",
        "description": "The corn plant appears healthy.",
        "symptoms": "No symptoms detected.",
        "remedy": "Maintain proper fertilization and irrigation schedules.",
        "prevention": "Regular scouting and good agronomic practices."
    },
    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "severity": "high",
        "description": "Caused by Guignardia bidwellii. Can destroy entire grape crop.",
        "symptoms": "Tan or brown circular spots on leaves. Shriveled, black, mummified berries.",
        "remedy": "Apply mancozeb or myclobutanil fungicides from bud break. Remove mummified berries.",
        "prevention": "Good canopy management. Remove infected material promptly."
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca (Black Measles)",
        "severity": "high",
        "description": "Complex disease caused by multiple fungi. Affects wood and leaves.",
        "symptoms": "Interveinal leaf discoloration (tiger stripe pattern). Shrunken, dark berries.",
        "remedy": "No curative treatment. Remove and destroy infected vines.",
        "prevention": "Protect pruning wounds. Avoid large pruning cuts."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf Blight",
        "severity": "medium",
        "description": "Caused by Isariopsis clavispora. Occurs late in the season.",
        "symptoms": "Dark brown, irregular spots on older leaves. Premature defoliation.",
        "remedy": "Apply copper-based fungicides. Ensure good drainage and air circulation.",
        "prevention": "Balanced nutrition. Avoid excessive nitrogen."
    },
    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "severity": "none",
        "description": "The grape plant appears healthy.",
        "symptoms": "No symptoms detected.",
        "remedy": "Continue proper pruning and canopy management.",
        "prevention": "Regular monitoring and balanced irrigation."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Huanglongbing (Citrus Greening)",
        "severity": "critical",
        "description": "Bacterial disease spread by Asian citrus psyllid. No cure available. Kills trees.",
        "symptoms": "Yellow shoots (blotchy mottle). Lopsided, bitter fruit. Stunted growth.",
        "remedy": "No cure. Remove and destroy infected trees immediately to prevent spread.",
        "prevention": "Control Asian citrus psyllid with insecticides. Use certified disease-free plants."
    },
    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "severity": "medium",
        "description": "Caused by Xanthomonas arboricola. Severe in warm, rainy weather.",
        "symptoms": "Small, water-soaked spots on leaves turning brown with yellow halos. Cracked fruit.",
        "remedy": "Apply copper-based bactericides. Avoid overhead irrigation.",
        "prevention": "Plant resistant varieties. Prune for good air circulation."
    },
    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "severity": "none",
        "description": "The peach plant appears healthy.",
        "symptoms": "No symptoms detected.",
        "remedy": "Maintain regular fertilization and proper pruning.",
        "prevention": "Annual dormant sprays and regular monitoring."
    },
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "disease": "Bacterial Spot",
        "severity": "medium",
        "description": "Caused by Xanthomonas campestris. Spreads in warm, wet conditions.",
        "symptoms": "Small, water-soaked lesions on leaves and fruit that turn brown and scabby.",
        "remedy": "Apply copper-based sprays. Remove infected plant debris. Avoid wetting foliage.",
        "prevention": "Use certified disease-free seeds. Crop rotation. Resistant varieties."
    },
    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper",
        "disease": "Healthy",
        "severity": "none",
        "description": "The bell pepper plant appears healthy.",
        "symptoms": "No symptoms detected.",
        "remedy": "Continue good care with balanced fertilization.",
        "prevention": "Regular watering and pest monitoring."
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "severity": "medium",
        "description": "Caused by Alternaria solani. Common in warm, humid weather.",
        "symptoms": "Dark brown lesions with concentric rings (target-board pattern) on older leaves.",
        "remedy": "Apply chlorothalonil or mancozeb fungicides. Remove infected lower leaves.",
        "prevention": "Crop rotation. Avoid overhead irrigation. Use healthy seed potatoes."
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "severity": "critical",
        "description": "Caused by Phytophthora infestans. The disease that caused the Irish Potato Famine.",
        "symptoms": "Water-soaked, dark brown lesions on leaves with white mold on undersides.",
        "remedy": "Apply metalaxyl or cymoxanil fungicides immediately. Destroy infected plants.",
        "prevention": "Use certified seed. Fungicide calendar spray program. Resistant varieties."
    },
    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "severity": "none",
        "description": "The potato plant appears healthy.",
        "symptoms": "No symptoms detected.",
        "remedy": "Maintain proper hilling and irrigation practices.",
        "prevention": "Use certified disease-free seed potatoes."
    },
    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "severity": "none",
        "description": "The raspberry plant appears healthy.",
        "symptoms": "No symptoms detected.",
        "remedy": "Proper pruning after harvest and balanced fertilization.",
        "prevention": "Good air circulation and weed control."
    },
    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "severity": "none",
        "description": "The soybean plant appears healthy.",
        "symptoms": "No symptoms detected.",
        "remedy": "Monitor regularly and maintain proper nitrogen fixation conditions.",
        "prevention": "Proper inoculation and balanced fertilization."
    },
    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "severity": "medium",
        "description": "Caused by Podosphaera xanthii. Very common in cucurbit crops.",
        "symptoms": "White powdery spots on leaves and stems. Yellowing and premature senescence.",
        "remedy": "Apply potassium bicarbonate, neem oil, or sulfur-based fungicides.",
        "prevention": "Avoid dense planting. Improve air circulation. Resistant varieties."
    },
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "severity": "medium",
        "description": "Caused by Diplocarpon earlianum. Common in warm, wet conditions.",
        "symptoms": "Small, dark purple spots on upper leaf surfaces. Leaves turn brown and scorch.",
        "remedy": "Apply captan fungicide. Remove old infected leaves after harvest.",
        "prevention": "Avoid overhead irrigation. Proper plant spacing."
    },
    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "severity": "none",
        "description": "The strawberry plant appears healthy.",
        "symptoms": "No symptoms detected.",
        "remedy": "Maintain mulching and proper irrigation.",
        "prevention": "Regular runner removal and renovation."
    },
    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "severity": "medium",
        "description": "Caused by Xanthomonas species. Spreads rapidly in warm, wet weather.",
        "symptoms": "Small, dark spots on leaves, stems, and fruit. Yellow halos around spots.",
        "remedy": "Apply copper hydroxide sprays. Remove infected plant parts. Avoid overhead watering.",
        "prevention": "Use disease-free seeds. Crop rotation. Resistant varieties."
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "severity": "medium",
        "description": "Caused by Alternaria solani. Affects leaves, stems, and fruit.",
        "symptoms": "Dark concentric rings (bull's eye) on older lower leaves. Yellowing around lesions.",
        "remedy": "Apply chlorothalonil or mancozeb. Remove infected leaves. Mulch to reduce splash.",
        "prevention": "Crop rotation. Stake plants for airflow. Avoid working when plants are wet."
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "severity": "critical",
        "description": "Caused by Phytophthora infestans. Can destroy entire crop within days.",
        "symptoms": "Large, water-soaked brown lesions. White mold on leaf undersides in humid conditions.",
        "remedy": "Apply metalaxyl + mancozeb immediately. Destroy all infected plants.",
        "prevention": "Avoid overhead irrigation. Plant in well-drained soil. Monitor weather forecasts."
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold",
        "severity": "medium",
        "description": "Caused by Passalora fulva. Common in greenhouses with high humidity.",
        "symptoms": "Pale yellow spots on upper leaf surface. Olive-green to gray mold on underside.",
        "remedy": "Reduce humidity. Apply chlorothalonil or copper-based fungicides. Improve ventilation.",
        "prevention": "Keep humidity below 85%. Space plants properly. Remove infected leaves."
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "severity": "medium",
        "description": "Caused by Septoria lycopersici. Very common, starts on lower leaves.",
        "symptoms": "Numerous small circular spots with dark borders and light centers on leaves.",
        "remedy": "Apply mancozeb or copper fungicides. Remove infected lower leaves. Avoid overhead watering.",
        "prevention": "Mulch around plants. Crop rotation. Stake plants."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Spider Mites (Two-Spotted)",
        "severity": "medium",
        "description": "Caused by Tetranychus urticae mites. Thrives in hot, dry conditions.",
        "symptoms": "Fine stippling on leaves. Yellowing. Fine webbing on leaf undersides.",
        "remedy": "Apply miticides or insecticidal soap. Spray undersides of leaves. Increase humidity.",
        "prevention": "Avoid drought stress. Avoid broad-spectrum insecticides that kill natural predators."
    },
    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "severity": "medium",
        "description": "Caused by Corynespora cassiicola. Affects all above-ground plant parts.",
        "symptoms": "Circular spots with concentric rings (target pattern). Brown lesions on fruit.",
        "remedy": "Apply azoxystrobin or chlorothalonil fungicides. Improve air circulation.",
        "prevention": "Crop rotation. Proper spacing. Stake plants."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Yellow Leaf Curl Virus",
        "severity": "critical",
        "description": "Transmitted by whiteflies (Bemisia tabaci). No cure once infected.",
        "symptoms": "Yellowing and upward curling of leaves. Stunted growth. Reduced fruiting.",
        "remedy": "No cure. Remove infected plants. Control whitefly populations with insecticides.",
        "prevention": "Use whitefly-resistant varieties. Yellow sticky traps. Reflective mulches."
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Mosaic Virus",
        "severity": "high",
        "description": "Caused by Tomato mosaic virus (ToMV). Spread by contact and tools.",
        "symptoms": "Mottled light and dark green mosaic pattern on leaves. Distorted, stunted growth.",
        "remedy": "No cure. Remove infected plants. Disinfect tools with bleach solution.",
        "prevention": "Wash hands before handling plants. Use virus-free seeds. Resistant varieties."
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "severity": "none",
        "description": "The tomato plant appears healthy.",
        "symptoms": "No symptoms detected.",
        "remedy": "Maintain regular watering, staking, and fertilization.",
        "prevention": "Monitor regularly for early signs of disease or pests."
    },
}

def get_disease_info(class_name):
    """Return disease info dict for a given class name. Returns a default if not found."""
    info = DISEASE_INFO.get(class_name)
    if info:
        return info
    # Fallback: parse from class name
    parts = class_name.split("___")
    plant = parts[0].replace("_", " ") if len(parts) > 0 else "Unknown"
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in class_name.lower()
    return {
        "plant": plant,
        "disease": disease,
        "severity": "none" if is_healthy else "medium",
        "description": f"Information for {disease} on {plant}.",
        "symptoms": "Please consult an agricultural expert.",
        "remedy": "Consult your local agricultural extension officer.",
        "prevention": "Regular monitoring and good agricultural practices."
    }
