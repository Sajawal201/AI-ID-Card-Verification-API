from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Flutter app connection settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Model Load karein
try:
    model = tf.keras.models.load_model('id_card_detector.h5')
    print("✅ AI Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def home():
    return {"message": "ID Card Verification API is live!"}

@app.post("/verify")
async def verify_id(file: UploadFile = File(...)):
    try:
        data = await file.read()
        image = prepare_image(data)
        
        # AI Prediction
        prediction_raw = model.predict(image)
        prediction_value = float(prediction_raw[0][0])
        
        print(f"--- New Request ---")
        print(f"File Name: {file.filename}")
        print(f"Model Raw Value: {prediction_value}")

        # LOGIC with Threshold
        # id_card = 0, not_id_card = 1
        if prediction_value < 0.5:
            confidence = (1 - prediction_value) * 100
            
            # Agar confidence 80% se kam hai, toh uncertain samjho
            if confidence < 80:
                result = {
                    "status": "failed",
                    "verified": False,
                    "label": "UNCERTAIN",
                    "confidence": f"{confidence:.2f}%",
                    "message": "Image is not clear or recognized. Please upload a high-quality original CNIC image."
                }
            else:
                result = {
                    "status": "success",
                    "verified": True,
                    "label": "ID_CARD",
                    "confidence": f"{confidence:.2f}%",
                    "message": "Valid National ID Card detected."
                }
        else:
            confidence = prediction_value * 100
            result = {
                "status": "failed",
                "verified": False,
                "label": "NOT_ID_CARD",
                "confidence": f"{confidence:.2f}%",
                "message": "Identity could not be verified. Please upload a valid ID card."
            }
        
        print(f"Final Decision: {result['label']} with {result['confidence']} confidence")
        return result
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
