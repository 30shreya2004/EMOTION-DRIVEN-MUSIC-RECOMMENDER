import cv2
import numpy as np
from datetime import datetime

# Indian songs database organized by emotion
indian_songs = {
    "happy": [
        {"title": "Senorita", "artist": "Zara Khan", "movie": "Zindagi Na Milegi Dobara"},
        {"title": "Ye Jawaani Hai Deewani", "artist": "Ranbir Kapoor", "movie": "Ye Jawaani Hai Deewani"},
        {"title": "Chaleya", "artist": "Arijit Singh, Shilpa Rao", "movie": "Jawan"},
        {"title": "Baarish Ban Jaana", "artist": "B Praak, Stebin Ben", "movie": "Sab Kushal Mangal"},
        {"title": "Tum Se Hi", "artist": "Mohit Chauhan", "movie": "Jab We Met"},
        {"title": "Dil Dhadakne Do", "artist": "Priyanka Chopra", "movie": "Dil Dhadakne Do"},
        {"title": "Galliyan", "artist": "Arijit Singh", "movie": "Ek Villain"},
        {"title": "Ajeeb Daastaan", "artist": "A.R. Rahman", "movie": "Bundh"},
    ],
    "sad": [
        {"title": "Ae Dil Hai Mushkil", "artist": "Arijit Singh", "movie": "Ae Dil Hai Mushkil"},
        {"title": "Kabira", "artist": "Arijit Singh", "movie": "Yeh Jawaani Hai Deewani"},
        {"title": "Shayad", "artist": "Arijit Singh", "movie": "Love Aaj Kal"},
        {"title": "Tum Itna Jo Muskurate Ho", "artist": "Asha Parekh", "movie": "Chitchor"},
        {"title": "Ek Haseena Thi", "artist": "K.K.", "movie": "Ek Haseena Thi"},
        {"title": "Lamha Lamha", "artist": "Javed Ali", "movie": "Lamha"},
        {"title": "Tujhe Yaad Na Meri Aaye", "artist": "Sonu Nigam", "movie": "Chandni Bar"},
        {"title": "Khwab Dekhe", "artist": "Atif Aslam", "movie": "Aashiqui 2"},
    ],
    "angry": [
        {"title": "Bekhayali", "artist": "Arijit Singh", "movie": "Kabali"},
        {"title": "Tune Maari Entriyaan", "artist": "A.R. Rahman", "movie": "Gunday"},
        {"title": "Safar", "artist": "Arijit Singh, Kavita Seth", "movie": "Jism 2"},
        {"title": "Bhula Dena", "artist": "Arijit Singh", "movie": "Aashiqui 2"},
        {"title": "Gulon Me Rang", "artist": "A.R. Rahman", "movie": "Rang De Basanti"},
        {"title": "Badshah", "artist": "Badshah", "movie": "New Song"},
        {"title": "Delhi Wali Girlfriend", "artist": "Pritam", "movie": "Delhi Belly"},
        {"title": "Tattoo Remover", "artist": "Pritam", "movie": "Ae Dil Hai Mushkil"},
    ],
    "neutral": [
        {"title": "Iktara", "artist": "Anirudh Ravichander", "movie": "Wake Up Sid"},
        {"title": "Baarish", "artist": "Half Girlfriend", "movie": "Half Girlfriend"},
        {"title": "Ek Villain", "artist": "Siddhant-Garima", "movie": "Ek Villain"},
        {"title": "Tumhi Ho", "artist": "A.R. Rahman", "movie": "Rang De Basanti"},
        {"title": "Jiya Lage Na", "artist": "A.R. Rahman", "movie": "Jab Tak Hai Jaan"},
        {"title": "Saansein", "artist": "Chandrasekhar", "movie": "Saansein"},
        {"title": "Maula Mere", "artist": "Vishal-Shekhar", "movie": "Anjaana Anjaani"},
        {"title": "Choo Lo Na", "artist": "Papon", "movie": "Jazbaa"},
    ],
    "fear": [
        {"title": "Bhool Ja", "artist": "Arijit Singh", "movie": "Kabali"},
        {"title": "Dil Dooba", "artist": "K.K.", "movie": "Khushi"},
        {"title": "Teri Aashiqui Mein", "artist": "Vivarte", "movie": "Various"},
        {"title": "Main Tera", "artist": "Arijit Singh", "movie": "Kabali"},
        {"title": "Jab Tak", "artist": "A.R. Rahman", "movie": "Jab Tak Hai Jaan"},
        {"title": "Tere Naam", "artist": "Udit Narayan", "movie": "Tere Naam"},
        {"title": "Raees", "artist": "Pritam", "movie": "Raees"},
        {"title": "Aashiqui", "artist": "Tulsi Kumar", "movie": "Aashiqui 2"},
    ],
    "disgust": [
        {"title": "Duma Dum", "artist": "Yo Yo Honey Singh", "movie": "Dum Laga Ke Haisha"},
        {"title": "Mummy Mummy", "artist": "Malaika Arora", "movie": "Housefull 3"},
        {"title": "Mundian To Bach Ke", "artist": "Panjabi MC", "movie": "Bend It Like Beckham"},
        {"title": "Balam Pichkari", "artist": "Vishal Dadlani", "movie": "Yeh Jawaani Hai Deewani"},
        {"title": "Party All Night", "artist": "Badshah", "movie": "New Movies"},
        {"title": "DJ Wala Babu", "artist": "Aashiqui 2", "movie": "Aashiqui 2"},
        {"title": "Govinda", "artist": "Govinda", "movie": "Various"},
        {"title": "Crazy Kiya Re", "artist": "Pritam", "movie": "Dhoom 2"},
    ],
    "surprise": [
        {"title": "Mere Sapno Ka Raaja", "artist": "Jaya Seal", "movie": "Jab We Met"},
        {"title": "Aira Gaira", "artist": "A.R. Rahman", "movie": "Rang De Basanti"},
        {"title": "Prem Ki Baarish", "artist": "Asha Parekh", "movie": "Humshakal"},
        {"title": "Kab Milega", "artist": "Arijit Singh", "movie": "Socha Na Tha"},
        {"title": "Chalti Hai Kya Nau Se Barah", "artist": "Lucky Ali", "movie": "Khuda Gawah"},
        {"title": "Raaja Raaja", "artist": "Pritam", "movie": "Raees"},
        {"title": "Khab Dekhe", "artist": "Pritam", "movie": "Laal Rang"},
        {"title": "Arre Bhai", "artist": "Kumar Sanu", "movie": "Various"},
    ]
}

# Load cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise RuntimeError("Could not access webcam.")

print("Emotion Detection & Indian Song Recommender")
print("=" * 50)
print("Press 'q' to quit")
print("Press 's' to get song recommendations")
print()

current_emotion = "neutral"
emotion_counter = {"happy": 0, "sad": 0, "neutral": 0}

def detect_emotion_by_features(face_roi, eyes):
    """Detect emotion based on face features"""
    h, w = face_roi.shape[:2]
    
    # Check if smiling (mouth region brightness)
    mouth_region = face_roi[int(h*0.6):h, :]
    mouth_brightness = np.mean(mouth_region)
    
    # Check eye openness
    if len(eyes) >= 2:
        eye_area = sum(eyes[i][2] * eyes[i][3] for i in range(min(2, len(eyes))))
        eye_area_ratio = eye_area / (w * h)
    else:
        eye_area_ratio = 0
    
    # Emotion detection logic
    if mouth_brightness > 100 and eye_area_ratio > 0.02:
        return "happy"
    elif eye_area_ratio < 0.01:
        return "fear"
    elif mouth_brightness < 80:
        return "sad"
    else:
        return "neutral"

def get_emotion_color(emotion):
    """Return color for emotion display"""
    colors = {
        "happy": (0, 255, 0),      # Green
        "sad": (255, 0, 0),        # Blue
        "angry": (0, 0, 255),      # Red
        "neutral": (200, 200, 0),  # Cyan
        "fear": (128, 0, 128),     # Purple
        "disgust": (0, 165, 255),  # Orange
        "surprise": (0, 255, 255)  # Yellow
    }
    return colors.get(emotion, (200, 200, 200))

def recommend_songs(emotion):
    """Recommend songs based on detected emotion"""
    songs = indian_songs.get(emotion, indian_songs["neutral"])
    print("\n" + "=" * 70)
    print(f"🎵 SONG RECOMMENDATIONS FOR {emotion.upper()} 🎵")
    print("=" * 70)
    for i, song in enumerate(songs, 1):
        print(f"{i}. {song['title']}")
        print(f"   Artist: {song['artist']}")
        print(f"   Movie: {song['movie']}")
        print()

try:
    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to capture frame.")
            break

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            for (x, y, fw, fh) in faces:
                # Draw face rectangle
                cv2.rectangle(img, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                
                # Extract face ROI
                roi = img[y:y+fh, x:x+fw]
                roi_gray = gray[y:y+fh, x:x+fw]
                
                # Detect eyes in face
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
                
                # Detect emotion based on features
                emotion = detect_emotion_by_features(roi, eyes)
                current_emotion = emotion
                color = get_emotion_color(emotion)
                
                # Display emotion
                cv2.putText(img, f"Emotion: {emotion.upper()}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display current emotion and instructions
        cv2.putText(img, f"Current: {current_emotion.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, get_emotion_color(current_emotion), 2)
        cv2.putText(img, "Press 's' for song recommendations", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, "Press 'q' to quit", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, timestamp, (w - 280, h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        
        cv2.imshow("Emotion Detection & Indian Song Recommender", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            recommend_songs(current_emotion)

except Exception as e:
    print(f"Error: {e}")

finally:
    cam.release()
    cv2.destroyAllWindows()
    print("\nProgram closed")