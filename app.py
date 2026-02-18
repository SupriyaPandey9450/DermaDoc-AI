
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os, uuid, json
from functools import wraps

app = Flask(__name__)
app.secret_key = "replace_with_secure_random_key"

# ---------------- Load Model ---------------- #
# model = load_model("skin_disease_model.h5")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "skin_disease_model.h5")
model = load_model(model_path)


# with open("class_labels.json", "r") as f:
    # class_labels = json.load(f)
labels_path = os.path.join(BASE_DIR, "class_labels.json")
with open(labels_path, "r") as f:
    class_labels = json.load(f)


# ---------------- Users Storage ---------------- #
USERS_FILE = "users.json"

def load_users():
    """Load users.json"""
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({"users": {}}, f)

    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(data):
    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ---------------- Login Required Decorator ---------------- #
def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapped

# ---------------- Image Preprocess ---------------- #
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128), color_mode="rgb")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------- Routes ---------------- #
@app.route("/")
def home():
    return redirect(url_for("index_page"))

@app.route("/index")
def index_page():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/chatbot")
@login_required
def chatbot():
    return render_template("chatbot.html")

@app.route("/result")
@login_required
def result():
    return render_template("result.html")

# ---------------- SIGNUP ---------------- #
@app.route("/signup", methods=["GET", "POST"])
def signup():
    error = None

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        phone = request.form.get("phone", "").strip()
        age = request.form.get("age", "").strip()
        gender = request.form.get("gender", "").strip()
        role = request.form.get("role", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not email or not password:
            error = "Email and password are required."

        elif password != confirm_password:
            error = "Passwords do not match."

        else:
            data = load_users()
            if email in data["users"]:
                error = "Account already exists."
            else:
                hashed = generate_password_hash(password)

                data["users"][email] = {
                    "name": name,
                    "phone": phone,
                    "age": age,
                    "gender": gender,
                    "role": role,
                    "password": hashed,
                    "profile_pic": None
                }

                save_users(data)
                session["user"] = email
                return redirect(url_for("chatbot"))

    return render_template("signup.html", error=error)

# ---------------- LOGIN ---------------- #
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        data = load_users()
        user = data["users"].get(email)

        if user and check_password_hash(user["password"], password):
            session["user"] = email
            return redirect(url_for("chatbot"))
        else:
            error = "Invalid email or password."

    return render_template("login.html", error=error)

# ---------------- LOGOUT ---------------- #
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ---------------- Chatbot API ---------------- #
@app.route("/chatbot-api", methods=["POST"])
@login_required
def chatbot_api():
    data = request.get_json()
    user_message = data.get("message", "")

    # Simple chatbot response (you can improve later)
    reply = f"I understand you said: {user_message}"
    return jsonify({"reply": reply})

# ---------------- Prediction API ---------------- #
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # os.makedirs("uploads", exist_ok=True)
    # filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    # filepath = os.path.join("uploads", filename)
    # file.save(filepath)
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    file.save(filepath)


    
        # img_array = preprocess_image(filepath)
        # prediction = model.predict(img_array)
        # predicted_index = np.argmax(prediction)
        # result = class_labels[predicted_index]

        # return jsonify({"prediction": result})
        # return render_template("result.html", prediction=result)
    try:
         img_array = preprocess_image(filepath)
         prediction = model.predict(img_array)
         predicted_index = np.argmax(prediction)
         result = class_labels[predicted_index]

    # Disease info mapping
         disease_info = {
        "acne": {
            "symptoms": "Pimples, blackheads, oily skin",
            "solution": "Use mild cleanser, avoid oily foods, consult dermatologist if severe."
        },
        "eczema": {
            "symptoms": "Itching, redness, irritation",
            "solution": "Use moisturizer, avoid allergens."
        }
    }

         info = disease_info.get(result, {"symptoms": "N/A", "solution": "N/A"})

         return render_template(
        "result.html",
        prediction=result,
        symptoms=info["symptoms"],
        solution=info["solution"]
    )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
      if os.path.exists(filepath):
        os.remove(filepath)
        # -------- Run -----------
# -------- Run App ----------- #
if __name__ == "__main__":
    app.run(debug=True)
