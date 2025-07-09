# TaleGenie 🧞‍♂️📚

TaleGenie is a personalized storytelling engine for children that generates illustrated stories based on uploaded photos and user inputs like name, age, gender, and outfit. It uses AI to train a LoRA model for a child’s likeness and crafts customized, interactive adventures in a children's book illustration style.

---

## 🛠 Features

- Upload 5–10 photos of a child
- Input their name, gender, age, and outfit
- Zips & uploads data to Cloudinary
- Trains a LoRA model using Replicate API
- Generates a character illustration and a story scene
- Presents interactive choices in story format
- Frontend built with plain HTML + JS, backend with Flask

---

## 🚀 How It Works (End-to-End Flow)

1. **User fills form**: Inputs name, gender, age, outfit + uploads 5–10 photos.
2. **Backend receives**:
   - Thumbnails & saves photos
   - Zips them and uploads securely to Cloudinary
3. **LoRA model training**:
   - Using `ostris/flux-dev-lora-trainer` on Replicate
   - Generates model version with a unique token (e.g., `a photo of TOK`)
4. **Character Generation**:
   - Prompt: "`TOK`, a [AGE]-year-old [GENDER] child wearing [OUTFIT], cartoon children’s book style"
5. **Scene Generation**:
   - E.g. “On grass holding a red truck, another boy looks sad…”
   - Prompts differ slightly based on gender
6. **Frontend displays**:
   - Character image preview
   - Scene image with a moral question & options
   - Interactive story experience

---

## 🧪 Local Setup

```bash
git clone https://github.com/verysanskarii/talegenie.git
cd talegenie
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
