import logging
import os
import time
import threading
import zipfile
import uuid
import requests
import json

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import replicate
import cloudinary
import cloudinary.uploader
from cloudinary.utils import private_download_url
from PIL import Image
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# --- Logging setup ---
logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Env & clients ---
load_dotenv()
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)
os.makedirs('temp_photos', exist_ok=True)
os.makedirs('temp_zips', exist_ok=True)

REPL_TOKEN = os.getenv('REPLICATE_API_TOKEN')
rep_client = replicate.Client(api_token=REPL_TOKEN)

cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

# --- In-memory store ---
store = {}

# --- Helper function for consistent story elements ---
def get_story_elements(child_gender, child_age):
    """Generate consistent story elements based on child's gender and age"""
    
    if child_gender.lower() in ['girl', 'female']:
        return {
            'toy_type': 'pink teddy bear',
            'toy_color': 'pink',
            'friend_gender': 'girl',
            'friend_description': 'different girl with shoulder-length blonde hair, wearing a purple dress with white flowers, white mary jane shoes, clearly different face and smaller build than main character',
            'teacher_description': 'adult female teacher with professional brown hair in a bun, wearing navy blue blouse and black dress pants, tall adult woman with mature facial features, clearly an adult not a child',
            'environment': 'cozy indoor playroom with soft carpet, colorful wall decorations, toy shelves, and warm lighting from windows'
        }
    else:  # boy or other
        return {
            'toy_type': 'toy truck',
            'toy_color': 'red',
            'friend_gender': 'boy', 
            'friend_description': 'different boy with short curly brown hair, wearing a green striped t-shirt and khaki shorts, sneakers, clearly different face and smaller build than main character',
            'teacher_description': 'adult male teacher with short dark hair, wearing a light blue button-up shirt and dark navy pants, tall adult man with mature facial features, clearly an adult not a child',
            'environment': 'sunny outdoor park with green grass, colorful playground equipment, trees, and blue sky'
        }

# --- Comic Story Structure (Age-appropriate for 2-year-old) ---
COMIC_SCENES = {
    'scene1': {
        'description': "{child_name} is sitting in the {environment} holding a {toy_color} {toy_type}.\nAnother little {friend_gender} sits nearby looking sad with no toys to play with.",
        'question': "What should {child_name} do?",
        'options': {
            'A': "Share the {toy_type} with the other {friend_gender}",
            'B': "Keep playing alone with the {toy_type}"
        },
        'prompt': "MAIN CHARACTER: {child_name} wearing {child_outfit} sitting in center foreground holding {toy_color} {toy_type} with both hands, happy expression, SIDE CHARACTER: {friend_description} sitting to the right side in background looking sad with empty hands reaching toward toy, SETTING: {environment}, IMPORTANT: {child_name} is the main focus in center, friend is clearly separate person on the side, very different faces and hair, digital children's book illustration for toddlers, soft warm colors"
    },
    'scene2A': {
        'description': "{child_name} stands up with the {toy_color} {toy_type} and walks toward the sad {friend_gender}.\n{child_name} has a big smile and is holding out the {toy_type} to share.",
        'question': "How should {child_name} share the {toy_type}?",
        'options': {
            'A': "Let the other {friend_gender} play with the {toy_type} first",
            'B': "Play with the {toy_type} together"
        },
        'prompt': "MAIN CHARACTER: {child_name} wearing {child_outfit} standing in center walking toward right side, holding {toy_color} {toy_type} extended in offering gesture, big smile, SIDE CHARACTER: {friend_description} sitting on right side looking up hopefully at {child_name}, SETTING: {environment}, IMPORTANT: {child_name} is taller and in center moving toward smaller friend on right, clearly different people, digital children's book illustration for toddlers, kind sharing moment"
    },
    'scene2B': {
        'description': "{child_name} keeps playing with the {toy_color} {toy_type} but looks over at the sad {friend_gender}.\n{child_name} feels a little confused about what to do.",
        'question': "What should {child_name} do now?",
        'options': {
            'A': "Go share the {toy_color} {toy_type}",
            'B': "Ask a grown-up for help"
        },
        'prompt': "MAIN CHARACTER: {child_name} wearing {child_outfit} sitting in center foreground holding {toy_color} {toy_type} but turning head to look toward right side, thoughtful confused expression, SIDE CHARACTER: {friend_description} sitting on right side in background looking sad, SETTING: {environment}, IMPORTANT: {child_name} is main focus in center, friend is clearly separate smaller person on right side, digital children's book illustration for toddlers, emotional moment"
    },
    'scene3A1': {
        'description': "The other {friend_gender} is now playing happily with the {toy_color} {toy_type}.\n{child_name} sits nearby watching with a big smile, waiting patiently.",
        'question': "What happens next?",
        'options': {
            'A': "They take turns with the {toy_type}",
            'B': "More kids want to play too"
        },
        'prompt': "MAIN CHARACTER: {child_name} wearing {child_outfit} sitting on left side watching with big patient smile, SIDE CHARACTER: {friend_description} on right side playing happily with {toy_color} {toy_type}, both children content, SETTING: {environment}, IMPORTANT: {child_name} on left is main character, friend on right is clearly different person with {friend_description}, digital children's book illustration for toddlers, sharing and patience"
    },
    'scene3A2': {
        'description': "{child_name} and the other {friend_gender} are both playing with the {toy_color} {toy_type} together.\nThey are having fun and both smiling.",
        'question': "What happens next?",
        'options': {
            'A': "They become good friends",
            'B': "They find more toys to share"
        },
        'prompt': "MAIN CHARACTER: {child_name} wearing {child_outfit} on left side holding one part of {toy_color} {toy_type}, SIDE CHARACTER: {friend_description} on right side holding other part of toy, both smiling and playing cooperatively, SETTING: {environment}, IMPORTANT: {child_name} is main character on left, friend is clearly different smaller person on right, digital children's book illustration for toddlers, friendship and teamwork"
    },
    'scene3B1': {
        'description': "{child_name} walks over to share the {toy_color} {toy_type} with the other {friend_gender}.\n{child_name} looks happy to help and make a new friend.",
        'question': "How does {child_name} feel about sharing?",
        'options': {
            'A': "Happy to make a friend",
            'B': "Proud of being kind"
        },
        'prompt': "MAIN CHARACTER: {child_name} wearing {child_outfit} walking from left to right carrying {toy_color} {toy_type}, happy helpful expression, SIDE CHARACTER: {friend_description} sitting on right side looking hopeful, SETTING: {environment}, IMPORTANT: {child_name} is main character moving from left to right, friend is clearly different smaller person on right, digital children's book illustration for toddlers, learning to share moment"
    },
    'scene3B2': {
        'description': "A grown-up comes over to help {child_name} learn about sharing.\nThe grown-up shows {child_name} how good it feels to share toys.",
        'question': "What does {child_name} learn?",
        'options': {
            'A': "Sharing makes everyone happy",
            'B': "Grown-ups help us learn"
        },
        'prompt': "MAIN CHARACTER: {child_name} wearing {child_outfit} sitting in center holding {toy_color} {toy_type}, looking up at adult, ADULT: {teacher_description} kneeling down next to {child_name} on left side, pointing gently toward friend, SIDE CHARACTER: {friend_description} sitting on right side in background, SETTING: {environment}, IMPORTANT: {child_name} is main character in center, adult is clearly tall grown-up with mature face, friend is clearly different smaller child on right, digital children's book illustration for toddlers, gentle teaching moment"
    },
    'scene4': {
        'description': "{child_name} and the other {friend_gender} are both playing happily together.\nThey are sharing the {toy_color} {toy_type} and smiling at each other.",
        'question': None,
        'options': None,
        'prompt': "MAIN CHARACTER: {child_name} wearing {child_outfit} on left side smiling while sharing {toy_color} {toy_type}, SIDE CHARACTER: {friend_description} on right side also smiling while playing with shared toy, both children happy and content, SETTING: {environment}, IMPORTANT: {child_name} is main character on left, friend is clearly different person on right with {friend_description}, digital children's book illustration for toddlers, friendship and happiness finale",
        'lesson': "Sharing the {toy_color} {toy_type} makes everyone happy and helps us make friends!"
    }
}

# --- Routes ---
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/upload-photos', methods=['POST'])
def upload_photos():
    run_id = str(uuid.uuid4())[:8]
    try:
        # 1) Metadata
        meta = {k: request.form[k] for k in ('name','gender','age','outfit')}
        store[run_id] = {'meta': meta}

        # ─── DEBUG ECHO ───────────────────────────────────────────────────
        # Log and return exactly what file‐fields were received
        incoming_keys = list(request.files.keys())
        logger.info(f"[{run_id}] Incoming file keys: {incoming_keys}")
        return jsonify({
            "success": True,
            "debug": {
                "file_keys": incoming_keys,
                "num_files": len(incoming_keys),
                "first_filenames": [f.filename for f in request.files.values()][:3]
            }
        })
        # ──────────────────────────────────────────────────────────────────

        # collect any file field whose name starts with "photos"
        photo_keys = [k for k in request.files.keys() if k.startswith('photo')]
        files = []
        for key in photo_keys:
            files.extend(request.files.getlist(key))

        # process & thumbnail each file
        paths = []
        for i, f in enumerate(files, start=1):
            if not f.filename:
                continue
            ext = secure_filename(f.filename).rsplit('.',1)[-1].lower() if '.' in f.filename else 'jpg'
            outp = f"temp_photos/{run_id}_{i}.{ext}"
            img = Image.open(f.stream).convert('RGB')
            img.thumbnail((1024,1024), Image.Resampling.LANCZOS)
            img.save(outp,'JPEG',quality=85)
            paths.append(outp)
            logger.info(f"[{run_id}] Processed photo {i}")

        # Zip up the thumbnails
        zip_path = f"temp_zips/{run_id}.zip"
        with zipfile.ZipFile(zip_path,'w') as zf:
            for idx, p in enumerate(paths, start=1):
                if os.path.exists(p):
                    zf.write(p, f"photo_{idx:02d}.jpg")
        logger.info(f"[{run_id}] Created zip at {zip_path}")

        # 4) Token—and store
        token_str = "TOK"
        store[run_id].update(token=token_str, zip_path=zip_path)

        return jsonify(
            success=True,
            character_id=run_id,
            token_string=token_str,
            zip_path=zip_path
        )

    except Exception as e:
        logger.exception(f"[{run_id}] upload_photos error")
        return jsonify(success=False, error=str(e)), 500

@app.route('/start-secure-pipeline', methods=['POST'])
def start_pipeline():
    data = request.get_json(force=True)
    run_id = data['character_id']
    logger.info(f"[{run_id}] Launching pipeline thread")
    threading.Thread(target=run_pipeline, args=(run_id,), daemon=True).start()
    return jsonify(success=True, run_id=run_id)

def run_pipeline(run_id):
    rec       = store.get(run_id, {})
    meta      = rec.get('meta', {})
    token     = rec.get('token')
    zip_path  = rec.get('zip_path')

    # Default to failed so we always set status
    result = {'run_id': run_id, 'success': False, 'status': 'failed', 'error': None}

    try:
        # 1) Upload to Cloudinary
        logger.info(f"[{run_id}] Uploading zip: {zip_path}")
        up = cloudinary.uploader.upload(
            zip_path,
            resource_type='raw',
            type='authenticated',
            public_id=f"training_zips/{run_id}",
            overwrite=True
        )
        signed_url = private_download_url(
            up['public_id'], 'zip',
            resource_type='raw',
            type='authenticated',
            expires_at=int(time.time())+3600
        )
        logger.info(f"[{run_id}] Signed URL: {signed_url}")

        # 2) Ensure model exists
        dest    = f"verysanskarii/{run_id}-lora"
        headers = {
            'Authorization': f"Token {REPL_TOKEN}",
            'Content-Type': 'application/json'
        }
        
        # Check if model exists, create if not
        model_check = requests.get(f"https://api.replicate.com/v1/models/{dest}", headers=headers, timeout=30)
        if model_check.status_code == 404:
            logger.info(f"[{run_id}] Creating model: {dest}")
            payload = {
                'owner': 'verysanskarii',
                'name':  f"{run_id}-lora",
                'visibility': 'private',
                'description': f"LoRA for {meta.get('name')}",
                'hardware': 'gpu-t4'
            }
            create_response = requests.post("https://api.replicate.com/v1/models",
                          headers=headers, json=payload, timeout=30)
            create_response.raise_for_status()
            logger.info(f"[{run_id}] Model created successfully")
        else:
            logger.info(f"[{run_id}] Model already exists")

        # 3) Start training - using direct API call with correct endpoint
        training_payload = {
            'version': "ostris/flux-dev-lora-trainer:e440909d3512c31646ee2e0c7d6f6f4923224863a6a10c494606e79fb5844497",
            'input': {
                'input_images': signed_url,
                'token_string': token,
                'caption_prefix': f"a photo of {token}",
                'max_train_steps': 1000,
                'learning_rate': 1e-4,
                'batch_size': 1,
                'resolution': 512
            },
            'destination': dest
        }

        # Use the correct training endpoint - this was the issue!
        training_response = requests.post(
            f"https://api.replicate.com/v1/models/ostris/flux-dev-lora-trainer/versions/e440909d3512c31646ee2e0c7d6f6f4923224863a6a10c494606e79fb5844497/trainings",
            headers=headers,
            json=training_payload,
            timeout=30
        )

        if training_response.status_code != 201:
            raise RuntimeError(f"Training creation failed: {training_response.status_code} - {training_response.text}")

        training_data = training_response.json()
        training_id = training_data['id']
        logger.info(f"[{run_id}] Training started: {training_id}")

        # 4) Poll until done with direct API calls
        status = None
        poll_count = 0
        max_polls = 120  # 30 minutes max (15 second intervals)

        while status not in ('succeeded', 'failed', 'canceled') and poll_count < max_polls:
            time.sleep(15)
            poll_count += 1
            
            try:
                status_response = requests.get(
                    f"https://api.replicate.com/v1/trainings/{training_id}",
                    headers=headers,
                    timeout=30
                )
                
                if status_response.status_code == 200:
                    training_data = status_response.json()
                    status = training_data['status']
                    logger.info(f"[{run_id}] Training status ({poll_count}/120): {status}")
                else:
                    logger.warning(f"[{run_id}] Status check failed: {status_response.status_code}")
                    if poll_count % 4 == 0:  # Every minute, log the error
                        logger.warning(f"[{run_id}] Status response: {status_response.text[:200]}")
        
            except requests.exceptions.Timeout:
                logger.warning(f"[{run_id}] Timeout checking training status")
            except requests.exceptions.RequestException as e:
                logger.warning(f"[{run_id}] Network error checking status: {e}")
            except json.JSONDecodeError as e:
                logger.warning(f"[{run_id}] JSON decode error: {e}")
                logger.warning(f"[{run_id}] Raw response: {status_response.text[:200]}")

        if poll_count >= max_polls:
            raise RuntimeError(f"Training timed out after {max_polls} polls")

        result['status'] = status

        if status != 'succeeded':
            raise RuntimeError(f"Training failed with status: {status}")

        # Get final training data
        final_response = requests.get(
            f"https://api.replicate.com/v1/trainings/{training_id}",
            headers=headers,
            timeout=30
        )

        if final_response.status_code != 200:
            raise RuntimeError(f"Could not get final training data: {final_response.status_code}")

        final_training_data = final_response.json()
        output = final_training_data.get('output', {})
        
        logger.info(f"[{run_id}] Training output keys: {list(output.keys())}")
        logger.info(f"[{run_id}] Full training output: {output}")

        # 6) Get model reference - FIXED VERSION
        model_ref = None
        
        # Method 1: Check for 'version' key (this is what your working training had)
        if 'version' in output:
            model_ref = output['version']
            logger.info(f"[{run_id}] Found model version: {model_ref}")
        
        # Method 2: Check for 'model' key
        elif 'model' in output:
            model_ref = output['model']
            logger.info(f"[{run_id}] Found model: {model_ref}")
            
        # Method 3: Use destination as fallback
        else:
            model_ref = dest
            logger.info(f"[{run_id}] Using destination as fallback: {model_ref}")

        if not model_ref:
            raise RuntimeError(f"No model reference found. Training output: {output}")

        # 7) Test the model first - using direct API
        logger.info(f"[{run_id}] Testing model: {model_ref}")

        test_payload = {
            'version': model_ref,
            'input': {
                'prompt': f"{token}, simple test",
                'num_outputs': 1,
                'num_inference_steps': 1
            }
        }

        test_response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=test_payload,
            timeout=30
        )

        if test_response.status_code != 201:
            raise RuntimeError(f"Model test failed: {test_response.status_code} - {test_response.text}")

        test_data = test_response.json()
        logger.info(f"[{run_id}] Model test successful")

        # 8) Generate ONE high-quality variation - OPTIMIZED FOR SPEED & QUALITY
        prompt = (
            f"TOK, a {meta.get('age')}-year-old {meta.get('gender')} child "
            f"wearing {meta.get('outfit')}, convert to children's book illustration style, "
            f"cartoon version of the person, animated character style, "
            f"maintain original facial features and likeness, same face structure, "
            f"same hair color and style, same eye color, same facial proportions, "
            f"digital art illustration, soft painting style, expressive cartoon eyes, "
            f"standing pose, full body visible, colorful children's book art"
        )
        
        logger.info(f"[{run_id}] Generating HIGH-QUALITY single variation with prompt: {prompt}")
        logger.info(f"[{run_id}] Using model: {model_ref}")
        
        # Generate character variation - using direct API
        generation_payload = {
            'version': model_ref,
            'input': {
                'model': 'dev',
                'prompt': prompt,
                'go_fast': False,
                'lora_scale': 1.2,
                'megapixels': '1',
                'num_outputs': 1,
                'aspect_ratio': '1:1',
                'output_format': 'webp',
                'guidance_scale': 4,
                'output_quality': 95,
                'prompt_strength': 0.9,
                'extra_lora_scale': 1.2,
                'num_inference_steps': 50
            }
        }

        generation_response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=generation_payload,
            timeout=30
        )

        if generation_response.status_code != 201:
            raise RuntimeError(f"Character generation failed: {generation_response.status_code} - {generation_response.text}")

        generation_data = generation_response.json()
        prediction_id = generation_data['id']

        # Poll for generation completion
        gen_status = None
        gen_poll_count = 0
        max_gen_polls = 40  # 10 minutes max

        while gen_status not in ('succeeded', 'failed', 'canceled') and gen_poll_count < max_gen_polls:
            time.sleep(15)
            gen_poll_count += 1
            
            gen_status_response = requests.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers=headers,
                timeout=30
            )
            
            if gen_status_response.status_code == 200:
                gen_data = gen_status_response.json()
                gen_status = gen_data['status']
                logger.info(f"[{run_id}] Generation status ({gen_poll_count}/40): {gen_status}")
                
                if gen_status == 'succeeded':
                    outs = gen_data.get('output', [])
                    break
            else:
                logger.warning(f"[{run_id}] Generation status check failed: {gen_status_response.status_code}")

        if gen_status != 'succeeded':
            raise RuntimeError(f"Character generation failed with status: {gen_status}")
        
        if outs:
            # Handle different output formats and ensure we get a URL string
            if isinstance(outs, list) and len(outs) > 0:
                selected_variation = str(outs[0])  # Convert to string to avoid FileOutput issues
            elif isinstance(outs, str):
                selected_variation = outs
            else:
                selected_variation = str(outs)
                
            logger.info(f"[{run_id}] Generated high-quality variation: {selected_variation}")
            
            # Start comic generation immediately
            logger.info(f"[{run_id}] Auto-starting comic generation...")
            threading.Thread(target=run_comic_generation, args=(run_id, selected_variation), daemon=True).start()
            
            result.update(
                success=True, 
                character_image=selected_variation,  # Single auto-selected variation as string
                model_ref=str(model_ref),  # Ensure this is also a string
                meta=meta,
                comic_started=True  # Flag that comic generation started
            )
        else:
            raise RuntimeError("No character variation generated")

    except Exception as err:
        logger.exception(f"[{run_id}] Pipeline error")
        result.update(success=False, error=str(err))
    finally:
        # Cleanup local files
        try:
            if zip_path and os.path.exists(zip_path):
                os.remove(zip_path)
                logger.info(f"[{run_id}] Cleaned up zip file")
        except Exception as cleanup_error:
            logger.warning(f"[{run_id}] Cleanup error: {cleanup_error}")
        
        store[run_id] = result

@app.route('/generate-comic', methods=['POST'])
def generate_comic():
    data = request.get_json(force=True)
    run_id = data['run_id']
    selected_variation = data['selected_variation']
    
    logger.info(f"[{run_id}] Starting comic generation with variation: {selected_variation}")
    threading.Thread(target=run_comic_generation, args=(run_id, selected_variation), daemon=True).start()
    return jsonify(success=True, comic_id=run_id)

def run_comic_generation(run_id, selected_variation_url):
    rec = store.get(run_id, {})
    meta = rec.get('meta', {})
    model_ref = rec.get('model_ref')
    child_name = meta.get('name', 'TOK')
    child_gender = meta.get('gender', 'boy')
    child_age = int(meta.get('age', 2))
    
    # Get consistent story elements based on gender
    story_elements = get_story_elements(child_gender, child_age)
    
    result = {'comic_id': run_id, 'success': False, 'status': 'generating', 'comic_scenes': {}}
    store[f"{run_id}_comic"] = result
    
    try:
        logger.info(f"[{run_id}] Generating comic scenes for {child_name} ({child_gender}, {child_age}y) using consistent characters")
        logger.info(f"[{run_id}] Story elements: {story_elements}")
        
        # Generate all 8 scenes with consistent characters
        comic_scenes = {}
        
        for scene_id, scene_data in COMIC_SCENES.items():
            logger.info(f"[{run_id}] Generating scene: {scene_id}")
            
            # Replace all placeholders in prompts and descriptions
            scene_prompt = scene_data['prompt'].format(
                child_name=child_name,
                child_outfit=meta.get('outfit', 'their outfit'),
                **story_elements
            )
            scene_description = scene_data['description'].format(
                child_name=child_name,
                child_outfit=meta.get('outfit', 'their outfit'),
                **story_elements
            )
            scene_question = scene_data.get('question', '').format(
                child_name=child_name,
                child_outfit=meta.get('outfit', 'their outfit'),
                **story_elements
            ) if scene_data.get('question') else None
            
            # Format options if they exist
            scene_options = None
            if scene_data.get('options'):
                scene_options = {
                    key: value.format(
                        child_name=child_name, 
                        child_outfit=meta.get('outfit', 'their outfit'),
                        **story_elements
                    ) 
                    for key, value in scene_data['options'].items()
                }
            
            # Format lesson if it exists
            scene_lesson = scene_data.get('lesson', '').format(
                child_name=child_name,
                child_outfit=meta.get('outfit', 'their outfit'),
                **story_elements
            ) if scene_data.get('lesson') else None
            
            # Create detailed prompt with character consistency instructions
            detailed_prompt = (
                f"{scene_prompt}, "
                f"featuring {child_name} a {child_age}-year-old {child_gender} "
                f"wearing {meta.get('outfit')}, "
                f"IMPORTANT: {child_name} must always wear {meta.get('outfit')} in every scene, "
                f"IMPORTANT: {story_elements['friend_description']} must always wear the same outfit throughout all scenes, "
                f"IMPORTANT: teacher/adult must always wear professional adult clothing never children's clothes, "
                f"IMPORTANT: maintain {child_name}'s exact appearance and facial features from reference image, "
                f"consistent character design, soft colors, friendly atmosphere, toddler appropriate"
            )
            
            try:
                # Use direct API call for scene generation to avoid FileOutput serialization issues
                headers = {
                    'Authorization': f"Token {REPL_TOKEN}",
                    'Content-Type': 'application/json'
                }
                
                scene_payload = {
                    'version': "black-forest-labs/flux-kontext-dev:a1408d2c4fd0af9bf22673accc5c066780fb26f50c70564eddf568601969aee8",
                    'input': {
                        'prompt': detailed_prompt,
                        'input_image': selected_variation_url,
                        'aspect_ratio': '1:1',
                        'num_inference_steps': 30,
                        'guidance': 2.5,
                        'output_format': 'jpg',
                        'output_quality': 80,
                        'go_fast': False
                    }
                }

                scene_response = requests.post(
                    "https://api.replicate.com/v1/predictions",
                    headers=headers,
                    json=scene_payload,
                    timeout=30
                )

                if scene_response.status_code != 201:
                    raise Exception(f"Scene generation failed: {scene_response.status_code} - {scene_response.text}")

                scene_data = scene_response.json()
                scene_prediction_id = scene_data['id']

                # Poll for scene completion
                scene_status = None
                scene_poll_count = 0
                max_scene_polls = 30  # 7.5 minutes max

                while scene_status not in ('succeeded', 'failed', 'canceled') and scene_poll_count < max_scene_polls:
                    time.sleep(15)
                    scene_poll_count += 1
                    
                    scene_status_response = requests.get(
                        f"https://api.replicate.com/v1/predictions/{scene_prediction_id}",
                        headers=headers,
                        timeout=30
                    )
                    
                    if scene_status_response.status_code == 200:
                        scene_status_data = scene_status_response.json()
                        scene_status = scene_status_data['status']
                        logger.info(f"[{run_id}] Scene {scene_id} status ({scene_poll_count}/30): {scene_status}")
                        
                        if scene_status == 'succeeded':
                            scene_output = scene_status_data.get('output', [])
                            break
                    else:
                        logger.warning(f"[{run_id}] Scene status check failed: {scene_status_response.status_code}")

                if scene_status != 'succeeded':
                    raise Exception(f"Scene generation failed with status: {scene_status}")
                
                # Extract the scene result URL
                if isinstance(scene_output, list) and len(scene_output) > 0:
                    scene_result_url = scene_output[0]
                elif isinstance(scene_output, str):
                    scene_result_url = scene_output
                else:
                    raise Exception(f"Unexpected scene output format: {type(scene_output)}")
                
                if scene_result_url:
                    comic_scenes[scene_id] = {
                        'image_url': scene_result_url,
                        'description': scene_description,
                        'question': scene_question,
                        'options': scene_options,
                        'lesson': scene_lesson
                    }
                    logger.info(f"[{run_id}] Successfully generated scene: {scene_id} -> {scene_result_url}")
                else:
                    raise Exception(f"No output generated for scene {scene_id}")
                    
            except Exception as scene_error:
                logger.error(f"[{run_id}] Error generating scene {scene_id}: {scene_error}")
                raise scene_error
            
            # Small delay between scenes to avoid rate limits
            time.sleep(3)
        
        result.update(success=True, status='completed', comic_scenes=comic_scenes)
        logger.info(f"[{run_id}] Comic generation completed successfully! Generated {len(comic_scenes)} scenes for {child_name}")
        
    except Exception as err:
        logger.exception(f"[{run_id}] Comic generation error")
        result.update(success=False, status='failed', error=str(err))
    
    store[f"{run_id}_comic"] = result

@app.route('/comic-results/<run_id>', methods=['GET'])
def get_comic_results(run_id):
    comic_key = f"{run_id}_comic"
    rec = store.get(comic_key)

    if rec and rec.get('status') == 'completed':
        # pop out the stored record
        record = store.pop(comic_key)
        # convert the scenes dict → a list of scene objects
        scenes_dict = record.get('comic_scenes', {})
        scene_list = [
            {"scene_id": sid, **data}
            for sid, data in scenes_dict.items()
        ]
        # return the reshaped payload
        return jsonify({
            "success":      record["success"],
            "status":       record["status"],
            "comic_scenes": scene_list
        })

    elif rec and rec.get('status') == 'failed':
        return jsonify(rec), 500

    return jsonify(success=False, status='generating'), 202

@app.route('/results/<run_id>', methods=['GET'])
def get_results(run_id):
    rec = store.get(run_id)
    if rec and rec.get('status'):
        return jsonify(rec)  # Don't pop yet, we need it for comic generation
    return jsonify(success=False, error='Not ready'), 202

# --- Quick Test Routes ---
@app.route('/test-one-scene', methods=['GET'])
def test_one_scene():
    """Test just one scene with Flux Kontext - for debugging"""
    
    character_image = "https://replicate.delivery/xezq/uDfi4E8gO1x2EafXqmAJECzSyIZMIxVvokTUlJd9cVaYWW7UA/out-0.webp"
    prompt = (
        "TOK sitting on playground holding bright red toy truck with yellow wheels, "
        "another small child nearby looking sad with no toys, colorful playground background, "
        "sunny day, digital children's book illustration for toddlers, soft warm colors, "
        "featuring TOK a 2-year-old boy wearing red polo tshirt and blue shorts and shoes"
    )
    
    try:
        result = rep_client.run(
            "black-forest-labs/flux-kontext-dev:a1408d2c4fd0af9bf22673accc5c066780fb26f50c70564eddf568601969aee8",
            input={
                'prompt': prompt,
                'input_image': character_image,
                'aspect_ratio': '1:1',
                'num_inference_steps': 30,
                'guidance': 2.5,
                'output_format': 'jpg',
                'output_quality': 80,
                'go_fast': False
            }
        )
        
        # Handle result properly
        if isinstance(result, str):
            output = result
        elif hasattr(result, '__iter__'):
            output = list(result)[0] if result else None
        else:
            output = str(result)
            
        return jsonify({
            'success': True,
            'generated_image': output,
            'message': 'Single scene test successful!'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# --- Debug/Test Routes ---
@app.route('/test-replicate', methods=['GET'])
def test_replicate():
    """Test Replicate connection"""
    try:
        # Test with direct API call instead of client
        headers = {
            'Authorization': f"Token {REPL_TOKEN}",
            'Content-Type': 'application/json'
        }
        
        response = requests.get(
            "https://api.replicate.com/v1/models",
            headers=headers,
            timeout=30,
            params={'limit': 3}
        )
        
        if response.status_code == 200:
            models_data = response.json()
            return jsonify({
                'success': True,
                'connection': 'OK',
                'status_code': response.status_code,
                'sample_models': [f"{m['owner']}/{m['name']}" for m in models_data.get('results', [])[:3]]
            })
        else:
            return jsonify({
                'success': False,
                'connection': 'Failed',
                'status_code': response.status_code,
                'error': response.text
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/test-cloudinary', methods=['GET'])
def test_cloudinary():
    """Test Cloudinary connection"""
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"test connection")
            tmp.flush()
            
            result = cloudinary.uploader.upload(
                tmp.name,
                resource_type="raw",
                public_id="test-connection"
            )
            
        os.unlink(tmp.name)
        return jsonify({
            'success': True,
            'cloudinary_working': True,
            'test_upload_url': result['secure_url']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check for Render"""
    return jsonify({
        'status': 'healthy',
        'message': 'TaleGenie AI Pipeline Server is running',
        'environment_check': {
            'replicate_token': '✓' if REPL_TOKEN else '✗',
            'cloudinary_cloud_name': '✓' if os.getenv('CLOUDINARY_CLOUD_NAME') else '✗',
            'cloudinary_api_key': '✓' if os.getenv('CLOUDINARY_API_KEY') else '✗',
            'cloudinary_api_secret': '✓' if os.getenv('CLOUDINARY_API_SECRET') else '✗'
        }
    })

if __name__ == '__main__':
    print("=== TaleGenie AI Pipeline Server ===")
    print("Make sure environment variables are set:")
    print(f"- REPLICATE_API_TOKEN: {'✓' if REPL_TOKEN else '✗'}")
    print(f"- CLOUDINARY_CLOUD_NAME: {'✓' if os.getenv('CLOUDINARY_CLOUD_NAME') else '✗'}")
    print(f"- CLOUDINARY_API_KEY: {'✓' if os.getenv('CLOUDINARY_API_KEY') else '✗'}")
    print(f"- CLOUDINARY_API_SECRET: {'✓' if os.getenv('CLOUDINARY_API_SECRET') else '✗'}")
    print("\n🧪 Test endpoints:")
    print("- Health check: http://localhost:8000/health")
    print("- Single scene: http://localhost:8000/test-one-scene")
    print("- Replicate test: http://localhost:8000/test-replicate")
    print("- Cloudinary test: http://localhost:8000/test-cloudinary")
    print("\nStarting server on http://localhost:8000")
    app.run(debug=True, host='0.0.0.0', port=8000)
