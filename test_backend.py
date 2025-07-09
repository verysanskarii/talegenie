import logging
import os
import time
import threading
import zipfile
import uuid
import requests

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

# --- Helper: gender → robust elements ---
def get_story_elements(child_gender, child_age):
    """Generate consistent story elements based on child's gender and age."""
    if child_gender.lower() in ['girl', 'female']:
        return {
            'toy_type': 'small pink teddy bear',
            'toy_color': 'pink',
            'friend_gender': 'girl',
            'friend_description': (
                'another little girl with silky blonde hair in two pigtails, '
                'wearing a light purple dress and white sandals, gently clutching her own tiny pink teddy bear'
            ),
            'teacher_description': (
                'a caring female teacher in a crisp blue blouse and charcoal slacks, '
                'smiling warmly as she watches the children'
            )
        }
    else:
        return {
            'toy_type': 'red toy truck with shiny yellow wheels',
            'toy_color': 'red',
            'friend_gender': 'boy',
            'friend_description': (
                'another little boy with tousled brown hair, '
                'wearing a green t-shirt and khaki shorts, excitedly holding a matching blue toy car'
            ),
            'teacher_description': (
                'a friendly male teacher in a neat button-up shirt and dark jeans, '
                'leaning forward with an encouraging smile'
            )
        }

# --- Comic story scenes ---
COMIC_SCENES = {
    'scene1': {
        'description': "{child_name} is sitting on the playground holding a bright {toy_color} {toy_type}.\nAnother little {friend_gender} sits nearby looking sad with no toys to play with.",
        'question': "What should {child_name} do?",
        'options': {
            'A': "Share the {toy_type} with the other {friend_gender}",
            'B': "Keep playing alone with the {toy_type}"
        },
        'prompt': "{child_name} sitting on soft playground grass holding bright {toy_color} {toy_type}, {friend_description}, colorful playground with slides in background, sunny day, simple children's book illustration for toddlers, soft warm colors, friendly atmosphere"
    },
    'scene2A': {
        'description': "{child_name} stands up with the {toy_color} {toy_type} and walks toward the sad {friend_gender}.\n{child_name} has a big smile and is holding out the {toy_type} to share.",
        'question': "How should {child_name} share the {toy_type}?",
        'options': {
            'A': "Let the other {friend_gender} play with the {toy_type} first",
            'B': "Play with the {toy_type} together"
        },
        'prompt': "{child_name} walking toward {friend_description}, offering the bright {toy_color} {toy_type} with a happy smile, simple playground background, digital children's book illustration style, warm and kind moment"
    },
    'scene2B': {
        'description': "{child_name} keeps playing with the {toy_color} {toy_type} but glances at the sad {friend_gender}.\n{child_name} feels unsure what to do.",
        'question': "What should {child_name} do now?",
        'options': {
            'A': "Go share the {toy_color} {toy_type}",
            'B': "Ask a grown-up for help"
        },
        'prompt': "{child_name} sitting on ground holding {toy_color} {toy_type}, turning head to look at {friend_description}, thoughtful expression, simple playground illustration for toddlers"
    },
    'scene3A1': {
        'description': "The other {friend_gender} is now playing happily with the {toy_color} {toy_type}.\n{child_name} watches nearby with a big smile, waiting patiently.",
        'question': "What happens next?",
        'options': {
            'A': "They take turns with the {toy_type}",
            'B': "More kids want to play too"
        },
        'prompt': "{friend_description} playing with {toy_color} {toy_type}, {child_name} smiling nearby, cooperative scene, children's book illustration style"
    },
    'scene3A2': {
        'description': "{child_name} and the other {friend_gender} are both playing with the {toy_color} {toy_type} together, both smiling.",
        'question': "What happens next?",
        'options': {
            'A': "They become good friends",
            'B': "They find more toys to share"
        },
        'prompt': "{child_name} and {friend_description} playing together with {toy_color} {toy_type}, joyful scene, toddler illustration style"
    },
    'scene3B1': {
        'description': "{child_name} stops playing and walks over to share the {toy_color} {toy_type} with the other {friend_gender}, feeling happy.",
        'question': "How does {child_name} feel about sharing?",
        'options': {
            'A': "Happy to make a friend",
            'B': "Proud of being kind"
        },
        'prompt': "{child_name} approaching {friend_description} with {toy_color} {toy_type}, warm smile, playground illustration"
    },
    'scene3B2': {
        'description': "A caring adult comes over to help {child_name} learn about sharing.\nThe adult smiles and offers guidance.",
        'question': "What does {child_name} learn?",
        'options': {
            'A': "Sharing makes everyone happy",
            'B': "Adults help us learn"
        },
        'prompt': "{teacher_description} kneeling beside {child_name} holding {toy_color} {toy_type}, gentle guidance moment, bright playground illustration for toddlers, maintain character likeness, keep {child_name} wearing the same outfit"
    },
    'scene4': {
        'description': "{child_name} and the other {friend_gender} play happily together, sharing the {toy_color} {toy_type} and smiling.",
        'question': None,
        'options': None,
        'prompt': "{child_name} and {friend_description} playing with {toy_color} {toy_type}, joyful friendship scene, children's book style illustration",
        'lesson': "Sharing your {toy_type} makes everyone happy and helps build friendships!"
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
        meta = {k: request.form[k] for k in ('name','gender','age','outfit')}
        store[run_id] = {'meta': meta}

        files = request.files.getlist('photos')
        if len(files) < 5:
            raise ValueError("Need at least 5 photos")
        paths = []
        for i, f in enumerate(files, 1):
            ext = secure_filename(f.filename).rsplit('.',1)[-1].lower() if f.filename and '.' in f.filename else 'jpg'
            outp = f"temp_photos/{run_id}_{i}.{ext}"
            img = Image.open(f.stream).convert('RGB')
            img.thumbnail((1024,1024), Image.Resampling.LANCZOS)
            img.save(outp,'JPEG',quality=85)
            paths.append(outp)
        zip_path = f"temp_zips/{run_id}.zip"
        with zipfile.ZipFile(zip_path,'w') as zf:
            for idx, p in enumerate(paths,1):
                zf.write(p, f"photo_{idx:02d}.jpg")
        token_str = "TOK"
        store[run_id].update(token=token_str, zip_path=zip_path)
        return jsonify(success=True, character_id=run_id, token_string=token_str, zip_path=zip_path)
    except Exception as e:
        logger.exception(f"[{run_id}] upload error")
        return jsonify(success=False, error=str(e)), 500

@app.route('/start-secure-pipeline', methods=['POST'])
def start_pipeline():
    data = request.get_json(force=True)
    run_id = data['character_id']
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
        model_check = requests.get(f"https://api.replicate.com/v1/models/{dest}", headers=headers)
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
                          headers=headers, json=payload)
            create_response.raise_for_status()
            logger.info(f"[{run_id}] Model created successfully")
        else:
            logger.info(f"[{run_id}] Model already exists")

        # 3) Start training
        training = rep_client.trainings.create(
            version="ostris/flux-dev-lora-trainer:e440909d3512c31646ee2e0c7d6f6f4923224863a6a10c494606e79fb5844497",
            input={
                'input_images': signed_url,
                'token_string': token,
                'caption_prefix': f"a photo of {token}",
                'max_train_steps': 1000,
                'learning_rate': 1e-4,
                'batch_size': 1,
                'resolution': 512
            },
            destination=dest
        )
        logger.info(f"[{run_id}] Training started: {training.id}")

        # 4) Poll until done with better error handling
        status = None
        timeout_count = 0
        max_timeouts = 3
        
        while status not in ('succeeded','failed','canceled'):
            try:
                time.sleep(15)
                training = rep_client.trainings.get(training.id)
                status = training.status
                logger.info(f"[{run_id}] Training status: {status}")
                timeout_count = 0  # Reset timeout count on successful check
                
            except Exception as poll_error:
                timeout_count += 1
                logger.warning(f"[{run_id}] Polling error ({timeout_count}/{max_timeouts}): {poll_error}")
                
                if timeout_count >= max_timeouts:
                    logger.error(f"[{run_id}] Too many polling failures, checking if training completed...")
                    
                    # Final attempt to get training status
                    try:
                        training = rep_client.trainings.get(training.id)
                        status = training.status
                        logger.info(f"[{run_id}] Final status check: {status}")
                        
                        if status == 'succeeded':
                            logger.info(f"[{run_id}] Training actually succeeded despite polling errors!")
                            break
                        elif status in ('failed', 'canceled'):
                            raise RuntimeError(f"Training {status}")
                        else:
                            # Training might still be running, wait a bit more
                            time.sleep(30)
                            training = rep_client.trainings.get(training.id)
                            status = training.status
                            if status == 'succeeded':
                                break
                            else:
                                raise RuntimeError(f"Training polling failed after multiple attempts. Last status: {status}")
                                
                    except Exception as final_error:
                        raise RuntimeError(f"Training polling failed: {final_error}")
                else:
                    # Wait longer before retrying
                    time.sleep(30)
                    
        result['status'] = status

        if status != 'succeeded':
            raise RuntimeError(f"Training failed with status {status}")

        # 5) Wait a moment and get final training output
        time.sleep(5)
        training = rep_client.trainings.get(training.id)
        output = training.output or {}
        
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

        # 7) Test the model first
        logger.info(f"[{run_id}] Testing model: {model_ref}")
        try:
            test_prompt = f"{token}, simple test"
            test_output = rep_client.run(
                model_ref,
                input={
                    'prompt': test_prompt,
                    'num_outputs': 1,
                    'num_inference_steps': 1
                }
            )
            test_result = list(test_output)
            logger.info(f"[{run_id}] Model test successful")
        except Exception as test_error:
            logger.error(f"[{run_id}] Model test failed: {test_error}")
            raise RuntimeError(f"Model test failed: {test_error}")

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
        
        outs = list(rep_client.run(
            model_ref,
            input={
                'model': 'dev',
                'prompt': prompt,
                'go_fast': False,
                'lora_scale': 1.2,             # Higher for better resemblance
                'megapixels': '1',
                'num_outputs': 1,              # ⭐ Only 1 instead of 4
                'aspect_ratio': '1:1',
                'output_format': 'webp',
                'guidance_scale': 4,           # Higher for better prompt following
                'output_quality': 95,          # Much higher quality
                'prompt_strength': 0.9,        # Stronger influence
                'extra_lora_scale': 1.2,       # Higher for better resemblance
                'num_inference_steps': 50      # ✅ FIXED: Changed from 75 to 50
            }
        ))
        
        if outs:
            # Directly use the single variation
            selected_variation = outs[0]
            logger.info(f"[{run_id}] Generated high-quality variation: {selected_variation}")
            
            # Start comic generation immediately
            logger.info(f"[{run_id}] Auto-starting comic generation...")
            threading.Thread(target=run_comic_generation, args=(run_id, selected_variation), daemon=True).start()
            
            result.update(
                success=True, 
                character_image=selected_variation,  # Single auto-selected variation
                model_ref=model_ref, 
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
                # Use WORKING Flux Kontext Dev version
                scene_output = rep_client.run(
                    "black-forest-labs/flux-kontext-dev:a1408d2c4fd0af9bf22673accc5c066780fb26f50c70564eddf568601969aee8",
                    input={
                        'prompt': detailed_prompt,
                        'input_image': selected_variation_url,
                        'aspect_ratio': '1:1',
                        'num_inference_steps': 30,
                        'guidance': 2.5,
                        'output_format': 'jpg',
                        'output_quality': 80,
                        'go_fast': False
                    }
                )
                
                # Handle different result types properly
                if isinstance(scene_output, str):
                    scene_result_url = scene_output
                elif hasattr(scene_output, '__iter__'):
                    scene_list = list(scene_output)
                    scene_result_url = scene_list[0] if scene_list else None
                else:
                    scene_result_url = str(scene_output)
                
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
        return jsonify(store.pop(comic_key))
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
        models = list(rep_client.models.list())[:3]
        return jsonify({
            'success': True,
            'connection': 'OK',
            'sample_models': [f"{m.owner}/{m.name}" for m in models]
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

if __name__ == '__main__':
    print("=== TaleGenie AI Pipeline Server ===")
    print("Make sure environment variables are set:")
    print(f"- REPLICATE_API_TOKEN: {'✓' if REPL_TOKEN else '✗'}")
    print(f"- CLOUDINARY_CLOUD_NAME: {'✓' if os.getenv('CLOUDINARY_CLOUD_NAME') else '✗'}")
    print(f"- CLOUDINARY_API_KEY: {'✓' if os.getenv('CLOUDINARY_API_KEY') else '✗'}")
    print(f"- CLOUDINARY_API_SECRET: {'✓' if os.getenv('CLOUDINARY_API_SECRET') else '✗'}")
    print("\n🧪 Test endpoints:")
    print("- Single scene: http://localhost:8000/test-one-scene")
    print("- Health check: http://localhost:8000/test-replicate")
    print("\nStarting server on http://localhost:8000")
    app.run(debug=True, host='0.0.0.0', port=8000)