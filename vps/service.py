from flask import Flask, request, jsonify
from pathlib import Path
import tempfile
import os
from .core import VisualPositioningSystem
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global VPS instance
vps = None

def create_app(config_path: str = "configs/default.yaml"):
    """Create and configure the Flask application."""
    global vps
    
    # Initialize VPS system
    vps = VisualPositioningSystem(config_path)
    logger.info("VPS system initialized successfully")
    
    return app

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'VPS',
        'reference_info': vps.get_reference_info()
    })

@app.route('/localize', methods=['POST'])
def localize():
    """Localization endpoint."""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Check file format
        allowed_extensions = vps.config['service']['supported_formats']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            return jsonify({'error': f'Unsupported file format. Supported: {allowed_extensions}'}), 400
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            file.save(tmp_file.name)
            query_image_path = tmp_file.name
        
        # Get optional parameters
        use_gt_depth = request.form.get('use_gt_depth', 'true').lower() == 'true'
        
        # Perform localization
        result = vps.localize(query_image_path, use_gt_depth=use_gt_depth)
        
        # Clean up temporary file
        if vps.config['service']['cleanup_temp_files']:
            os.unlink(query_image_path)
        
        # Convert numpy arrays to lists for JSON serialization
        if result['pose'] is not None:
            result['pose'] = result['pose'].tolist()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during localization: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/reference_info', methods=['GET'])
def get_reference_info():
    """Get reference database information."""
    return jsonify(vps.get_reference_info())

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False) 