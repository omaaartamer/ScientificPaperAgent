from flask import Blueprint, request, Response, render_template, jsonify, send_from_directory
from flask_cors import CORS
from flaskApp.config import Config
from agentAll import run_agent
import requests
import json
import time
import asyncio
from langchain_core.messages import HumanMessage  # Add this import
# Create blueprints
webhook_bp = Blueprint('webhook', __name__)
web_bp = Blueprint('web', __name__)

# Web UI Routes
@web_bp.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@web_bp.route('/api/chat', methods=['POST'])
def chat():
    """Handle web chat requests"""
    try:
        all_start = time.time()
        data = request.json
        message = data.get('message')
        
        if not message:
            return jsonify({
                'status': 'error',
                'message': 'No message provided'
            }), 400 
        start = time.time()
        response = asyncio.run(run_agent(message))
        end = time.time()

        #print('WEB endpoint - response generation: ', end-start)
        #print('WEB endpoint - response generation: ', end-all_start)

        return jsonify({
            'status': 'success',
            'response': response
        })
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
        
@web_bp.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    try:
        return send_from_directory('static', path)
    except Exception as e:
        print(f"Error serving static file: {str(e)}")
        return Response(status=404)

# Error handlers
@web_bp.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Resource not found'
    }), 404

@web_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

# Health check endpoint
@web_bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Research Assistant',
        'version': '1.0.0'
    })
