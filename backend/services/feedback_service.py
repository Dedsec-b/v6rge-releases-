from flask import Blueprint, request, jsonify
import requests
import traceback
from config import ADMIN_EMAIL, MAILERSEND_API_KEY, SENDER_EMAIL

feedback_bp = Blueprint('feedback', __name__)

@feedback_bp.route('/send_feedback', methods=['POST'])
def send_feedback():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        user_email = data.get('email', 'Anonymous')
        feedback_type = data.get('type', 'General')
        description = data.get('description', '')

        if not description:
            return jsonify({"error": "Description is required"}), 400

        # Construct Email
        subject = f"V6rge Feedback: {feedback_type.upper()}"
        
        # HTML Content
        html_content = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; border: 1px solid #ddd; max-width: 600px;">
            <h2 style="color: #333;">New Feedback Received</h2>
            <hr>
            <p><strong>Type:</strong> {feedback_type}</p>
            <p><strong>From:</strong> {user_email}</p>
            <hr>
            <h3 style="color: #555;">Message:</h3>
            <p style="white-space: pre-wrap; background: #f9f9f9; padding: 15px; border-radius: 5px;">{description}</p>
        </div>
        """
        
        text_content = f"New Feedback ({feedback_type}) from {user_email}:\n\n{description}"

        # MailerSend API Endpoint
        url = "https://api.mailersend.com/v1/email"
        
        headers = {
            "Authorization": f"Bearer {MAILERSEND_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "from": {
                "email": SENDER_EMAIL,
                "name": "V6rge Feedback System"
            },
            "to": [
                {
                    "email": ADMIN_EMAIL,
                    "name": "Admin"
                }
            ],
            "subject": subject,
            "text": text_content,
            "html": html_content,
            "reply_to": {
                "email": user_email if user_email and '@' in user_email else ADMIN_EMAIL,
                "name": "User"
            }
        }

        response = requests.post(url, json=payload, headers=headers)

        # MailerSend returns 202 Accepted on success
        if response.status_code in [200, 201, 202]:
            return jsonify({"status": "success", "message": "Feedback sent successfully"}), 200
        else:
            print(f"MailerSend Error: {response.status_code} - {response.text}")
            return jsonify({"error": f"Failed to send email: {response.text}"}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
