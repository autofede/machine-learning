from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import numpy as np
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Database connection configuration
db_config = {
	'host': 'localhost',
	'user': 'root',
	'password': 'admin123',
	'database': '9014_db'
}


# Helper function to establish database connection
def get_db_connection():
	return mysql.connector.connect(**db_config)


# Helper to convert datetime objects to string in JSON responses
def datetime_converter(o):
	if isinstance(o, datetime):
		return o.isoformat()


# 1. API Endpoints for Survey and Questions
@app.route('/api/surveys', methods=['GET'])
def get_surveys():
	"""Get all available surveys"""
	try:
		conn = get_db_connection()
		cursor = conn.cursor(dictionary=True)

		query = "SELECT survey_id, title, description FROM survey"
		cursor.execute(query)
		surveys = cursor.fetchall()

		cursor.close()
		conn.close()

		return jsonify({"success": True, "data": surveys})
	except Exception as e:
		return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/surveys/<int:survey_id>/questions', methods=['GET'])
def get_survey_questions(survey_id):
	"""Get all questions for a specific survey"""
	try:
		conn = get_db_connection()
		cursor = conn.cursor(dictionary=True)

		# Get survey details
		survey_query = "SELECT title, description FROM survey WHERE survey_id = %s"
		cursor.execute(survey_query, (survey_id,))
		survey = cursor.fetchone()

		if not survey:
			cursor.close()
			conn.close()
			return jsonify({"success": False, "error": "Survey not found"}), 404

		# Get questions for survey
		query = """
        SELECT q.question_id, q.question_text, q.description, q.type_id, 
               qt.type_name AS question_type, q.is_required, q.sequence_number
        FROM question q
        JOIN question_type qt ON q.type_id = qt.type_id
        WHERE q.survey_id = %s
        ORDER BY q.sequence_number
        """
		cursor.execute(query, (survey_id,))
		questions = cursor.fetchall()

		# For each question, get options if needed
		for q in questions:
			if q['type_id'] in [1, 2]:  # Only fetch options for dropdown and checkbox types
				options_query = """
                SELECT option_id, option_text, sequence_number
                FROM question_option
                WHERE question_id = %s
                ORDER BY sequence_number
                """
				cursor.execute(options_query, (q['question_id'],))
				q['options'] = cursor.fetchall()

		cursor.close()
		conn.close()

		return jsonify({
			"success": True,
			"data": {
				"survey": survey,
				"questions": questions
			}
		})
	except Exception as e:
		return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/surveys', methods=['POST'])
def create_survey():
	"""Create a new survey"""
	try:
		data = request.json

		if not data or 'title' not in data:
			return jsonify({"success": False, "error": "Title is required"}), 400

		conn = get_db_connection()
		cursor = conn.cursor(dictionary=True)

		# Insert new survey
		insert_query = """
        INSERT INTO survey (title, description) 
        VALUES (%s, %s)
        """
		cursor.execute(insert_query, (data['title'], data.get('description', '')))
		survey_id = cursor.lastrowid
		conn.commit()

		# If questions are provided, add them
		if 'questions' in data and isinstance(data['questions'], list):
			for q in data['questions']:
				# Insert question
				q_query = """
                INSERT INTO question 
                (survey_id, question_text, description, type_id, is_required, sequence_number) 
                VALUES (%s, %s, %s, %s, %s, %s)
                """
				cursor.execute(q_query, (
					survey_id,
					q['question_text'],
					q.get('description', ''),
					q.get('type_id', 3),  # Default to text input if not specified
					q.get('is_required', 0),
					q.get('sequence_number', 1)
				))
				question_id = cursor.lastrowid

				# Insert options if provided
				if 'options' in q and isinstance(q['options'], list):
					for i, opt in enumerate(q['options']):
						opt_query = """
                        INSERT INTO question_option 
                        (question_id, option_text, sequence_number) 
                        VALUES (%s, %s, %s)
                        """
						cursor.execute(opt_query, (
							question_id,
							opt.get('option_text', ''),
							opt.get('sequence_number', i + 1)
						))

				conn.commit()

		cursor.close()
		conn.close()

		return jsonify({
			"success": True,
			"message": "Survey created successfully",
			"survey_id": survey_id
		})
	except Exception as e:
		return jsonify({"success": False, "error": str(e)}), 500


# 2. API Endpoints for Satisfaction Data
@app.route('/api/satisfaction/product-quality', methods=['GET'])
def get_product_quality_satisfaction():
	"""Get product quality satisfaction data (from question_id 6)"""
	try:
		conn = get_db_connection()
		cursor = conn.cursor(dictionary=True)

		query = """
        SELECT 
            numerical_answer AS rating,
            COUNT(*) AS count
        FROM answer
        WHERE question_id = 6
        GROUP BY numerical_answer
        ORDER BY numerical_answer
        """
		cursor.execute(query)
		ratings = cursor.fetchall()

		cursor.close()
		conn.close()

		return jsonify({"success": True, "data": ratings})
	except Exception as e:
		return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/satisfaction/role', methods=['GET'])
def get_role_satisfaction():
	"""Get role satisfaction data (from question_id 56)"""
	try:
		conn = get_db_connection()
		cursor = conn.cursor(dictionary=True)

		query = """
        SELECT 
            numerical_answer AS rating,
            COUNT(*) AS count
        FROM answer
        WHERE question_id = 56
        GROUP BY numerical_answer
        ORDER BY numerical_answer
        """
		cursor.execute(query)
		ratings = cursor.fetchall()

		cursor.close()
		conn.close()

		return jsonify({"success": True, "data": ratings})
	except Exception as e:
		return jsonify({"success": False, "error": str(e)}), 500


# 3. API Endpoint for Age Distribution
@app.route('/api/age-distribution', methods=['GET'])
def get_age_distribution():
	"""Get age distribution in 10-year intervals"""
	try:
		conn = get_db_connection()
		cursor = conn.cursor(dictionary=True)

		# Count total respondents
		respondent_query = "SELECT COUNT(DISTINCT respondent_id) as total FROM respondent"
		cursor.execute(respondent_query)
		total_respondents = cursor.fetchone()['total']

		# Get ages from question_id = 1 (assuming this is an age question)
		age_query = """
        SELECT text_answer FROM answer
        WHERE question_id = 1 AND text_answer REGEXP '^[0-9]+$'
        """
		cursor.execute(age_query)
		ages_data = cursor.fetchall()

		cursor.close()
		conn.close()

		# Process age data
		ages = []
		for row in ages_data:
			try:
				age = int(row['text_answer'])
				if 0 <= age <= 120:  # Basic age validation
					ages.append(age)
			except (ValueError, TypeError):
				continue

		# Create age groups (10-year intervals)
		bins = list(range(0, 121, 10))
		labels = [f"{bin}-{bin + 9}" for bin in bins[:-1]]

		# Count ages by group
		age_counts, _ = np.histogram(ages, bins=bins)

		# Create result
		age_distribution = []
		for i, label in enumerate(labels):
			if age_counts[i] > 0:  # Only include non-empty groups
				age_distribution.append({
					"ageGroup": label,
					"count": int(age_counts[i])  # Convert numpy int to Python int for JSON
				})

		return jsonify({
			"success": True,
			"data": {
				"totalRespondents": total_respondents,
				"distribution": age_distribution
			}
		})
	except Exception as e:
		return jsonify({"success": False, "error": str(e)}), 500


# 4. API Endpoint for Respondent Details with Pagination
@app.route('/api/respondents', methods=['GET'])
def get_respondents():
	"""Get paginated respondent details"""
	try:
		page = int(request.args.get('page', 1))
		page_size = int(request.args.get('page_size', 50))

		# Validate pagination parameters
		if page < 1:
			page = 1
		if page_size < 1 or page_size > 100:
			page_size = 50

		offset = (page - 1) * page_size

		conn = get_db_connection()
		cursor = conn.cursor(dictionary=True)

		# Get total count
		count_query = "SELECT COUNT(*) as total FROM respondent"
		cursor.execute(count_query)
		total_count = cursor.fetchone()['total']

		# Calculate total pages
		total_pages = (total_count + page_size - 1) // page_size

		# Get paginated respondent data
		query = """
        SELECT 
            respondent_id, 
            user_name, 
            email, 
            ip_address, 
            created_at, 
            updated_at
        FROM respondent
        ORDER BY respondent_id
        LIMIT %s OFFSET %s
        """
		cursor.execute(query, (page_size, offset))
		respondents = cursor.fetchall()

		cursor.close()
		conn.close()

		# Convert datetime objects for JSON serialization
		for resp in respondents:
			if 'created_at' in resp and resp['created_at']:
				resp['created_at'] = resp['created_at'].isoformat()
			if 'updated_at' in resp and resp['updated_at']:
				resp['updated_at'] = resp['updated_at'].isoformat()

		return jsonify({
			"success": True,
			"data": {
				"respondents": respondents,
				"pagination": {
					"page": page,
					"pageSize": page_size,
					"totalPages": total_pages,
					"totalCount": total_count
				}
			}
		})
	except Exception as e:
		return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/respondents/ids', methods=['GET'])
def get_respondent_ids():
	"""Get list of all respondent IDs"""
	try:
		conn = get_db_connection()
		cursor = conn.cursor(dictionary=True)

		query = "SELECT DISTINCT respondent_id FROM respondent ORDER BY respondent_id"
		cursor.execute(query)
		respondent_ids = [row['respondent_id'] for row in cursor.fetchall()]

		cursor.close()
		conn.close()

		return jsonify({"success": True, "data": respondent_ids})
	except Exception as e:
		return jsonify({"success": False, "error": str(e)}), 500


# 5. API Endpoints for Respondent Answers
# @app.route('/api/respondents/<int:respondent_id>/answers', methods=['GET'])
# def get_respondent_answers(respondent_id):
# 	"""Get all answers for a specific respondent"""
# 	try:
# 		conn = get_db_connection()
# 		cursor = conn.cursor(dictionary=True)
#
# 		# Check if respondent exists
# 		check_query = "SELECT COUNT(*) as count FROM respondent WHERE respondent_id = %s"
# 		cursor.execute(check_query, (respondent_id,))
# 		if cursor.fetchone()['count'] == 0:
# 			cursor.close()
# 			conn.close()
# 			return jsonify({"success": False, "error": "Respondent not found"}), 404
#
# 		# Get respondent's answers
# 		query = """
#         SELECT
#             a.answer_id,
#             a.response_id,
#             q.question_id,
#             q.question_text,
#             q.type_id,
#             a.text_answer,
#             a.numerical_answer,
#             a.option_id,
#             (
#                 SELECT qo.option_text
#                 FROM question_option qo
#                 WHERE qo.question_id = a.question_id
#                   AND qo.sequence_number = a.option_id
#             ) AS selected_option
#         FROM answer a
#         JOIN question q ON a.question_id = q.question_id
#         WHERE a.response_id = %s
#         ORDER BY q.question_id
#         """
# 		cursor.execute(query, (respondent_id,))
# 		answers = cursor.fetchall()
#
# 		cursor.close()
# 		conn.close()
#
# 		return jsonify({"success": True, "data": answers})
# 	except Exception as e:
# 		return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/respondents/<int:respondent_id>/answers', methods=['GET'])
def get_respondent_answers(respondent_id):
    """Get all answers for a specific respondent"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Check if respondent exists
        check_query = "SELECT COUNT(*) as count FROM respondent WHERE respondent_id = %s"
        cursor.execute(check_query, (respondent_id,))
        if cursor.fetchone()['count'] == 0:
            cursor.close()
            conn.close()
            return jsonify({"success": False, "error": "Respondent not found"}), 404

        # Get respondent's answers
        query = """
        SELECT 
            a.answer_id,
            a.response_id,
            q.question_id,
            q.question_text,
            q.type_id,
            a.text_answer,
            a.numerical_answer,
            a.option_id,
            (
                SELECT qo.option_text 
                FROM question_option qo 
                WHERE qo.question_id = a.question_id 
                  AND qo.sequence_number = a.option_id
            ) AS selected_option
        FROM answer a 
        JOIN question q ON a.question_id = q.question_id
        WHERE a.response_id = %s 
        ORDER BY q.question_id
        """
        cursor.execute(query, (respondent_id,))
        answers = cursor.fetchall()

        # For each answer, check if question type_id is 1 and fetch all options
        for answer in answers:
            if answer['type_id'] == 1:  # For single choice questions
                options_query = """
                SELECT option_id, option_text, sequence_number
                FROM question_option
                WHERE question_id = %s
                ORDER BY sequence_number
                """
                cursor.execute(options_query, (answer['question_id'],))
                answer['options'] = cursor.fetchall()

        cursor.close()
        conn.close()

        return jsonify({"success": True, "data": answers})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# @app.route('/api/respondents/<int:respondent_id>/answers', methods=['PUT'])
# def update_answer(answer_id):
# 	"""Update a specific answer"""
# 	try:
# 		data = request.json
# 		if not data:
# 			return jsonify({"success": False, "error": "No data provided"}), 400
#
# 		conn = get_db_connection()
# 		cursor = conn.cursor(dictionary=True)
#
# 		# Check if answer exists
# 		check_query = "SELECT COUNT(*) as count FROM answer WHERE answer_id = %s"
# 		cursor.execute(check_query, (answer_id,))
# 		if cursor.fetchone()['count'] == 0:
# 			cursor.close()
# 			conn.close()
# 			return jsonify({"success": False, "error": "Answer not found"}), 404
#
# 		# Update answer
# 		update_fields = []
# 		update_values = []
#
# 		if 'text_answer' in data:
# 			update_fields.append("text_answer = %s")
# 			update_values.append(data['text_answer'])
#
# 		if 'numerical_answer' in data:
# 			update_fields.append("numerical_answer = %s")
# 			update_values.append(data['numerical_answer'])
#
# 		if 'option_id' in data:
# 			update_fields.append("option_id = %s")
# 			update_values.append(data['option_id'])
#
# 		if not update_fields:
# 			cursor.close()
# 			conn.close()
# 			return jsonify({"success": False, "error": "No valid fields to update"}), 400
#
# 		# Add answer_id to values
# 		update_values.append(answer_id)
#
# 		# Create and execute update query
# 		update_query = f"UPDATE answer SET {', '.join(update_fields)} WHERE answer_id = %s"
# 		cursor.execute(update_query, tuple(update_values))
# 		conn.commit()
#
# 		cursor.close()
# 		conn.close()
#
# 		return jsonify({
# 			"success": True,
# 			"message": "Answer updated successfully"
# 		})
# 	except Exception as e:
# 		return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/respondents/<int:respondent_id>/answers', methods=['PUT'])
def update_respondent_answers(respondent_id):
	"""Update answers for a specific respondent"""
	try:
		data = request.json
		if not data or 'answers' not in data:
			return jsonify({"success": False, "error": "No answers data provided"}), 400

		conn = get_db_connection()
		cursor = conn.cursor(dictionary=True)

		# Check if respondent exists
		check_query = "SELECT COUNT(*) as count FROM respondent WHERE respondent_id = %s"
		cursor.execute(check_query, (respondent_id,))
		if cursor.fetchone()['count'] == 0:
			cursor.close()
			conn.close()
			return jsonify({"success": False, "error": "Respondent not found"}), 404

		# Track which answers were updated successfully
		updated_answers = []
		failed_answers = []

		# Process each answer in the array
		for answer_data in data['answers']:
			try:
				# Validate required fields
				if 'answer_id' not in answer_data or 'question_id' not in answer_data or 'response_id' not in answer_data:
					failed_answers.append({
						"answer_data": answer_data,
						"error": "Missing required fields (answer_id, question_id, or response_id)"
					})
					continue

				# Check if answer exists and matches the respondent and question
				check_answer_query = """
                SELECT COUNT(*) as count 
                FROM answer 
                WHERE answer_id = %s AND question_id = %s AND response_id = %s
                """
				cursor.execute(check_answer_query, (
					answer_data['answer_id'],
					answer_data['question_id'],
					answer_data['response_id']
				))

				if cursor.fetchone()['count'] == 0:
					failed_answers.append({
						"answer_id": answer_data['answer_id'],
						"error": "Answer not found or doesn't match question_id and response_id"
					})
					continue

				# Prepare update fields
				update_fields = []
				update_values = []

				# Handle text_answer
				if 'text_answer' in answer_data and answer_data['text_answer'] is not None:
					update_fields.append("text_answer = %s")
					update_values.append(answer_data['text_answer'])

				# Handle numerical_answer
				if 'numerical_answer' in answer_data and answer_data['numerical_answer'] is not None:
					update_fields.append("numerical_answer = %s")
					update_values.append(answer_data['numerical_answer'])

				# Handle selected_option by finding its option_id
				if 'selected_option' in answer_data and answer_data['selected_option'] is not None:
					option_lookup_query = """
                    SELECT option_id 
                    FROM question_option 
                    WHERE question_id = %s AND option_text = %s
                    """
					cursor.execute(option_lookup_query, (
						answer_data['question_id'],
						answer_data['selected_option']
					))
					option_result = cursor.fetchone()

					if option_result:
						update_fields.append("option_id = %s")
						update_values.append(option_result['option_id'])
					else:
						failed_answers.append({
							"answer_id": answer_data['answer_id'],
							"error": f"Option '{answer_data['selected_option']}' not found for question_id {answer_data['question_id']}"
						})
						continue

				# Skip if nothing to update
				if not update_fields:
					continue

				# Add answer_id to values for WHERE clause
				update_values.append(answer_data['answer_id'])

				# Execute update query
				update_query = f"UPDATE answer SET {', '.join(update_fields)} WHERE answer_id = %s"
				cursor.execute(update_query, tuple(update_values))

				updated_answers.append(answer_data['answer_id'])

			except Exception as e:
				failed_answers.append({
					"answer_id": answer_data.get('answer_id', 'unknown'),
					"error": str(e)
				})

		# Commit changes
		conn.commit()
		cursor.close()
		conn.close()

		return jsonify({
			"success": True,
			"message": f"Updated {len(updated_answers)} answers successfully",
			"updated_answers": updated_answers,
			"failed_answers": failed_answers
		})

	except Exception as e:
		return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/answers/<int:answer_id>', methods=['DELETE'])
def delete_answer(answer_id):
	"""Delete a specific answer"""
	try:
		conn = get_db_connection()
		cursor = conn.cursor(dictionary=True)

		# Check if answer exists
		check_query = "SELECT COUNT(*) as count FROM answer WHERE answer_id = %s"
		cursor.execute(check_query, (answer_id,))
		if cursor.fetchone()['count'] == 0:
			cursor.close()
			conn.close()
			return jsonify({"success": False, "error": "Answer not found"}), 404

		# Delete answer
		delete_query = "DELETE FROM answer WHERE answer_id = %s"
		cursor.execute(delete_query, (answer_id,))
		conn.commit()

		cursor.close()
		conn.close()

		return jsonify({
			"success": True,
			"message": "Answer deleted successfully"
		})
	except Exception as e:
		return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/login', methods=['POST'])
def login():
	"""Authenticate a user by checking if username exists in respondent table"""
	try:
		data = request.json

		if not data or 'username' not in data:
			return jsonify({"success": False, "error": "Username is required"}), 400

		username = data['username']

		conn = get_db_connection()
		cursor = conn.cursor(dictionary=True)

		# Check if username exists in respondent table
		query = "SELECT respondent_id FROM respondent WHERE user_name = %s"
		cursor.execute(query, (username,))
		result = cursor.fetchone()

		cursor.close()
		conn.close()

		if result:
			return jsonify({
				"success": True,
				"message": "Login successful",
				"respondent_id": result['respondent_id']
			})
		else:
			return jsonify({
				"success": False,
				"message": "Invalid username"
			}), 401

	except Exception as e:
		return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/surveys/<int:survey_id>', methods=['GET'])
def get_survey_by_id(survey_id):
    """Get a specific survey by ID with all its questions and options"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Get survey details
        survey_query = "SELECT survey_id, title, description FROM survey WHERE survey_id = %s"
        cursor.execute(survey_query, (survey_id,))
        survey = cursor.fetchone()

        if not survey:
            cursor.close()
            conn.close()
            return jsonify({"success": False, "error": "Survey not found"}), 404

        # Get questions for this survey along with their types
        questions_query = """
        SELECT q.question_id, q.question_text, q.description, q.is_required, 
               q.sequence_number, q.type_id, qt.type_name
        FROM question q
        JOIN question_type qt ON q.type_id = qt.type_id
        WHERE q.survey_id = %s
        ORDER BY q.sequence_number
        """
        cursor.execute(questions_query, (survey_id,))
        questions = cursor.fetchall()

        # For each question, get options if it's single_choice or multiple_choice
        for question in questions:
            if question['type_name'] in ['single_choice', 'multiple_choice']:
                options_query = """
                SELECT option_id, option_text, sequence_number
                FROM question_option
                WHERE question_id = %s
                ORDER BY sequence_number
                """
                cursor.execute(options_query, (question['question_id'],))
                question['options'] = cursor.fetchall()
            else:
                question['options'] = []

        cursor.close()
        conn.close()

        return jsonify({
            "success": True,
            "data": {
                "survey": survey,
                "questions": questions
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/register', methods=['POST'])
def register():
	"""Register a new user by adding their details to the respondent table"""
	try:
		data = request.json

		if not data or 'username' not in data or 'password' not in data:
			return jsonify({"success": False, "error": "Username and password are required"}), 400

		username = data['username']
		# Note: In a real application, you would hash the password before storing it
		# This implementation doesn't actually store the password as the table doesn't have a password field

		# Generate random IP address
		import random
		ip_parts = [str(random.randint(1, 255)) for _ in range(4)]
		random_ip = '.'.join(ip_parts)

		# Generate email based on username
		email = f'user{username}@example.com'

		conn = get_db_connection()
		cursor = conn.cursor(dictionary=True)

		# Check if username already exists
		check_query = "SELECT COUNT(*) as count FROM respondent WHERE user_name = %s"
		cursor.execute(check_query, (username,))
		if cursor.fetchone()['count'] > 0:
			cursor.close()
			conn.close()
			return jsonify({"success": False, "error": "Username already exists"}), 409

		# Insert new respondent
		current_time = datetime.now()
		insert_query = """
        INSERT INTO respondent (respondent_id, user_name, email, ip_address, created_at, updated_at) 
        VALUES (%s, %s, %s, %s, %s, %s)
        """
		cursor.execute(insert_query, (
			username,  # Using username as respondent_id as requested
			username,
			email,
			random_ip,
			current_time,
			current_time
		))

		conn.commit()
		cursor.close()
		conn.close()

		return jsonify({
			"success": True,
			"message": "Registration successful",
			"respondent_id": username
		})

	except Exception as e:
		return jsonify({"success": False, "error": str(e)}), 500


# @app.route('/api/answers', methods=['POST'])
# def add_answers():
# 	"""Add answers to the database"""
# 	try:
# 		data = request.json
#
# 		if not data or 'answers' not in data or 'username' not in data:
# 			return jsonify({"success": False, "error": "Username and answers are required"}), 400
#
# 		username = data['username']
# 		answers = data['answers']
#
# 		if not isinstance(answers, list) or len(answers) == 0:
# 			return jsonify({"success": False, "error": "Answers must be a non-empty array"}), 400
#
# 		conn = get_db_connection()
# 		cursor = conn.cursor(dictionary=True)
#
# 		# Verify if username exists in respondent table
# 		check_query = "SELECT respondent_id FROM respondent WHERE user_name = %s"
# 		cursor.execute(check_query, (username,))
# 		result = cursor.fetchone()
#
# 		if not result:
# 			cursor.close()
# 			conn.close()
# 			return jsonify({"success": False, "error": "Username not found"}), 404
#
# 		respondent_id = result['respondent_id']
#
# 		# First, create a new response_record entry
# 		current_time = datetime.now()
#
# 		# Determine which survey_id to use from the first question
# 		# Assuming all questions in the payload belong to the same survey
# 		if answers and 'question_id' in answers[0]:
# 			survey_query = "SELECT survey_id FROM question WHERE question_id = %s"
# 			cursor.execute(survey_query, (answers[0]['question_id'],))
# 			survey_result = cursor.fetchone()
#
# 			if not survey_result:
# 				cursor.close()
# 				conn.close()
# 				return jsonify({"success": False, "error": f"Question ID {answers[0]['question_id']} not found"}), 404
#
# 			survey_id = survey_result['survey_id']
# 		else:
# 			cursor.close()
# 			conn.close()
# 			return jsonify({"success": False, "error": "No valid question in answers array"}), 400
#
# 		# Insert response record with all required fields
# 		insert_response_query = """
#         INSERT INTO response_record (
#             survey_id,
#             respondent_id,
#             start_time,
#             is_complete
#         ) VALUES (%s, %s, %s, %s)
#         """
#
# 		# Determine if the response is complete based on whether all questions have answers
# 		is_complete = False
# 		if len(answers) > 0:
# 			# Simple check - we'll consider it complete if there's at least one answer
# 			# A more complex implementation would check if all required questions are answered
# 			is_complete = True
#
# 		cursor.execute(insert_response_query, (
# 			survey_id,
# 			respondent_id,
# 			current_time,
# 			is_complete
# 		))
#
# 		response_id = cursor.lastrowid
#
# 		# If the response is complete, update the complete_time
# 		if is_complete:
# 			update_query = """
#             UPDATE response_record
#             SET complete_time = %s
#             WHERE response_id = %s
#             """
# 			cursor.execute(update_query, (current_time, response_id))
#
# 		conn.commit()
#
# 		# Track successfully added answers and failed ones
# 		added_answers = []
# 		failed_answers = []
#
# 		# Process each answer in the array
# 		for answer in answers:
# 			try:
# 				question_id = answer.get('question_id')
#
# 				# Skip if question_id is missing
# 				if not question_id:
# 					failed_answers.append({
# 						"answer": answer,
# 						"error": "Missing question_id"
# 					})
# 					continue
#
# 				# Handle text_answer if not empty
# 				if 'text_answer' in answer and answer['text_answer'] and answer['text_answer'] != "":
# 					insert_query = """
#                     INSERT INTO answer (
#                         response_id,
#                         question_id,
#                         text_answer
#                     ) VALUES (%s, %s, %s)
#                     """
# 					cursor.execute(insert_query, (
# 						response_id,
# 						question_id,
# 						answer['text_answer']
# 					))
# 					added_answers.append({
# 						"question_id": question_id,
# 						"type": "text_answer"
# 					})
#
# 				# Handle numerical_answer if not zero
# 				if 'numerical_answer' in answer and answer['numerical_answer'] and answer['numerical_answer'] != 0:
# 					insert_query = """
#                     INSERT INTO answer (
#                         response_id,
#                         question_id,
#                         numerical_answer
#                     ) VALUES (%s, %s, %s)
#                     """
# 					cursor.execute(insert_query, (
# 						response_id,
# 						question_id,
# 						answer['numerical_answer']
# 					))
# 					added_answers.append({
# 						"question_id": question_id,
# 						"type": "numerical_answer"
# 					})
#
# 				# Handle option_id if not empty array
# 				if 'option_id' in answer and answer['option_id'] and isinstance(answer['option_id'], list) and len(
# 						answer['option_id']) > 0:
# 					for option in answer['option_id']:
# 						# Verify option exists for this question
# 						option_check_query = """
#                         SELECT option_id FROM question_option
#                         WHERE question_id = %s AND option_id = %s
#                         """
# 						cursor.execute(option_check_query, (question_id, option))
# 						option_exists = cursor.fetchone()
#
# 						if not option_exists:
# 							failed_answers.append({
# 								"question_id": question_id,
# 								"option_id": option,
# 								"error": "Option does not exist for this question"
# 							})
# 							continue
#
# 						insert_query = """
#                         INSERT INTO answer (
#                             response_id,
#                             question_id,
#                             option_id
#                         ) VALUES (%s, %s, %s)
#                         """
# 						cursor.execute(insert_query, (
# 							response_id,
# 							question_id,
# 							option
# 						))
# 						added_answers.append({
# 							"question_id": question_id,
# 							"type": "option_id",
# 							"option": option
# 						})
#
# 			except Exception as e:
# 				failed_answers.append({
# 					"answer": answer,
# 					"error": str(e)
# 				})
#
# 		# Commit all changes
# 		conn.commit()
# 		cursor.close()
# 		conn.close()
#
# 		return jsonify({
# 			"success": True,
# 			"message": f"Successfully added {len(added_answers)} answers",
# 			"response_id": response_id,
# 			"added_answers": added_answers,
# 			"failed_answers": failed_answers
# 		})
#
# 	except Exception as e:
# 		return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/answers', methods=['POST'])
def add_answers():
	"""Add answers to the database using client-provided response_id"""
	try:
		data = request.json

		if not data or 'answers' not in data or 'username' not in data:
			return jsonify({"success": False, "error": "Username and answers are required"}), 400

		username = data['username']
		answers = data['answers']

		if not isinstance(answers, list) or len(answers) == 0:
			return jsonify({"success": False, "error": "Answers must be a non-empty array"}), 400

		conn = get_db_connection()
		cursor = conn.cursor(dictionary=True)

		# Verify if username exists in respondent table
		check_query = "SELECT respondent_id FROM respondent WHERE user_name = %s"
		cursor.execute(check_query, (username,))
		result = cursor.fetchone()

		if not result:
			cursor.close()
			conn.close()
			return jsonify({"success": False, "error": "Username not found"}), 404

		respondent_id = result['respondent_id']
		current_time = datetime.now()

		# Get response_id from first answer (assuming all answers have the same response_id)
		if answers and 'response_id' in answers[0]:
			response_id = answers[0]['response_id']
		else:
			cursor.close()
			conn.close()
			return jsonify({"success": False, "error": "No valid response_id in answers array"}), 400

		# Determine which survey_id to use from the first question
		if answers and 'question_id' in answers[0]:
			survey_query = "SELECT survey_id FROM question WHERE question_id = %s"
			cursor.execute(survey_query, (answers[0]['question_id'],))
			survey_result = cursor.fetchone()

			if not survey_result:
				cursor.close()
				conn.close()
				return jsonify({"success": False, "error": f"Question ID {answers[0]['question_id']} not found"}), 404

			survey_id = survey_result['survey_id']
		else:
			cursor.close()
			conn.close()
			return jsonify({"success": False, "error": "No valid question in answers array"}), 400

		# Check if response_record already exists
		check_response_query = "SELECT response_id FROM response_record WHERE response_id = %s"
		cursor.execute(check_response_query, (response_id,))
		existing_response = cursor.fetchone()

		# If response_record doesn't exist, create it
		if not existing_response:
			insert_response_query = """
            INSERT INTO response_record (
                response_id,
                survey_id, 
                respondent_id, 
                start_time, 
                is_complete
            ) VALUES (%s, %s, %s, %s, %s)
            """

			is_complete = len(answers) > 0

			cursor.execute(insert_response_query, (
				response_id,
				survey_id,
				respondent_id,
				current_time,
				is_complete
			))

			# If the response is complete, update the complete_time
			if is_complete:
				update_query = """
                UPDATE response_record 
                SET complete_time = %s 
                WHERE response_id = %s
                """
				cursor.execute(update_query, (current_time, response_id))

			conn.commit()

		# Track successfully added answers and failed ones
		added_answers = []
		failed_answers = []

		# Process each answer in the array
		for answer in answers:
			try:
				question_id = answer.get('question_id')
				answer_response_id = answer.get('response_id')

				# Skip if question_id or response_id is missing or response_ids don't match
				if not question_id or not answer_response_id or str(answer_response_id) != str(response_id):
					failed_answers.append({
						"answer": answer,
						"error": "Missing question_id, response_id, or response_id mismatch"
					})
					continue

				# Handle text_answer if not empty
				if 'text_answer' in answer and answer['text_answer'] and answer['text_answer'] != "":
					insert_query = """
                    INSERT INTO answer (
                        response_id, 
                        question_id, 
                        text_answer
                    ) VALUES (%s, %s, %s)
                    """
					cursor.execute(insert_query, (
						response_id,
						question_id,
						answer['text_answer']
					))
					added_answers.append({
						"question_id": question_id,
						"type": "text_answer"
					})

				# Handle numerical_answer if not zero
				if 'numerical_answer' in answer and answer['numerical_answer'] and answer['numerical_answer'] != 0:
					insert_query = """
                    INSERT INTO answer (
                        response_id, 
                        question_id, 
                        numerical_answer
                    ) VALUES (%s, %s, %s)
                    """
					cursor.execute(insert_query, (
						response_id,
						question_id,
						answer['numerical_answer']
					))
					added_answers.append({
						"question_id": question_id,
						"type": "numerical_answer"
					})

				# Handle option_id if not empty array
				if 'option_id' in answer and answer['option_id'] and isinstance(answer['option_id'], list) and len(
						answer['option_id']) > 0:
					for option in answer['option_id']:
						# Verify option exists for this question
						option_check_query = """
                        SELECT option_id FROM question_option 
                        WHERE question_id = %s AND option_id = %s
                        """
						cursor.execute(option_check_query, (question_id, option))
						option_exists = cursor.fetchone()

						if not option_exists:
							failed_answers.append({
								"question_id": question_id,
								"option_id": option,
								"error": "Option does not exist for this question"
							})
							continue

						insert_query = """
                        INSERT INTO answer (
                            response_id, 
                            question_id, 
                            option_id
                        ) VALUES (%s, %s, %s)
                        """
						cursor.execute(insert_query, (
							response_id,
							question_id,
							option
						))
						added_answers.append({
							"question_id": question_id,
							"type": "option_id",
							"option": option
						})

			except Exception as e:
				failed_answers.append({
					"answer": answer,
					"error": str(e)
				})

		# Commit all changes
		conn.commit()
		cursor.close()
		conn.close()

		return jsonify({
			"success": True,
			"message": f"Successfully added {len(added_answers)} answers",
			"response_id": response_id,
			"added_answers": added_answers,
			"failed_answers": failed_answers
		})

	except Exception as e:
		return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/responses', methods=['DELETE'])
def delete_response():
	"""Delete response data for a user including answers, response record, and respondent data"""
	try:
		data = request.json

		if not data or 'response_id' not in data or 'user_name' not in data:
			return jsonify({"success": False, "error": "Response ID and username are required"}), 400

		response_id = data['response_id']
		user_name = data['user_name']

		conn = get_db_connection()
		cursor = conn.cursor(dictionary=True)

		# Keep track of deleted records
		deleted_counts = {
			'answers': 0,
			'response_records': 0,
			'respondents': 0
		}

		# 1. Delete answers with the specific response_id
		delete_answers_query = "DELETE FROM answer WHERE response_id = %s"
		cursor.execute(delete_answers_query, (response_id,))
		deleted_counts['answers'] = cursor.rowcount

		# 2. Delete response_record entries where respondent_id = user_name
		delete_responses_query = "DELETE FROM response_record WHERE respondent_id = %s"
		cursor.execute(delete_responses_query, (user_name,))
		deleted_counts['response_records'] = cursor.rowcount

		# 3. Delete respondent entry with the specified user_name
		delete_respondent_query = "DELETE FROM respondent WHERE user_name = %s"
		cursor.execute(delete_respondent_query, (user_name,))
		deleted_counts['respondents'] = cursor.rowcount

		# Commit the transaction
		conn.commit()
		cursor.close()
		conn.close()

		return jsonify({
			"success": True,
			"message": "User data deleted successfully",
			"details": {
				"deleted_answers": deleted_counts['answers'],
				"deleted_response_records": deleted_counts['response_records'],
				"deleted_respondents": deleted_counts['respondents']
			}
		})

	except Exception as e:
		return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/survey-logic/<int:survey_id>', methods=['GET'])
def get_survey_logic(survey_id):
    """Get specific logic fields for a survey"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Check if survey exists
        survey_query = "SELECT COUNT(*) as count FROM survey WHERE survey_id = %s"
        cursor.execute(survey_query, (survey_id,))
        if cursor.fetchone()['count'] == 0:
            cursor.close()
            conn.close()
            return jsonify({"success": False, "error": "Survey not found"}), 404

        # Get only the requested fields from survey_logic
        logic_query = """
        SELECT 
            survey_id,
            question_id,
            option_id,
            action_type,
            target_question_id
        FROM survey_logic
        WHERE survey_id = %s
        """
        cursor.execute(logic_query, (survey_id,))
        logic_rules = cursor.fetchall()

        cursor.close()
        conn.close()

        return jsonify({
            "success": True,
            "data": logic_rules
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
	app.run(debug=True, port=5000)