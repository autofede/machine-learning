-- Survey Database Test Data Generation Script
-- This script generates test data for the survey database schema

USE 9014_db;

-- Clear existing data (if any)
-- SET FOREIGN_KEY_CHECKS = 0;
/*
TRUNCATE TABLE answer;
TRUNCATE TABLE survey_logic;
TRUNCATE TABLE question_option;
TRUNCATE TABLE response;
TRUNCATE TABLE respondent;
TRUNCATE TABLE question;
TRUNCATE TABLE survey_distribution;
TRUNCATE TABLE survey_template;
TRUNCATE TABLE survey;
SET FOREIGN_KEY_CHECKS = 1;
*/
-- Generate 2 surveys
INSERT INTO survey (title, description, start_date, end_date, status)
VALUES 
/*
('Customer Satisfaction Survey', 'Help us understand your experience with our products and services', 
 '2024-01-01 00:00:00', '2024-12-31 23:59:59', 'active', 1000),
('Employee Engagement Survey', 'Annual survey to measure employee satisfaction and engagement', 
 '2024-02-01 00:00:00', '2024-10-31 23:59:59', 'active', 1000);
 */
('Customer Satisfaction Survey', 'Help us understand your experience with our products and services', 
 '2024-01-01 00:00:00', '2024-12-31 23:59:59', 'active'),
('Employee Engagement Survey', 'Annual survey to measure employee satisfaction and engagement', 
 '2024-02-01 00:00:00', '2024-10-31 23:59:59', 'active');
 
-- Generate 50 questions for each survey (100 total)
-- First survey: Customer Satisfaction Survey (survey_id = 1)
INSERT INTO question (survey_id, type_id, question_text, description, is_required, sequence_number, max_length)
VALUES
-- Demographics
(1, 3, 'What is your age?', 'Please enter your age in years', TRUE, 1, 3),
(1, 1, 'What is your gender?', NULL, TRUE, 2, NULL),
(1, 1, 'What is your employment status?', NULL, TRUE, 3, NULL),
(1, 1, 'What is your highest level of education?', NULL, TRUE, 4, NULL),
(1, 1, 'How often do you use our products?', NULL, TRUE, 5, NULL),

-- Product Experience
(1, 5, 'How would you rate our product quality?', 'Please rate on a scale of 1-5', TRUE, 6, NULL),
(1, 5, 'How would you rate the value for money of our products?', 'Please rate on a scale of 1-5', TRUE, 7, NULL),
(1, 5, 'How would you rate the usability of our products?', 'Please rate on a scale of 1-5', TRUE, 8, NULL),
(1, 5, 'How would you rate our product design?', 'Please rate on a scale of 1-5', TRUE, 9, NULL),
(1, 5, 'How would you rate our product performance?', 'Please rate on a scale of 1-5', TRUE, 10, NULL),

-- Customer Service
(1, 5, 'How would you rate our customer service?', 'Please rate on a scale of 1-5', TRUE, 11, NULL),
(1, 5, 'How would you rate the response time of our customer service?', 'Please rate on a scale of 1-5', TRUE, 12, NULL),
(1, 5, 'How would you rate the knowledge of our customer service representatives?', 'Please rate on a scale of 1-5', TRUE, 13, NULL),
(1, 5, 'How would you rate the politeness of our customer service representatives?', 'Please rate on a scale of 1-5', TRUE, 14, NULL),
(1, 5, 'How would you rate the resolution of your issues?', 'Please rate on a scale of 1-5', TRUE, 15, NULL),

-- Website Experience
(1, 5, 'How would you rate our website usability?', 'Please rate on a scale of 1-5', TRUE, 16, NULL),
(1, 5, 'How would you rate our website design?', 'Please rate on a scale of 1-5', TRUE, 17, NULL),
(1, 5, 'How would you rate our website load time?', 'Please rate on a scale of 1-5', TRUE, 18, NULL),
(1, 5, 'How would you rate the ease of finding information on our website?', 'Please rate on a scale of 1-5', TRUE, 19, NULL),
(1, 5, 'How would you rate the checkout process on our website?', 'Please rate on a scale of 1-5', TRUE, 20, NULL),

-- Product Preferences
(1, 2, 'Which features do you value most in our products?', 'Please select all that apply', TRUE, 21, NULL),
(1, 3, 'Which of our products do you currently use?', 'Please select all that apply', TRUE, 22, NULL),
(1, 3, 'Which of our products do you use most frequently?', NULL, TRUE, 23, NULL),
(1, 3, 'Which product category are you most interested in?', NULL, TRUE, 24, NULL),
(1, 3, 'What improvements would you like to see in our products?', 'Please select all that apply', TRUE, 25, NULL),

-- Recommendations
(1, 6, 'I would recommend our products to friends and family', 'Please indicate your level of agreement', TRUE, 26, NULL),
(1, 6, 'I am likely to purchase our products again', 'Please indicate your level of agreement', TRUE, 27, NULL),
(1, 6, 'I prefer our products over competitors', 'Please indicate your level of agreement', TRUE, 28, NULL),
(1, 6, 'I am satisfied with the overall quality of our products', 'Please indicate your level of agreement', TRUE, 29, NULL),
(1, 6, 'I feel that our products offer good value for money', 'Please indicate your level of agreement', TRUE, 30, NULL),

-- Competitors
(1, 3, 'Which of our competitors\' products do you also use?', 'Please select all that apply', FALSE, 31, NULL),
(1, 3, 'Which competitor do you prefer?', NULL, FALSE, 32, NULL),
(1, 6, 'Our products are better than competitors\' products', 'Please indicate your level of agreement', FALSE, 33, NULL),
(1, 6, 'Our products are priced competitively', 'Please indicate your level of agreement', FALSE, 34, NULL),
(1, 6, 'Our products offer unique features compared to competitors', 'Please indicate your level of agreement', FALSE, 35, NULL),

-- Marketing
(1, 3, 'How did you hear about us?', NULL, TRUE, 36, NULL),
(1, 5, 'How would you rate our marketing materials?', 'Please rate on a scale of 1-5', FALSE, 37, NULL),
(1, 5, 'How would you rate our social media presence?', 'Please rate on a scale of 1-5', FALSE, 38, NULL),
(1, 5, 'How would you rate our email communications?', 'Please rate on a scale of 1-5', FALSE, 39, NULL),
(1, 5, 'How would you rate our promotional offers?', 'Please rate on a scale of 1-5', FALSE, 40, NULL),

-- Feedback
(1, 4, 'What do you like most about our products?', NULL, FALSE, 41, 500),
(1, 4, 'What do you dislike most about our products?', NULL, FALSE, 42, 500),
(1, 4, 'What improvements would you suggest for our products?', NULL, FALSE, 43, 500),
(1, 4, 'What additional products would you like us to offer?', NULL, FALSE, 44, 500),
(1, 4, 'Any other comments or suggestions?', NULL, FALSE, 45, 500),

-- Net Promoter Score
(1, 7, 'On a scale of 0-10, how likely are you to recommend our company to a friend or colleague?', 'Please enter a number between 0 and 10', TRUE, 46, NULL),
(1, 3, 'If you rated us less than 9, what would it take to raise your rating?', NULL, FALSE, 47, NULL),
(1, 3, 'If you rated us 9 or 10, what specifically made you give this rating?', NULL, FALSE, 48, NULL),
(1, 4, 'Please elaborate on your rating', NULL, FALSE, 49, 500),
(1, 14, 'If you would like to be contacted about your feedback, please provide your email', NULL, FALSE, 50, 100);

-- Second survey: Employee Engagement Survey (survey_id = 2)
INSERT INTO question (survey_id, type_id, question_text, description, is_required, sequence_number, max_length)
VALUES
-- Demographics
(2, 3, 'How many years have you worked at the company?', 'Please enter the number of years', TRUE, 1, 2),
(2, 1, 'What department do you work in?', NULL, TRUE, 2, NULL),
(2, 1, 'What is your job level?', NULL, TRUE, 3, NULL),
(2, 1, 'What is your employment type?', NULL, TRUE, 4, NULL),
(2, 1, 'What is your work location?', NULL, TRUE, 5, NULL),

-- Job Satisfaction
(2, 5, 'How satisfied are you with your current role?', 'Please rate on a scale of 1-5', TRUE, 6, NULL),
(2, 5, 'How satisfied are you with your work-life balance?', 'Please rate on a scale of 1-5', TRUE, 7, NULL),
(2, 5, 'How satisfied are you with your compensation?', 'Please rate on a scale of 1-5', TRUE, 8, NULL),
(2, 5, 'How satisfied are you with your benefits?', 'Please rate on a scale of 1-5', TRUE, 9, NULL),
(2, 5, 'How satisfied are you with your career growth opportunities?', 'Please rate on a scale of 1-5', TRUE, 10, NULL),

-- Management and Leadership
(2, 5, 'How would you rate your manager\'s leadership abilities?', 'Please rate on a scale of 1-5', TRUE, 11, NULL),
(2, 5, 'How would you rate the communication from senior leadership?', 'Please rate on a scale of 1-5', TRUE, 12, NULL),
(2, 5, 'How would you rate the feedback you receive from your manager?', 'Please rate on a scale of 1-5', TRUE, 13, NULL),
(2, 5, 'How would you rate the recognition you receive for your work?', 'Please rate on a scale of 1-5', TRUE, 14, NULL),
(2, 5, 'How would you rate the company\'s vision and direction?', 'Please rate on a scale of 1-5', TRUE, 15, NULL),

-- Work Environment
(2, 5, 'How would you rate the company culture?', 'Please rate on a scale of 1-5', TRUE, 16, NULL),
(2, 5, 'How would you rate the work environment?', 'Please rate on a scale of 1-5', TRUE, 17, NULL),
(2, 5, 'How would you rate the collaboration within your team?', 'Please rate on a scale of 1-5', TRUE, 18, NULL),
(2, 5, 'How would you rate the collaboration between departments?', 'Please rate on a scale of 1-5', TRUE, 19, NULL),
(2, 5, 'How would you rate the tools and resources available to you?', 'Please rate on a scale of 1-5', TRUE, 20, NULL),

-- Professional Development
(2, 5, 'How would you rate the training opportunities provided by the company?', 'Please rate on a scale of 1-5', TRUE, 21, NULL),
(2, 5, 'How would you rate the mentoring opportunities provided by the company?', 'Please rate on a scale of 1-5', TRUE, 22, NULL),
(2, 5, 'How would you rate the clarity of your career path?', 'Please rate on a scale of 1-5', TRUE, 23, NULL),
(2, 5, 'How would you rate your opportunities to utilize your skills and abilities?', 'Please rate on a scale of 1-5', TRUE, 24, NULL),
(2, 5, 'How would you rate the company\'s support for your professional development?', 'Please rate on a scale of 1-5', TRUE, 25, NULL),

-- Engagement
(2, 6, 'I feel proud to work for this company', 'Please indicate your level of agreement', TRUE, 26, NULL),
(2, 6, 'I would recommend this company as a great place to work', 'Please indicate your level of agreement', TRUE, 27, NULL),
(2, 6, 'I see myself working here in two years', 'Please indicate your level of agreement', TRUE, 28, NULL),
(2, 6, 'I feel motivated to go beyond what is expected of me', 'Please indicate your level of agreement', TRUE, 29, NULL),
(2, 6, 'I find my work meaningful and purposeful', 'Please indicate your level of agreement', TRUE, 30, NULL),

-- Diversity and Inclusion
(2, 6, 'The company values diversity', 'Please indicate your level of agreement', TRUE, 31, NULL),
(2, 6, 'I feel included and accepted at work', 'Please indicate your level of agreement', TRUE, 32, NULL),
(2, 6, 'My opinions are valued', 'Please indicate your level of agreement', TRUE, 33, NULL),
(2, 6, 'The company promotes a supportive and inclusive environment', 'Please indicate your level of agreement', TRUE, 34, NULL),
(2, 6, 'All employees are treated fairly regardless of their background', 'Please indicate your level of agreement', TRUE, 35, NULL),

-- Communication
(2, 6, 'Communication within my team is effective', 'Please indicate your level of agreement', TRUE, 36, NULL),
(2, 6, 'Communication between departments is effective', 'Please indicate your level of agreement', TRUE, 37, NULL),
(2, 6, 'I receive clear direction on my work priorities', 'Please indicate your level of agreement', TRUE, 38, NULL),
(2, 6, 'I am kept informed about company changes that affect me', 'Please indicate your level of agreement', TRUE, 39, NULL),
(2, 6, 'I feel comfortable sharing my ideas and opinions', 'Please indicate your level of agreement', TRUE, 40, NULL),

-- Feedback
(2, 4, 'What do you like most about working here?', NULL, FALSE, 41, 500),
(2, 4, 'What do you dislike most about working here?', NULL, FALSE, 42, 500),
(2, 4, 'What suggestions do you have for improving the company?', NULL, FALSE, 43, 500),
(2, 4, 'What additional benefits would you like to see offered?', NULL, FALSE, 44, 500),
(2, 4, 'Any other comments or suggestions?', NULL, FALSE, 45, 500),

-- Well-being
(2, 5, 'How would you rate your current stress level at work?', 'Please rate on a scale of 1-5 (1 being low, 5 being high)', TRUE, 46, NULL),
(2, 5, 'How would you rate the company\'s efforts to support employee well-being?', 'Please rate on a scale of 1-5', TRUE, 47, NULL),
(2, 5, 'How would you rate your work-life balance?', 'Please rate on a scale of 1-5', TRUE, 48, NULL),
(2, 5, 'How would you rate your overall job satisfaction?', 'Please rate on a scale of 1-5', TRUE, 49, NULL),
(2, 5, 'How would you rate your likelihood to stay with the company for the next year?', 'Please rate on a scale of 1-5', TRUE, 50, NULL);

update question set type_id=3 where type_id >5;

-- Generate question options for multiple-choice questions
-- Gender options (Question ID 2)
INSERT INTO question_option (question_id, option_text, sequence_number, is_other_option)
VALUES
(2, 'Male', 1, FALSE),
(2, 'Female', 2, FALSE);
/*
(2, 'Non-binary', 3, FALSE),
(2, 'Prefer not to say', 4, FALSE),
(2, 'Other', 5, TRUE);
*/

-- Employment status options (Question ID 3)
INSERT INTO question_option (question_id, option_text, sequence_number, is_other_option)
VALUES
(3, 'Full-time employed', 1, FALSE),
(3, 'Part-time employed', 2, FALSE),
(3, 'Self-employed', 3, FALSE),
(3, 'Student', 4, FALSE),
(3, 'Retired', 5, FALSE),
(3, 'Unemployed', 6, FALSE),
(3, 'Other', 7, TRUE);

-- Education options (Question ID 4)
INSERT INTO question_option (question_id, option_text, sequence_number, is_other_option)
VALUES
(4, 'High school or equivalent', 1, FALSE),
(4, 'Some college', 2, FALSE),
(4, 'Associate degree', 3, FALSE),
(4, 'Bachelor\'s degree', 4, FALSE),
(4, 'Master\'s degree', 5, FALSE),
(4, 'Doctoral degree', 6, FALSE),
(4, 'Professional degree', 7, FALSE),
(4, 'Other', 8, TRUE);

-- Product usage frequency options (Question ID 5)
INSERT INTO question_option (question_id, option_text, sequence_number, is_other_option)
VALUES
(5, 'Daily', 1, FALSE),
(5, 'Several times a week', 2, FALSE),
(5, 'Once a week', 3, FALSE),
(5, 'Several times a month', 4, FALSE),
(5, 'Once a month', 5, FALSE),
(5, 'Less than once a month', 6, FALSE),
(5, 'This is my first time', 7, FALSE);

-- Product features options (Question ID 21)
INSERT INTO question_option (question_id, option_text, sequence_number, is_other_option)
VALUES
(21, 'Quality', 1, FALSE),
(21, 'Price', 2, FALSE),
(21, 'Durability', 3, FALSE),
(21, 'Design', 4, FALSE),
(21, 'Functionality', 5, FALSE),
(21, 'Customer support', 6, FALSE),
(21, 'Warranty', 7, FALSE),
(21, 'Ease of use', 8, FALSE),
(21, 'Other', 9, TRUE);

-- Department options (Question ID 52)
INSERT INTO question_option (question_id, option_text, sequence_number, is_other_option)
VALUES
(52, 'Sales', 1, FALSE),
(52, 'Marketing', 2, FALSE),
(52, 'Product Development', 3, FALSE),
(52, 'IT', 4, FALSE),
(52, 'Human Resources', 5, FALSE),
(52, 'Finance', 6, FALSE),
(52, 'Customer Support', 7, FALSE),
(52, 'Operations', 8, FALSE),
(52, 'R&D', 9, FALSE),
(52, 'Management', 10, FALSE),
(52, 'Other', 11, TRUE);

-- Job level options (Question ID 53)
INSERT INTO question_option (question_id, option_text, sequence_number, is_other_option)
VALUES
(53, 'Entry level', 1, FALSE),
(53, 'Junior', 2, FALSE),
(53, 'Mid-level', 3, FALSE),
(53, 'Senior', 4, FALSE),
(53, 'Manager', 5, FALSE),
(53, 'Director', 6, FALSE),
(53, 'Executive', 7, FALSE),
(53, 'Other', 8, TRUE);

-- Employment type options (Question ID 54)
INSERT INTO question_option (question_id, option_text, sequence_number, is_other_option)
VALUES
(54, 'Full-time', 1, FALSE),
(54, 'Part-time', 2, FALSE),
(54, 'Contract', 3, FALSE),
(54, 'Temporary', 4, FALSE),
(54, 'Intern', 5, FALSE),
(54, 'Other', 6, TRUE);

-- Work location options (Question ID 55)
INSERT INTO question_option (question_id, option_text, sequence_number, is_other_option)
VALUES
(55, 'Headquarters', 1, FALSE),
(55, 'Branch office', 2, FALSE),
(55, 'Remote or Work from home', 3, FALSE),
(55, 'Hybrid', 4, FALSE),
(55, 'Other', 5, TRUE);


-- Create survey logic
-- Skip logic for Customer Satisfaction Survey
INSERT INTO survey_logic (survey_id, question_id, option_id, action_type, target_question_id)
VALUES
-- add survey logic for survey1
-- 1. Skip in-depth product usage questions for first-time users
-- If respondent selects "This is my first time" for product usage frequency, skip to question 30
(1, 5, 24, 'skip_to', 30),

-- add survey logic for survey2
-- 1. Skip workplace environment questions for remote workers
-- If work location is "Remote/Work from home", skip to remote work experience questions
(2, 55, 61, 'skip_to', 65);


/*
-- Create survey templates
INSERT INTO survey_template (title, description, category, is_public, created_by, original_survey_id, times_used)
VALUES
('Basic Customer Satisfaction', 'A simple template for measuring customer satisfaction', 'Customer Feedback', TRUE, 1, 1, 12),
('Comprehensive Employee Engagement', 'A detailed template for measuring employee engagement', 'Employee Feedback', TRUE, 1, 2, 8),
('Product Feedback', 'Template for gathering feedback on specific products', 'Product Development', TRUE, 1, 1, 5);
*/

/*
-- Create survey distributions
INSERT INTO survey_distribution (survey_id, channel, unique_code, name, created_by, expires_at, max_responses, current_responses)
VALUES
(1, 'email', 'CSAT2024', 'Customer Satisfaction 2024 Email Campaign', 1, '2024-12-31 23:59:59', 1000, 0),
(1, 'link', 'CSATLINK', 'Customer Satisfaction 2024 Website Link', 1, '2024-12-31 23:59:59', 1000, 0),
(2, 'email', 'EMP2024', 'Employee Engagement 2024 Email Campaign', 1, '2024-10-31 23:59:59', 1000, 0),
(2, 'qr', 'EMPQR', 'Employee Engagement 2024 QR Code', 1, '2024-10-31 23:59:59', 1000, 0);
*/
