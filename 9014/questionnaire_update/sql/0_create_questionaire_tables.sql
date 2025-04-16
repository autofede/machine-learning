drop database if exists 9014_db; 

create database 9014_db; 

use 9014_db;  

-- Survey Database Schema Design
-- This schema supports creating, managing, and analyzing user survey questionnaires

-- 1. Survey Table - Stores basic information about each survey
CREATE TABLE survey (
    survey_id INT PRIMARY KEY AUTO_INCREMENT COMMENT 'Unique survey identifier',
    title VARCHAR(255) NOT NULL COMMENT 'Survey title',
    description TEXT COMMENT 'Survey description and purpose',
    start_date DATETIME NOT NULL COMMENT 'Survey start date and time',
    end_date DATETIME COMMENT 'Survey end date and time (NULL if ongoing)',
    status ENUM('draft', 'active', 'paused', 'completed', 'archived') NOT NULL DEFAULT 'draft' COMMENT 'Current survey status',
    # created_by INT NOT NULL COMMENT 'User ID who created the survey',
    # created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    # updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update timestamp',
    # is_anonymous BOOLEAN NOT NULL DEFAULT FALSE COMMENT 'Whether responses are anonymous',
    # target_responses INT COMMENT 'Target number of responses',
    INDEX idx_status (status), # create index to improve search efficiency
    INDEX idx_dates (start_date, end_date)
) COMMENT 'Main survey information table';

-- 2. Question Type Table - Catalogs different question types
CREATE TABLE question_type (
    type_id INT PRIMARY KEY AUTO_INCREMENT COMMENT 'Question type identifier',
    type_name VARCHAR(50) NOT NULL UNIQUE COMMENT 'Question type name',
    description VARCHAR(255) COMMENT 'Description of this question type'
) COMMENT 'Catalog of available question types';

-- 3. Question Table - Stores individual questions within surveys
CREATE TABLE question (
    question_id INT PRIMARY KEY AUTO_INCREMENT COMMENT 'Unique question identifier',
    survey_id INT NOT NULL COMMENT 'Survey this question belongs to',
    type_id INT NOT NULL COMMENT 'Question type',
    question_text TEXT NOT NULL COMMENT 'The actual question text',
    description TEXT COMMENT 'Additional description or instructions',
    is_required BOOLEAN NOT NULL DEFAULT FALSE COMMENT 'Whether an answer is required',
    sequence_number INT NOT NULL COMMENT 'Display order within the survey',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update timestamp',
    max_length INT COMMENT 'Maximum length for text answers',
    validation_regex VARCHAR(255) COMMENT 'Optional regex for answer validation',
    # INDEX idx_survey_seq (survey_id, sequence_number),
    FOREIGN KEY (survey_id) REFERENCES survey(survey_id) ON DELETE CASCADE,
    FOREIGN KEY (type_id) REFERENCES question_type(type_id)
) COMMENT 'Survey questions table';

-- 4. Option Table - Stores predefined options for multiple choice questions
CREATE TABLE question_option (
    option_id INT PRIMARY KEY AUTO_INCREMENT COMMENT 'Option identifier',
    question_id INT NOT NULL COMMENT 'Question this option belongs to',
    option_text VARCHAR(255) NOT NULL COMMENT 'Option text',
    sequence_number INT NOT NULL COMMENT 'Display order of this option',
    is_other_option BOOLEAN NOT NULL DEFAULT FALSE COMMENT 'Whether this is an "Other" option that allows custom input',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update timestamp',
    #INDEX idx_question_seq (question_id, sequence_number),
    FOREIGN KEY (question_id) REFERENCES question(question_id) ON DELETE CASCADE
) COMMENT 'Options for multiple choice questions';

-- 5. Respondent Table - Stores information about survey respondents
CREATE TABLE respondent (
    respondent_id INT PRIMARY KEY AUTO_INCREMENT COMMENT 'Unique respondent identifier',
    user_name INT COMMENT 'User name if respondent is a registered user (NULL for anonymous/external)',
    email VARCHAR(255) COMMENT 'Email address (NULL for anonymous responses)',
    ip_address VARCHAR(45) COMMENT 'IP address of respondent',
    #user_agent TEXT COMMENT 'Browser/client information',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update timestamp'
    # INDEX idx_user_name (user_name),
    #INDEX idx_email (email)
) COMMENT 'Survey respondents information';

-- 6. Response Table - Stores completed survey submissions
CREATE TABLE response_record (
    response_id INT PRIMARY KEY AUTO_INCREMENT COMMENT 'Unique response identifier',
    survey_id INT NOT NULL COMMENT 'Survey being responded to',
    respondent_id INT NOT NULL COMMENT 'Respondent who completed this survey',
    start_time DATETIME NOT NULL COMMENT 'When respondent started the survey',
    complete_time DATETIME COMMENT 'When respondent completed the survey (NULL if incomplete)',
    is_complete BOOLEAN NOT NULL DEFAULT FALSE COMMENT 'Whether the survey was completed',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update timestamp',
    # INDEX idx_survey_respondent (survey_id, respondent_id),
    # INDEX idx_survey_complete (survey_id, is_complete),
    FOREIGN KEY (survey_id) REFERENCES survey(survey_id),
    FOREIGN KEY (respondent_id) REFERENCES respondent(respondent_id)
) COMMENT 'Survey response submissions';

-- 7. Answer Table - Stores individual question answers
CREATE TABLE answer (
    answer_id INT PRIMARY KEY AUTO_INCREMENT COMMENT 'Unique answer identifier',
    response_id INT NOT NULL COMMENT 'Response this answer belongs to',
    question_id INT NOT NULL COMMENT 'Question being answered',
    option_id INT COMMENT 'Selected option for multiple choice (NULL for text answers)',
    text_answer TEXT COMMENT 'Text answer (NULL for option-based answers)',
    numerical_answer DECIMAL(10,2) COMMENT 'Numerical answer value (for numerical questions)',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update timestamp',
    # INDEX idx_response_question (response_id, question_id),
    #  UNIQUE KEY unq_response_question (response_id, question_id),
    FOREIGN KEY (response_id) REFERENCES response_record(response_id) ON DELETE CASCADE,
    FOREIGN KEY (question_id) REFERENCES question(question_id),
    FOREIGN KEY (option_id) REFERENCES question_option(option_id)
) COMMENT 'Individual question answers';

-- 8. Survey Logic Table - Stores conditional logic and skip patterns
CREATE TABLE survey_logic (
    logic_id INT PRIMARY KEY AUTO_INCREMENT COMMENT 'Logic rule identifier',
    survey_id INT NOT NULL COMMENT 'Survey this logic belongs to',
    question_id INT NOT NULL COMMENT 'Question that triggers this logic',
    option_id INT COMMENT 'Option that triggers this logic (NULL if based on any answer)',
    action_type ENUM('show', 'hide', 'skip_to', 'end_survey') NOT NULL COMMENT 'Type of action to perform',
    target_question_id INT COMMENT 'Question affected by this logic rule',
    condition_expression VARCHAR(255) COMMENT 'Custom condition expression',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update timestamp',
    # INDEX idx_survey (survey_id),
    # INDEX idx_question (question_id),
    FOREIGN KEY (survey_id) REFERENCES survey(survey_id) ON DELETE CASCADE,
    FOREIGN KEY (question_id) REFERENCES question(question_id) ON DELETE CASCADE,
    FOREIGN KEY (option_id) REFERENCES question_option(option_id),
    FOREIGN KEY (target_question_id) REFERENCES question(question_id)
) COMMENT 'Conditional logic rules for dynamic surveys';

/*
-- 9. Survey Distribution Table - Tracks survey distribution channels
CREATE TABLE survey_distribution (
    distribution_id INT PRIMARY KEY AUTO_INCREMENT COMMENT 'Distribution identifier',
    survey_id INT NOT NULL COMMENT 'Survey being distributed',
    channel ENUM('email', 'link', 'embed', 'qr', 'social') NOT NULL COMMENT 'Distribution channel',
    unique_code VARCHAR(50) UNIQUE COMMENT 'Unique identification code for this distribution',
    name VARCHAR(100) NOT NULL COMMENT 'Name of this distribution (for tracking)',
    created_by INT NOT NULL COMMENT 'User who created this distribution',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update timestamp',
    expires_at DATETIME COMMENT 'When this distribution expires',
    max_responses INT COMMENT 'Maximum allowed responses for this distribution',
    current_responses INT NOT NULL DEFAULT 0 COMMENT 'Current response count',
    INDEX idx_survey (survey_id),
    INDEX idx_code (unique_code),
    FOREIGN KEY (survey_id) REFERENCES survey(survey_id) ON DELETE CASCADE
) COMMENT 'Survey distribution channels and tracking';
*/

/*
-- 10. Survey Template Tablequestion_type - Stores reusable survey templates
CREATE TABLE survey_template (
    template_id INT PRIMARY KEY AUTO_INCREMENT COMMENT 'Template identifier',
    title VARCHAR(255) NOT NULL COMMENT 'Template title',
    description TEXT COMMENT 'Template description',
    category VARCHAR(100) COMMENT 'Template category',
    is_public BOOLEAN NOT NULL DEFAULT FALSE COMMENT 'Whether this template is publicly available',
    created_by INT NOT NULL COMMENT 'User who created this template',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp',
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update timestamp',
    original_survey_id INT COMMENT 'Original survey this template was created from',
    times_used INT NOT NULL DEFAULT 0 COMMENT 'Number of times this template was used',
    INDEX idx_category (category),
    INDEX idx_public (is_public),
    FOREIGN KEY (original_survey_id) REFERENCES survey(survey_id) ON DELETE SET NULL
) COMMENT 'Reusable survey templates';
*/

-- Initial data for question types
INSERT INTO question_type (type_name, description) VALUES
('single_choice', 'Single choice from multiple options'),
('multiple_choice', 'Multiple selections from options'),
('text', 'Free text input'),
('textarea', 'Multi-line text input'),
('rating', 'Rating scale (e.g., 1-5 stars)'),
('likert', 'Likert scale (e.g., agreement scale)'),
('numeric', 'Numerical input only'),
('date', 'Date input'),
('time', 'Time input'),
('file', 'File upload'),
('matrix', 'Matrix of questions with the same response options'),
('ranking', 'Rank options in order of preference'),
('slider', 'Slider scale for numeric values'),
('email', 'Email address input'),
('phone', 'Phone number input');