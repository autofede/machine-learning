-- Generate responses for Survey 1 (Customer Satisfaction)
DELIMITER //
CREATE PROCEDURE generate_survey1_responses()
BEGIN
    DECLARE i INT DEFAULT 1;
    DECLARE response_id_var INT;
    DECLARE start_time_var DATETIME;
    DECLARE complete_time_var DATETIME;
    DECLARE rating_var INT;
    DECLARE nps_score_var INT;
    DECLARE gender_option INT;
    DECLARE employment_option INT;
    DECLARE education_option INT;
    DECLARE usage_option INT;
    
    WHILE i <= 1000 DO
        -- Generate response times
        SET start_time_var = DATE_ADD('2024-01-01 00:00:00', INTERVAL FLOOR(RAND() * 90) DAY);
        SET complete_time_var = DATE_ADD(start_time_var, INTERVAL FLOOR(RAND() * 30) MINUTE);
        
        -- Create response record
        INSERT INTO response_record (survey_id, respondent_id, start_time, complete_time, is_complete)
        VALUES (1, i, start_time_var, complete_time_var, TRUE);
        
        -- Get the response ID
        SET response_id_var = LAST_INSERT_ID();
        
        -- Demographics
        -- Age
        INSERT INTO answer (response_id, question_id, text_answer)
        VALUES (response_id_var, 1, 18 + FLOOR(RAND() * 70));
        
        -- Gender
        -- SET gender_option = 1 + FLOOR(RAND() * 4);
		SET gender_option = 1 + FLOOR(RAND() * 2);
        INSERT INTO answer (response_id, question_id, option_id)
        VALUES (response_id_var, 2, gender_option);
        
        -- Employment status - with logic
        SET employment_option = 1 + FLOOR(RAND() * 6);
        INSERT INTO answer (response_id, question_id, option_id)
        VALUES (response_id_var, 3, employment_option);
        
        -- Education - some logic may depend on this
        SET education_option = 1 + FLOOR(RAND() * 7);
        INSERT INTO answer (response_id, question_id, option_id)
        VALUES (response_id_var, 4, education_option);
        
        -- Product usage - with logic
        SET usage_option = 1 + FLOOR(RAND() * 7);
        INSERT INTO answer (response_id, question_id, option_id)
        VALUES (response_id_var, 5, usage_option);
        
        -- Apply skip logic for Survey 1
        
        -- Skip logic 1: If respondent selects "This is my first time" for product usage frequency (option_id = 24), skip to question 30
        IF usage_option != 24 THEN
            -- Product ratings (questions 6-20)
            -- Skip ratings for users who haven't used the product (usage_option = 1 means "Never used")
            IF usage_option != 1 THEN
                SET rating_var = 1 + FLOOR(RAND() * 5);
                
                INSERT INTO answer (response_id, question_id, numerical_answer)
                VALUES 
                (response_id_var, 6, rating_var),
                (response_id_var, 7, rating_var),
                (response_id_var, 8, rating_var),
                (response_id_var, 9, rating_var),
                (response_id_var, 10, rating_var),
                (response_id_var, 11, rating_var),
                (response_id_var, 12, rating_var),
                (response_id_var, 13, rating_var),
                (response_id_var, 14, rating_var),
                (response_id_var, 15, rating_var),
                (response_id_var, 16, rating_var),
                (response_id_var, 17, rating_var),
                (response_id_var, 18, rating_var),
                (response_id_var, 19, rating_var),
                (response_id_var, 20, rating_var);
                
                -- Product features (multi-select) - only for those who used the product
                INSERT INTO answer (response_id, question_id, option_id)
                VALUES 
                (response_id_var, 21, 1 + FLOOR(RAND() * 8));
           --  END IF;
--             
--             -- Skip questions 22-30 for respondents with education_option = 1 (No formal education)
--             IF education_option != 1 THEN
                INSERT INTO answer (response_id, question_id, text_answer)
                VALUES 
                (response_id_var, 22, CONCAT('Answer for question 22 from respondent ', i)),
                (response_id_var, 23, CONCAT('Answer for question 23 from respondent ', i)),
                (response_id_var, 24, CONCAT('Answer for question 24 from respondent ', i)),
                (response_id_var, 25, CONCAT('Answer for question 25 from respondent ', i)),
                (response_id_var, 26, CONCAT('Answer for question 26 from respondent ', i)),
                (response_id_var, 27, CONCAT('Answer for question 27 from respondent ', i)),
                (response_id_var, 28, CONCAT('Answer for question 28 from respondent ', i)),
                (response_id_var, 29, CONCAT('Answer for question 29 from respondent ', i));
            END IF;
        END IF;
        
        -- Question 30 (everyone answers this, even those who skipped from Q5)
        INSERT INTO answer (response_id, question_id, text_answer)
        VALUES (response_id_var, 30, CONCAT('Answer for question 30 from respondent ', i));
        
        -- Questions 31-40 for all respondents
        INSERT INTO answer (response_id, question_id, text_answer)
        VALUES 
        (response_id_var, 31, CONCAT('Answer for question 31 from respondent ', i)),
        (response_id_var, 32, CONCAT('Answer for question 32 from respondent ', i)),
        (response_id_var, 33, CONCAT('Answer for question 33 from respondent ', i)),
        (response_id_var, 34, CONCAT('Answer for question 34 from respondent ', i)),
        (response_id_var, 35, CONCAT('Answer for question 35 from respondent ', i)),
        (response_id_var, 36, CONCAT('Answer for question 36 from respondent ', i)),
        (response_id_var, 37, CONCAT('Answer for question 37 from respondent ', i)),
        (response_id_var, 38, CONCAT('Answer for question 38 from respondent ', i)),
        (response_id_var, 39, CONCAT('Answer for question 39 from respondent ', i)),
        (response_id_var, 40, CONCAT('Answer for question 40 from respondent ', i));
        
		INSERT INTO answer (response_id, question_id, text_answer)
		VALUES 
		(response_id_var, 41, CONCAT('I like the quality and reliability of your products. They consistently meet my expectations and the customer service is excellent.')),
		(response_id_var, 42, CONCAT('Sometimes the products are a bit expensive compared to alternatives, but the quality makes up for it.')),
		(response_id_var, 43, CONCAT('I would suggest improving the mobile app experience and adding more customization options.')),
		(response_id_var, 44, CONCAT('I would like to see more eco-friendly options and packaging in your product line.')),
		(response_id_var, 45, CONCAT('Overall great experience with your company. Looking forward to seeing new products in the future!'));
		
		-- NPS score (question 46) only for respondents who used the product
	-- 	IF usage_option != 1 THEN
-- 			SET nps_score_var = FLOOR(RAND() * 11);
-- 			INSERT INTO answer (response_id, question_id, numerical_answer)
-- 			VALUES (response_id_var, 46, nps_score_var);
-- 			
-- 			-- Follow-up based on NPS score
-- 			IF nps_score_var < 9 THEN
-- 				INSERT INTO answer (response_id, question_id, text_answer)
-- 				VALUES (response_id_var, 47, 'I would need to see better customer service and faster resolution of issues.');
-- 			ELSE
-- 				INSERT INTO answer (response_id, question_id, text_answer)
-- 				VALUES (response_id_var, 48, 'Your exceptional product quality and customer service made me give this high rating.');
-- 			END IF;
-- 		END IF;
        
		SET nps_score_var = FLOOR(RAND() * 11);
		INSERT INTO answer (response_id, question_id, numerical_answer) VALUES (response_id_var, 46, nps_score_var);
		
		INSERT INTO answer (response_id, question_id, text_answer)
		VALUES (response_id_var, 47, 'I would need to see better customer service and faster resolution of issues.');
	
		INSERT INTO answer (response_id, question_id, text_answer)
		VALUES (response_id_var, 48, 'Your exceptional product quality and customer service made me give this high rating.');

		-- Question 49 for those who didn't skip
		INSERT INTO answer (response_id, question_id, text_answer)
		VALUES (response_id_var, 49, CONCAT('Additional feedback about products from respondent ', i));
	
	-- Last question (50) for everyone
	INSERT INTO answer (response_id, question_id, text_answer)
	VALUES (response_id_var, 50, CONCAT('Suggestions for improvement from respondent ', i));

        SET i = i + 1;
    END WHILE;
END //
DELIMITER ;

-- Execute the stored procedure
CALL generate_survey1_responses();
-- Delete the stored procedure
DROP PROCEDURE generate_survey1_responses;

-- Generate responses for Survey 2 (Product Feedback)
DELIMITER //
CREATE PROCEDURE generate_survey2_responses()
BEGIN
    DECLARE i INT DEFAULT 1;  -- Start from 1 (not 1001)
    DECLARE response_id_var INT;
    DECLARE start_time_var DATETIME;
    DECLARE complete_time_var DATETIME;
    DECLARE rating_var INT;
    DECLARE satisfaction_var INT;
    DECLARE option1_var INT;
    DECLARE option2_var INT;
    DECLARE age_var INT;
    DECLARE gender_option INT;
    DECLARE location_option INT;
    DECLARE income_option INT;
    DECLARE purchase_option INT;
    DECLARE job_level_option INT;
    DECLARE feature_pref_option INT;
    DECLARE recommendation_var INT;
    
    WHILE i <= 1000 DO
        -- Generate response times
        SET start_time_var = DATE_ADD('2024-01-01 00:00:00', INTERVAL FLOOR(RAND() * 90) DAY);
        SET complete_time_var = DATE_ADD(start_time_var, INTERVAL FLOOR(RAND() * 30) MINUTE);
        
        -- Create response record with respondent_id starting from 1001
        INSERT INTO response_record (survey_id, respondent_id, start_time, complete_time, is_complete)
        VALUES (2, i + 1000, start_time_var, complete_time_var, TRUE);  -- Add 1000 to get respondent IDs 1001-2000
        
        -- Get the response ID
        SET response_id_var = LAST_INSERT_ID();
        
        -- Demographics
        -- Age
        SET age_var = 18 + FLOOR(RAND() * 70);
        INSERT INTO answer (response_id, question_id, text_answer)
        VALUES (response_id_var, 51, age_var);
        
        -- Gender
        SET gender_option = 1 + FLOOR(RAND() * 4);
        INSERT INTO answer (response_id, question_id, option_id)
        VALUES (response_id_var, 52, gender_option);
        
        -- Job level (question 53) - with logic
        SET job_level_option = 1 + FLOOR(RAND() * 5);
        INSERT INTO answer (response_id, question_id, option_id)
        VALUES (response_id_var, 53, job_level_option);
        
        -- Income bracket
        SET income_option = 1 + FLOOR(RAND() * 5);
        INSERT INTO answer (response_id, question_id, option_id)
        VALUES (response_id_var, 54, income_option);
        
        -- Work location (question 55) - with logic
        SET location_option = 1 + FLOOR(RAND() * 5);
        INSERT INTO answer (response_id, question_id, option_id)
        VALUES (response_id_var, 55, location_option);
        
        -- Apply skip logic for Survey 2
        
        -- Skip logic 1: If work location is "Remote/Work from home" (option_id = 61), skip to question 65
        IF location_option != 61 THEN
            -- Product satisfaction ratings (questions 56-64)
            SET rating_var = 1 + FLOOR(RAND() * 5);
            
            INSERT INTO answer (response_id, question_id, numerical_answer)
            VALUES 
            (response_id_var, 56, rating_var),
            (response_id_var, 57, rating_var),
            (response_id_var, 58, rating_var),
            (response_id_var, 59, rating_var),
            (response_id_var, 60, rating_var),
            (response_id_var, 61, rating_var),
            (response_id_var, 62, rating_var),
            (response_id_var, 63, rating_var),
            (response_id_var, 64, rating_var);
        END IF;
        
        -- Question 65 (everyone answers this, even those who skipped from Q55)
        INSERT INTO answer (response_id, question_id, numerical_answer)
        VALUES (response_id_var, 65, 1 + FLOOR(RAND() * 5));
        
        -- Product preferences (multi-select) - for everyone
        SET option1_var = 1 + FLOOR(RAND() * 8);
        SET option2_var = 1 + FLOOR(RAND() * 8);
        
        -- Make sure they are different options
        WHILE option2_var = option1_var DO
            SET option2_var = 1 + FLOOR(RAND() * 8);
        END WHILE;
        
        INSERT INTO answer (response_id, question_id, option_id)
        VALUES 
        (response_id_var, 66, option1_var),
        (response_id_var, 67, option2_var);
        
        -- Free-form feedback (questions 68-71)
        INSERT INTO answer (response_id, question_id, text_answer)
        VALUES 
        (response_id_var, 68, CONCAT('Product feedback from respondent ', i + 1000)),
        (response_id_var, 69, CONCAT('Service feedback from respondent ', i + 1000)),
        (response_id_var, 70, CONCAT('Website feedback from respondent ', i + 1000)),
        (response_id_var, 71, CONCAT('Mobile app feedback from respondent ', i + 1000));
        
        -- Conditionally add feedback based on job level
--         IF job_level_option = 1 THEN
            -- Entry level feedback (questions 72-74)
		INSERT INTO answer (response_id, question_id, text_answer)
		VALUES 
		(response_id_var, 72, CONCAT('I enjoyed using your product because it solves my problem efficiently and has an intuitive interface.')),
		(response_id_var, 73, CONCAT('The delivery service could be improved by providing more accurate tracking information.')),
		(response_id_var, 74, CONCAT('I would recommend adding more payment options and improving the checkout process.'));
   --  ELSE
            -- Management feedback (questions 72-74) - only for non-entry level
		INSERT INTO answer (response_id, question_id, text_answer)
		VALUES 
		(response_id_var, 72, CONCAT('Management feedback from higher-level respondent ', i + 1000)),
		(response_id_var, 73, CONCAT('Leadership assessment from higher-level respondent ', i + 1000)),
		(response_id_var, 74, CONCAT('Organizational structure feedback from higher-level respondent ', i + 1000));
   --  END IF;
        
        -- Questions 75-76 for everyone (after potential skip)
        INSERT INTO answer (response_id, question_id, text_answer)
        VALUES 
        (response_id_var, 75, CONCAT('I found your customer service team to be very responsive and helpful when I had questions.')),
        (response_id_var, 76, CONCAT('Overall, I am satisfied with my purchase experience and would buy from you again.'));
        
        -- Overall satisfaction (question 77) - for everyone
        SET satisfaction_var = 1 + FLOOR(RAND() * 10);
        INSERT INTO answer (response_id, question_id, numerical_answer)
        VALUES (response_id_var, 77, satisfaction_var);
        
        -- Follow-up based on satisfaction score
--         IF satisfaction_var < 7 THEN
--             INSERT INTO answer (response_id, question_id, text_answer)
--             VALUES (response_id_var, 78, 'To improve my satisfaction, I would like to see better product quality and faster shipping options.');
--         ELSE
--             INSERT INTO answer (response_id, question_id, text_answer)
--             VALUES (response_id_var, 79, 'I gave a high rating because the product exceeded my expectations and the customer service was excellent.');
--         END IF;
        
		INSERT INTO answer (response_id, question_id, text_answer)
		VALUES (response_id_var, 78, 'To improve my satisfaction, I would like to see better product quality and faster shipping options.');
		INSERT INTO answer (response_id, question_id, text_answer)
		VALUES (response_id_var, 79, 'I gave a high rating because the product exceeded my expectations and the customer service was excellent.');
	
        -- Additional feedback for everyone
        INSERT INTO answer (response_id, question_id, text_answer)
        VALUES 
        (response_id_var, 80, CONCAT('Additional product suggestions from respondent ', i + 1000)),
        (response_id_var, 81, CONCAT('Future feature requests from respondent ', i + 1000));

        -- NEW QUESTIONS (82-100) --
        
        -- Question 82: Product feature preferences (option-based)
        SET feature_pref_option = 1 + FLOOR(RAND() * 6);
        INSERT INTO answer (response_id, question_id, option_id)
        VALUES (response_id_var, 82, feature_pref_option);
        
        -- Question 83: Frequency of product use (numerical)
        INSERT INTO answer (response_id, question_id, numerical_answer)
        VALUES (response_id_var, 83, 1 + FLOOR(RAND() * 30));
        
        -- Question 84-85: Product comparison feedback
        INSERT INTO answer (response_id, question_id, text_answer)
        VALUES 
        (response_id_var, 84, CONCAT('Your product is better than competitors because of its reliability and ease of use.')),
        (response_id_var, 85, CONCAT('Competitor products have better pricing structures but lack the quality features yours has.'));
        
        -- Question 86: Likelihood to recommend (NPS-style)
        SET recommendation_var = FLOOR(RAND() * 11);
        INSERT INTO answer (response_id, question_id, numerical_answer)
        VALUES (response_id_var, 86, recommendation_var);
        
        -- Question 87-88: Sharing based on recommendation score
--         IF recommendation_var >= 9 THEN
--             INSERT INTO answer (response_id, question_id, text_answer)
--             VALUES (response_id_var, 87, CONCAT('I would recommend your product because it has exceptional quality and reliable performance.'));
--         ELSE
--             INSERT INTO answer (response_id, question_id, text_answer)
--             VALUES (response_id_var, 88, CONCAT('I would be more likely to recommend if you improved the pricing and added more customization options.'));
--         END IF;
        
		INSERT INTO answer (response_id, question_id, text_answer)
		VALUES (response_id_var, 87, CONCAT('I would recommend your product because it has exceptional quality and reliable performance.'));
		INSERT INTO answer (response_id, question_id, text_answer)
		VALUES (response_id_var, 88, CONCAT('I would be more likely to recommend if you improved the pricing and added more customization options.'));
	
        -- Question 89: Preferred communication channel (option-based)
        INSERT INTO answer (response_id, question_id, option_id)
        VALUES (response_id_var, 89, 1 + FLOOR(RAND() * 5));
        
        -- Question 90: Preferred update frequency (option-based)
        INSERT INTO answer (response_id, question_id, option_id)
        VALUES (response_id_var, 90, 1 + FLOOR(RAND() * 4));
        
        -- Question 91-93: Product feature importance (numerical ratings)
        INSERT INTO answer (response_id, question_id, numerical_answer)
        VALUES 
        (response_id_var, 91, 1 + FLOOR(RAND() * 5)),
        (response_id_var, 92, 1 + FLOOR(RAND() * 5)),
        (response_id_var, 93, 1 + FLOOR(RAND() * 5));
        
        -- Question 94: Technical support experience (option-based)
        INSERT INTO answer (response_id, question_id, option_id)
        VALUES (response_id_var, 94, 1 + FLOOR(RAND() * 5));
        
        -- Question 95: Purchase decision factors (option-based)
        INSERT INTO answer (response_id, question_id, option_id)
        VALUES (response_id_var, 95, 1 + FLOOR(RAND() * 7));
        
        -- Question 96-97: Future product interest (text answers)
        INSERT INTO answer (response_id, question_id, text_answer)
        VALUES 
        (response_id_var, 96, CONCAT('I would be interested in more advanced analytics features in future versions.')),
        (response_id_var, 97, CONCAT('I hope to see better integration with other software platforms in upcoming releases.'));
        
        -- Question 98: Desired price point (numerical)
        INSERT INTO answer (response_id, question_id, numerical_answer)
        VALUES (response_id_var, 98, 50 + FLOOR(RAND() * 200));
        
        -- Question 99: Preferred payment model (option-based)
        INSERT INTO answer (response_id, question_id, option_id)
        VALUES (response_id_var, 99, 1 + FLOOR(RAND() * 3));
        
        -- Question 100: Final comments (text answer)
        INSERT INTO answer (response_id, question_id, text_answer)
        VALUES (response_id_var, 100, CONCAT('Thank you for this survey. I appreciate the opportunity to provide feedback and hope to see continuous improvement in your products and services.'));
        
        SET i = i + 1;
    END WHILE;
END //
DELIMITER ;

-- Execute the stored procedure
CALL generate_survey2_responses();
-- Delete the stored procedure
DROP PROCEDURE generate_survey2_responses;