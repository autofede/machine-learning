-- Generate 2000 respondents
DELIMITER //
CREATE PROCEDURE generate_respondents()
BEGIN
    DECLARE i INT DEFAULT 1;
    
    WHILE i <= 2000 DO
        #INSERT INTO respondent (user_id, email, ip_address，user_agent)
        INSERT INTO respondent (user_name, email, ip_address)
        VALUES (
            i,
            CONCAT('user', i, '@example.com'),
            CONCAT('192.168.', FLOOR(RAND() * 255), '.', FLOOR(RAND() * 255))
            /*
            CONCAT('Mozilla/5.0 (', 
                  CASE FLOOR(RAND() * 3)
                      WHEN 0 THEN 'Windows'
                      WHEN 1 THEN 'Macintosh'
                      ELSE 'Linux'
                  END,
                  ') AppleWebKit/537.36 (KHTML, like Gecko) Chrome/', 
                  80 + FLOOR(RAND() * 20), 
                  '.0.', 
                  FLOOR(RAND() * 5000), 
                  '.', 
                  FLOOR(RAND() * 100))
                  */
        );
        SET i = i + 1;
    END WHILE;
END //
DELIMITER ;

CALL generate_respondents();
DROP PROCEDURE generate_respondents;

