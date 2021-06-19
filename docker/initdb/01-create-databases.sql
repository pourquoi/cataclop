CREATE DATABASE IF NOT EXISTS cataclop
    CHARACTER SET = 'utf8mb4'
;

CREATE DATABASE IF NOT EXISTS cataclop_test
    CHARACTER SET = 'utf8mb4'
;

GRANT ALL PRIVILEGES ON cataclop.* TO 'cataclop'@'%' IDENTIFIED BY 'cataclop';
GRANT ALL PRIVILEGES ON cataclop_test.* TO 'cataclop'@'%' IDENTIFIED BY 'cataclop';