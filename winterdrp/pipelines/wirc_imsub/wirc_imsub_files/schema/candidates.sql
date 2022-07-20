CREATE TABLE IF NOT EXISTS candidates (
    candid INT PRIMARY KEY,
    name VARCHAR(15),
    ra REAL,
    dec REAL,
    fwhm REAL,
    jd REAL,
    fid INT,
    diffimg VARCHAR(255),
    sciimg VARCHAR(255),
    refimg VARCHAR(255),
    psfmag REAL,
    sigmapsf REAL,
    chipsf REAL,
    aimage REAL,
    bimage REAL,
    aimagerat REAL,
    bimagerat REAL,
    elong REAL,
    psra1 REAL,
    psdec1 REAL,
    scorr REAL,
    xpos REAL,
    ypos REAL,
    magzpsci REAL,
    magzpsciunc REAL,
    tmjmag1 REAL,
    tmhmag1 REAL,
    tmkmag1 REAL,
    tmobjectid1 VARCHAR(25)
);