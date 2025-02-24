#### Notes
There is a file that recodes the data?
need to create the proper panel dataset. 
State, year, and then the data variables that we want. 
Should be the simplest data management project (start earlier rather than later)

CPI data -> use the monthly data (look at the posted code)
    taking the mean accross the month. 
    Need a value for the year --> can take a monthly value or mean all months.
    has to be at a yearly level, data downloaded at the month level. 

Download as an excel file(?) --> Skip first 11 rows, acc data starts on row 12. 

Do a left join.
Take price per pack, re-adjust the index, etc. There will be a base year.
    CPI of 1 means $1 (it's the base year)
    Changing the year would mean you need to adjust based off of the index of that year. 

--> need to create a column that represents the CPI (" X dollars in year Y costs Z dollars in [insert base year])
