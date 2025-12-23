use credit_card;
select count(*) from credit_dataset;

#1)Group the customers based on their income type 
#and find the average of their annual income.

SELECT Type_income as Type_Income, AVG(Annual_income) AS Average_Income
FROM credit_dataset
GROUP BY Type_income;

#2)Find the female owners of cars and property
SELECT *
FROM credit_dataset
WHERE GENDER = 'F'
AND (Propert_Owner = 'Y' and Car_Owner = 'Y');

#To find total number of females owning car and property
SELECT count(*) as Total_Number_of_females_Having_Cars_Property
FROM credit_dataset
WHERE GENDER = 'F'
AND (Propert_Owner = 'Y' and Car_Owner = 'Y');

#3)Find the female customers who are staying with the family.
SELECT *
FROM credit_dataset
WHERE GENDER = 'F'
AND Marital_status= 'Married'
AND Family_Members > 1;

#4)please list the top five people having the highest income.

SELECT *
FROM credit_dataset
ORDER BY Annual_income DESC
LIMIT 5;

#5)How many married couple are having bad credit.

SELECT COUNT(*) AS num_married_couples_bad_credit
FROM credit_dataset
WHERE label = '1'
AND marital_status = "Married";

#6)what is the highest education and what is the total count.

select EDUCATION, count(*) as Total_Members
from credit_dataset
Group by EDUCATION
order by count(*) desc
limit 1;

#7)between Married males and females who is having more bad credit.

SELECT gender, COUNT(*) AS Total_Count
FROM credit_dataset
WHERE label= '1'
And Marital_status='Married'
GROUP BY gender
ORDER BY Total_Count DESC
limit 2;