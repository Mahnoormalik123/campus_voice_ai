import pandas as pd
import random

random.seed(42)

items = [
    ("Teacher is unfair and does not explain lectures well", "Faculty Complaint", "Negative", "Medium", "CS"),
    ("Hostel security is weak at night and students feel unsafe", "Hostel Issue", "Fearful", "High", "Hostel"),
    ("I am stressed because of exam overload and anxiety", "Mental Health", "Stressed", "High", "Psychology"),
    ("Labs are often closed during scheduled time", "Academic Issue", "Negative", "Medium", "Engineering"),
    ("A student is being harassed in the cafeteria", "Harassment", "Angry", "High", "General"),
    ("Department timetable is confusing and classes overlap", "Academic Issue", "Neutral", "Low", "Business"),
    ("Washrooms are dirty and need maintenance", "Hostel Issue", "Negative", "Medium", "Hostel"),
    ("I feel depressed and need counseling support", "Mental Health", "Stressed", "High", "Psychology"),
    ("Professor returned grades late again", "Faculty Complaint", "Negative", "Low", "CS"),
    ("Someone threatened me in the hostel corridor", "Harassment", "Fearful", "High", "Hostel"),
    ("Library hours are good and staff is helpful", "Academic Issue", "Positive", "Low", "Library"),
    ("Canteen food quality is poor and expensive", "General Complaint", "Negative", "Medium", "Cafeteria"),
    ("Internet in hostel is unstable and slow", "Hostel Issue", "Negative", "Medium", "Hostel"),
    ("Counseling center should have more appointments", "Mental Health", "Neutral", "Low", "Psychology"),
    ("Teacher shouted during class and embarrassed students", "Faculty Complaint", "Angry", "High", "CS"),
    ("There is a threat from unknown person near gate", "Harassment", "Fearful", "High", "Security"),
    ("Assignments are too many this week", "Academic Issue", "Stressed", "Medium", "Engineering"),
    ("The hostel water supply is excellent", "Hostel Issue", "Positive", "Low", "Hostel"),
    ("Class schedule is not updated on time", "Academic Issue", "Neutral", "Low", "Business"),
    ("The department office is very helpful", "General Complaint", "Positive", "Low", "Administration"),
    ("The quiz was unfair and too difficult", "Academic Issue", "Negative", "Medium", "CS"),
    ("Hostel lights are broken on our floor", "Hostel Issue", "Negative", "High", "Hostel"),
    ("I need mental health counseling urgently", "Mental Health", "Stressed", "High", "Psychology"),
    ("The professor ignores student questions", "Faculty Complaint", "Negative", "Medium", "CS"),
    ("Someone is bullying juniors in the hostel", "Harassment", "Angry", "High", "Hostel"),
    ("The lab assistant was rude to students", "Faculty Complaint", "Negative", "Medium", "Engineering"),
    ("Food in the cafeteria is cold every day", "General Complaint", "Negative", "Medium", "Cafeteria"),
    ("The class is well organized today", "Academic Issue", "Positive", "Low", "Business"),
    ("I feel unsafe walking back from campus at night", "Hostel Issue", "Fearful", "High", "Security"),
    ("The dean listened to our complaint carefully", "General Complaint", "Positive", "Low", "Administration"),
    ("My anxiety is getting worse before exams", "Mental Health", "Stressed", "High", "Psychology"),
    ("Lecture slides are not uploaded on LMS", "Academic Issue", "Negative", "Medium", "CS"),
    ("The hostel washroom smell is very bad", "Hostel Issue", "Negative", "Medium", "Hostel"),
    ("A student threatened me in the classroom", "Harassment", "Fearful", "High", "General"),
    ("The teacher gives helpful feedback on homework", "Faculty Complaint", "Positive", "Low", "CS"),
    ("Library computers are always occupied", "Academic Issue", "Negative", "Low", "Library"),
    ("I am feeling hopeless and need support", "Mental Health", "Stressed", "High", "Psychology"),
    ("Campus transport is late every morning", "General Complaint", "Negative", "Medium", "Administration"),
    ("The hostel gate is not monitored properly", "Hostel Issue", "Negative", "High", "Security"),
    ("Professor uses unfair marking criteria", "Faculty Complaint", "Negative", "Medium", "CS"),
    ("There was shouting and harassment in the corridor", "Harassment", "Angry", "High", "Hostel"),
    ("The exam schedule was shared on time", "Academic Issue", "Positive", "Low", "Business"),
    ("The cafeteria menu has good variety", "General Complaint", "Positive", "Low", "Cafeteria"),
    ("Room cleaning is not done regularly", "Hostel Issue", "Negative", "Medium", "Hostel"),
    ("I cannot concentrate because of stress", "Mental Health", "Stressed", "High", "Psychology"),
    ("The assignment deadline is impossible", "Academic Issue", "Negative", "Medium", "Engineering"),
    ("The faculty member was very supportive", "Faculty Complaint", "Positive", "Low", "CS"),
    ("Someone keeps threatening students online", "Harassment", "Fearful", "High", "Security"),
    ("The hostel environment is peaceful", "Hostel Issue", "Positive", "Low", "Hostel"),
    ("My course schedule keeps changing", "Academic Issue", "Negative", "Medium", "Business"),
    ("The administration solved my problem fast", "General Complaint", "Positive", "Low", "Administration"),
]

rows = []
for i in range(4):
    for t, c, s, p, d in items:
        txt = t
        if i == 1:
            txt = t.replace("the ", "").replace("The ", "")
        if i == 2:
            txt = t + " please help"
        if i == 3:
            txt = t + " urgently"
        rows.append([txt, c, s, p, d])

df = pd.DataFrame(rows, columns=["text", "category", "sentiment", "priority", "department"])
df.to_csv("complaints.csv", index=False)
print("Saved complaints.csv with", len(df), "rows")