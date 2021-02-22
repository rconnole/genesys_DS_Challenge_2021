from datetime import datetime


def clean_data(data_set, encode_data):
    # data cleaning
    # Any unknown outcome will be assumed as a failure
    data_set.Outcome = data_set["Outcome"].fillna("failure")
    data_set.Outcome = data_set.apply(lambda row: get_outcome_as_int(row["Outcome"]), axis=1)
    data_set.Job = data_set["Job"].fillna("Not Specified")
    data_set.Education = data_set["Education"].fillna("Not Specified")
    data_set.Communication = data_set["Communication"].fillna("Not Specified")
    # special case for testing Data, no CarInsurance data was found
    # so do not want it affecting predictions
    data_set.CarInsurance = data_set["CarInsurance"].fillna("-1")
    data_set.Age = data_set["Age"].apply(
        lambda x: get_age_bucket(age=x, encode_as_int=encode_data))
    # calculate call duration from callStart and callEnd
    data_set["CallDuration"] = data_set.apply(
        lambda row: get_call_duration_in_minutes(row["CallStart"], row['CallEnd'],
                                                 encode_as_int=encode_data), axis=1)
    if encode_data:
        data_set.Job = data_set.apply(lambda row: get_job_as_int(row["Job"]), axis=1)
        data_set.Marital = data_set.apply(lambda row: get_marital_status_as_int(row["Marital"]), axis=1)
        data_set.Education = data_set.apply(lambda row: get_education_as_int(row["Education"]), axis=1)
        data_set.Communication = data_set.apply(lambda row: get_communication_as_int(row["Communication"]), axis=1)
        data_set.LastContactMonth = data_set.apply(lambda row: get_month_as_int(row["LastContactMonth"]), axis=1)

    data_set.drop("Id", axis=1, inplace=True)
    data_set.drop("CallStart", axis=1, inplace=True)
    data_set.drop("CallEnd", axis=1, inplace=True)
    data_set.drop("Default", axis=1, inplace=True)
    return data_set


def get_communication_as_int(communication):
    if communication == "telephone":
        return 0
    elif communication == "cellular":
        return 1
    elif communication == "Not Specified":
        return 2

def get_month_as_int(contact_month):
    if contact_month == "jan":
        return 0
    elif contact_month == "feb":
        return 1
    elif contact_month == "mar":
        return 2
    elif contact_month == "apr":
        return 3
    elif contact_month == "may":
        return 4
    elif contact_month == "jun":
        return 5
    elif contact_month == "jul":
        return 6
    elif contact_month == "aug":
        return 7
    elif contact_month == "sep":
        return 8
    elif contact_month == "oct":
        return 9
    elif contact_month == "nov":
        return  10
    elif contact_month == "dec":
        return 11


def get_marital_status_as_int(marital_status):
    if marital_status == "single":
        return 0
    elif marital_status == "married":
        return 1
    elif marital_status == "divorced":
        return 2


def get_education_as_int(education):
    if education == "primary":
        return 0
    elif education == "secondary":
        return 1
    elif education == "tertiary":
        return 2
    elif education == "Not Specified":
        return 3


def get_job_as_int(job):
    if job == "Not Specified":
        return 0
    elif job == "unemployed":
        return 1
    elif job == "blue-collar":
        return 2
    elif job == "entrepreneur":
        return 3
    elif job == "admin.":
        return 4
    elif job == "housemaid":
        return 5
    elif job == "management":
        return 6
    elif job == "retired":
        return 7
    elif job == "self-employed":
        return 8
    elif job == "services":
        return 9
    elif job == "student":
        return 10
    elif job == "technician":
        return 11


def get_outcome_as_int(outcome):
    if outcome == "failure":
        return 0
    else:
        return 1


def get_age_bucket(age, encode_as_int=False):
    if age <= 20:
        return 0 if encode_as_int else "under 20"
    elif 21 <= age <= 30:
        return 1 if encode_as_int else "21 to 30"
    elif 31 <= age <= 40:
        return 2 if encode_as_int else "31 to 40"
    elif 41 <= age <= 50:
        return 3 if encode_as_int else "41 to 50"
    elif 51 <= age <= 60:
        return 4 if encode_as_int else "51 to 60"
    elif 61 <= age <= 70:
        return 5 if encode_as_int else "61 to 70"
    elif 71 <= age <= 80:
        return 6 if encode_as_int else "71 to 80"
    elif 81 <= age <= 90:
        return 7 if encode_as_int else "81 to 90"
    elif age >= 91:
        return 8 if encode_as_int else "91 and over"


def get_call_duration_in_minutes(callstart, callend, encode_as_int=False):
    duration = datetime.strptime(callend, "%H:%M:%S") - datetime.strptime(callstart, "%H:%M:%S")
    duration = duration.total_seconds()

    if duration == 0:
        return 0 if encode_as_int else "No Answer"
    elif 0 <= duration <= 60:
        return 1 if encode_as_int else "under a minute"
    elif 61 <= duration <= 120:
        return 2 if encode_as_int else "1 minute"
    elif 121 <= duration <= 180:
        return 3 if encode_as_int else "2 minutes"
    elif 181 <= duration <= 240:
        return 4 if encode_as_int else "3 minutes"
    elif 241 <= duration <= 300:
        return 5 if encode_as_int else "4 minutes"
    elif 301 <= duration <= 360:
        return 6 if encode_as_int else "5 minutes"
    elif 361 <= duration <= 420:
        return 7 if encode_as_int else "6 minutes"
    elif 421 <= duration <= 480:
        return 8 if encode_as_int else "7 minutes"
    elif 481 <= duration <= 540:
        return 9 if encode_as_int else "8 minutes"
    elif 541 <= duration <= 900:
        return 10 if encode_as_int else "9 minutes"
    elif duration >= 600:
        return 11 if encode_as_int else "over 10 minutes"
