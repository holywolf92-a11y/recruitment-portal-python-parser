"""
Field clean-up after the LLM parse (cv_field_cleanup.py, enhance_nationality.py).

Each case mirrors an error found in the 2026-10-26 audit of reprocessed CVs.
All names, numbers and CV texts below are synthetic.
"""
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_field_cleanup import (
    PRESENT,
    clean_cv_fields,
    compute_experience_months,
    is_course_entry,
    is_internship_entry,
    normalize_country_of_interest,
    normalize_driver_position,
    normalize_phone_e164,
    parse_cv_date,
)
from enhance_nationality import enhance_nationality_with_ai

TODAY = date(2026, 10, 26)


def clean(parsed, raw_text=None):
    return clean_cv_fields(parsed, raw_text, today=TODAY)


# --- 1. experience_years --------------------------------------------------------


def test_present_current_and_continue_mean_ongoing():
    for value in ['Present', 'CURRENT', 'to Continue......', 'November 2024 (Continuous)',
                  'To  Date', 'till date', 'Ongoing', 'present']:
        assert parse_cv_date(value) == PRESENT, value


def test_cv_date_formats():
    assert parse_cv_date('2019-04-01') == (2019, 4)
    assert parse_cv_date('2022-04-31') == (2022, 4)  # invalid day as written in a CV
    assert parse_cv_date('15-12-2023') == (2023, 12)
    assert parse_cv_date('03/2023') == (2023, 3)
    assert parse_cv_date('Sept 2018') == (2018, 9)
    assert parse_cv_date('Feb- 2026') == (2026, 2)
    assert parse_cv_date('16th October 2009') == (2009, 10)
    assert parse_cv_date('2016') == (2016, None)
    assert parse_cv_date('1 Year') is None


def test_ongoing_job_is_counted_up_to_today():
    parsed = {
        'experience_years': 3,  # LLM stopped counting at the start of the current job
        'experience': [
            {'title': 'Room Attendant', 'company': 'Hotel A', 'start_date': '03/2023', 'end_date': 'Present'},
            {'title': 'Room Attendant', 'company': 'Hotel B', 'start_date': '05/2018', 'end_date': '09/2020'},
        ],
    }
    clean(parsed)
    # Mar 2023 -> Oct 2026 (43 months) + May 2018 -> Sep 2020 (28 months) = 5.9 years
    assert parsed['experience_years'] == 6
    assert parsed['experience_years_source'] == 'experience_dates'


def test_latest_job_with_no_end_date_is_ongoing():
    parsed = {
        'experience_years': 11,
        'experience': [
            {'title': 'Accounts Assistant', 'start_date': '2012-11-01', 'end_date': None},
            {'title': 'Admin Clerk', 'start_date': '2001-05-07', 'end_date': '2012-10-01'},
        ],
    }
    clean(parsed)
    assert parsed['experience_years'] == 25


def test_older_job_with_no_end_date_is_not_assumed_ongoing():
    months, complete = compute_experience_months([
        {'title': 'Researcher', 'start_date': '2023', 'end_date': 'Present'},
        {'title': 'Mud Logger', 'start_date': '2016', 'end_date': None},
        {'title': 'Site Geologist', 'start_date': '2014', 'end_date': '2016'},
    ], TODAY)
    assert not complete
    assert round(months) == 24 + 45  # 2014-2016 plus Jan 2023 -> Oct 2026


def test_parallel_jobs_are_not_double_counted():
    parsed = {
        'experience_years': 30,
        'experience': [
            {'title': 'Electrician', 'start_date': '2015', 'end_date': 'Present'},
            {'title': 'Contractor', 'start_date': '2015', 'end_date': 'Present'},
            {'title': 'Plumber', 'start_date': '2016', 'end_date': 'Present'},
        ],
    }
    clean(parsed)
    assert parsed['experience_years'] == 12  # Jan 2015 -> Oct 2026


def test_duration_only_job_is_added():
    parsed = {
        'experience_years': 1,
        'experience': [
            {'title': 'Lathe Machine Operator', 'company': 'Factory', 'start_date': None, 'end_date': None,
             'duration': '05 Months'},
            {'title': 'Maintenance Fitter', 'company': 'Textile Mill', 'start_date': 'Dec, 2023',
             'end_date': 'to Continue......'},
        ],
    }
    clean(parsed)
    assert parsed['experience_years'] == 3  # 34 + 5 months


def test_complete_dates_beat_an_llm_overcount():
    parsed = {
        'experience_years': 6,
        'experience': [
            {'title': 'Instrument Tech', 'start_date': '25-03-2025', 'end_date': '17-05-2025'},
            {'title': 'Instrument Tech', 'start_date': '15-12-2023', 'end_date': '07-01-2025'},
            {'title': 'Instrument Tech', 'start_date': '01-04-2019', 'end_date': '31-04-2022'},
        ],
    }
    clean(parsed)
    assert parsed['experience_years'] == 4


def test_undated_jobs_keep_the_larger_llm_number():
    parsed = {
        'experience_years': 7,
        'experience': [
            {'title': 'Accountant', 'company': 'Auto Works', 'start_date': None, 'end_date': None},
            {'title': 'Operations Manager', 'start_date': '2015-01-01', 'end_date': '2015-12-31'},
        ],
    }
    clean(parsed)
    assert parsed['experience_years'] == 7
    assert parsed['experience_years_source'] == 'llm'


def test_stated_total_wins_over_short_dated_history():
    parsed = {
        'experience_years': 8,
        'stated_experience_years': 8,  # "mujhe 8 saal ka tajurba hai"
        'experience': [{'title': 'Electrician', 'start_date': 'Aug 2026', 'end_date': 'Nov 2026'}],
    }
    clean(parsed)
    assert parsed['experience_years'] == 8
    assert parsed['experience_years_source'] == 'stated'


def test_future_end_date_is_clamped_to_today():
    months, complete = compute_experience_months(
        [{'title': 'Welder', 'start_date': 'Aug 2026', 'end_date': 'Nov 2026'}], TODAY)
    assert complete and months == 2


def test_gcc_years_include_the_current_gulf_job():
    parsed = {
        'experience': [
            {'title': 'Welder', 'location': 'Taif, Saudi Arabia', 'start_date': '2022-11-11', 'end_date': 'Current'},
            {'title': 'Welder', 'location': 'Lahore', 'start_date': '2018-01', 'end_date': '2022-06'},
        ],
    }
    clean(parsed)
    assert parsed['gcc_years'] == 4


def test_international_company_or_training_duty_is_still_a_job():
    assert not is_internship_entry({'title': 'Driver', 'company': 'Redco International'})
    assert not is_internship_entry({'title': 'Maintenance Technician', 'company': 'The International School'})
    assert not is_internship_entry({'title': 'Building Electrician', 'company': 'Al Noor Co',
                                    'description': 'International assignment; trained new staff'})
    assert is_internship_entry({'title': 'Trainee Engineer', 'company': 'Pump Works'})
    assert is_internship_entry({'title': 'Maintenance Intern', 'company': 'Dyeing Mills'})


def test_job_mentioning_certificates_is_not_a_course():
    assert not is_course_entry({'title': 'MEP Project Manager', 'company': 'Builders',
                                'description': 'Issued test certificates; ran safety courses'})
    assert not is_course_entry({'title': 'Workshop Supervisor', 'company': 'Auto Garage'})
    assert is_course_entry({'title': 'Plumbing Course', 'company': 'Technical Training Centre'})
    assert is_course_entry({'title': '3 Months Diploma in Industrial Electrician'})


# --- 2. country_of_interest --------------------------------------------------


def test_country_of_interest_without_a_country_is_dropped():
    for value in ['international opportunities', 'factory or related field', 'null', 'abroad', 'N/A']:
        assert normalize_country_of_interest(value) is None, value


def test_country_of_interest_maps_cities_and_aliases():
    assert normalize_country_of_interest('Peshawar/Islamabad / Lahore') == 'Pakistan'
    assert normalize_country_of_interest('Gulf Opportunities Dubai Saudi • Qatar') == 'UAE, Saudi Arabia, Qatar'
    assert normalize_country_of_interest('KSA') == 'Saudi Arabia'
    assert normalize_country_of_interest('Saudia, Oman') == 'Saudi Arabia, Oman'
    assert normalize_country_of_interest('Italy') == 'Italy'
    assert normalize_country_of_interest(['UAE', 'Qatar']) == 'UAE, Qatar'


def test_country_of_interest_regions_only_without_a_country():
    assert normalize_country_of_interest('Seeking opportunities with leading European employers') == 'Europe'
    assert normalize_country_of_interest('Gulf region') == 'GCC'
    assert normalize_country_of_interest('Europe, preferably Italy') == 'Italy'


def test_literal_null_country_becomes_none_in_full_clean():
    parsed = {'country_of_interest': 'null', 'previous_employment': 'null'}
    clean(parsed)
    assert parsed['country_of_interest'] is None
    assert parsed['previous_employment'] is None


# --- 3. phone ----------------------------------------------------------------


def test_first_of_several_numbers_is_kept_as_e164():
    parsed = {'phone': '0300-1234567, 0312-7654321'}
    clean(parsed)
    assert parsed['phone'] == '+923001234567'
    assert parsed['additional_phones'] == ['+923127654321']


def test_gulf_and_pakistani_numbers():
    parsed = {'phone': '+966-512345678 / +92-3011234567'}
    clean(parsed)
    assert parsed['phone'] == '+966512345678'
    assert parsed['additional_phones'] == ['+923011234567']


def test_phone_formats_to_e164():
    assert normalize_phone_e164('0092 333 1234567') == '+923331234567'
    assert normalize_phone_e164('+00966 55 123 4567') == '+966551234567'
    assert normalize_phone_e164('(+92) 302-1112223') == '+923021112223'
    assert normalize_phone_e164('+92 0300 1234567') == '+923001234567'
    assert normalize_phone_e164('+92.300.7654321') == '+923007654321'
    assert normalize_phone_e164('3451234567') == '+923451234567'
    assert normalize_phone_e164('966512345678') == '+966512345678'
    assert normalize_phone_e164('051-1234567') is None  # landline without country code: ambiguous


def test_phone_list_from_llm():
    parsed = {'phone': ['0300-1234567', '+966512345678']}
    clean(parsed)
    assert parsed['phone'] == '+923001234567'
    assert parsed['additional_phones'] == ['+966512345678']


def test_unnormalizable_number_is_still_a_single_number():
    parsed = {'phone': '051-1234567 / 0300-7654321'}
    clean(parsed)
    assert parsed['phone'] == '051-1234567'


# --- 4. email ----------------------------------------------------------------

CV_WITH_CERTIFICATE = """
CURRICULUM VITAE
Kamran Akhtar
Cell: 0300-1234567
Address: Hayatabad, Peshawar
Nationality: Pakistani
EXPERIENCE
Surgical Technician, Al Noor Hospital, 2013 - 2016
TO WHOM IT MAY CONCERN
This is to certify that Mr. Kamran Akhtar worked with us as CSSD Technician.
We wish him all the best in his future endeavors.
Medical Director
Ph: 091-1111111, Email: info@alnoorhospital.pk, Web: www.alnoorhospital.pk
"""


def test_email_from_bundled_certificate_is_dropped():
    parsed = {'email': 'info@alnoorhospital.pk'}
    clean(parsed, CV_WITH_CERTIFICATE)
    assert parsed['email'] is None


def test_company_gmail_in_certificate_is_replaced_by_applicants_email():
    raw = """Resume
NADEEM ABBAS
Mob: 0321-7654321
Email: nadeem.abbas.cv@gmail.com
OBJECTIVE
To work as a professional driver.
EXPERIENCE CERTIFICATE
Throughout their tenure with us, Nadeem Abbas showed professionalism.
We wish Nadeem Abbas continued success.
HR Manager, Doha - Qatar
citybuilders.qatar@gmail.com
"""
    parsed = {'email': 'citybuilders.qatar@gmail.com'}
    clean(parsed, raw)
    assert parsed['email'] == 'nadeem.abbas.cv@gmail.com'


def test_role_addresses_are_never_the_applicants():
    for email in ['hr@buildco.pk', 'info123@company.com', 'careers@hotel.ae', 'hr.dept@factory.com.pk']:
        parsed = {'email': email}
        clean(parsed)
        assert parsed['email'] is None, email


def test_personal_email_is_kept():
    parsed = {'email': 'zahid.mehmood.cv@gmail.com'}
    clean(parsed, 'Zahid Mehmood\nzahid.mehmood.cv@gmail.com\nSKILLS\nWelding')
    assert parsed['email'] == 'zahid.mehmood.cv@gmail.com'


# --- 5. nationality ------------------------------------------------------------


def test_gulf_phone_alone_does_not_set_nationality():
    data = {'phone': '+966 57 123 4567', 'nationality': None,
            'raw_text': 'Address: Jeddah, Saudi Arabia\nSaudi Iqama: Valid\nSchool: Govt High School'}
    enhance_nationality_with_ai(data)
    assert data['nationality'] is None


def test_uae_number_with_pakistani_schooling_does_not_become_uae():
    data = {'phone': '+971-501234567', 'nationality': None,
            'raw_text': 'MEP Supervisor, Dubai\nMatric, Board of Lahore, Pakistan'}
    enhance_nationality_with_ai(data)
    assert data['nationality'] is None


def test_llm_gulf_nationality_without_a_statement_is_dropped():
    data = {'phone': '+966512345678', 'nationality': 'Saudi Arabia',
            'raw_text': 'Warehouse Helper\nRiyadh, Saudi Arabia'}
    enhance_nationality_with_ai(data)
    assert data['nationality'] is None


def test_stated_gulf_nationality_is_kept():
    data = {'nationality': 'Saudi', 'raw_text': 'Personal Details\nNationality: Saudi\nRiyadh'}
    enhance_nationality_with_ai(data)
    assert data['nationality'] == 'Saudi Arabia'


def test_pakistani_number_counts_only_with_a_pakistan_mention():
    with_location = {'phone': '+923001234567', 'nationality': None, 'raw_text': 'Address: Lahore\nSkills: Welding'}
    enhance_nationality_with_ai(with_location)
    assert with_location['nationality'] == 'Pakistan'
    assert with_location['nationality_inferred_from'] == 'phone_country_code+cv_location'

    phone_only = {'phone': '+923001234567', 'nationality': None, 'raw_text': 'Skills: Welding\nLanguages: English'}
    enhance_nationality_with_ai(phone_only)
    assert phone_only['nationality'] is None


def test_second_pakistani_number_with_pakistan_address():
    parsed = {'phone': '+966-512345678 / +92-3011234567', 'nationality': None,
              'experience': [{'title': 'Plumber', 'location': 'Islamabad, Pakistan',
                              'start_date': '2018', 'end_date': '2022'}]}
    clean(parsed)
    enhance_nationality_with_ai(parsed)
    assert parsed['nationality'] == 'Pakistan'


def test_cnic_still_sets_pakistan():
    data = {'cnic': '35202-1234567-1', 'phone': '+966512345678', 'nationality': None}
    enhance_nationality_with_ai(data)
    assert data['nationality'] == 'Pakistan'
    assert data['nationality_inferred_from'] == 'cnic'


# --- 6. previous_employment ------------------------------------------------------


def test_reference_line_is_not_previous_employment():
    for value in ['Reference Will be provide on demand.', 'References available upon request',
                  'References will be furnished on request.']:
        parsed = {'previous_employment': value}
        clean(parsed)
        assert parsed['previous_employment'] is None, value


def test_real_previous_employment_is_kept():
    parsed = {'previous_employment': 'Electrician at Power Co; Helper at Steel Mill'}
    clean(parsed)
    assert parsed['previous_employment'] == 'Electrician at Power Co; Helper at Steel Mill'


# --- 7. position ---------------------------------------------------------------


def test_missing_position_uses_current_job_not_oldest():
    parsed = {
        'position': None,
        'experience': [
            {'title': 'Lathe Machine Operator', 'start_date': None, 'end_date': '2023-05'},
            {'title': 'Maintenance Fitter', 'start_date': '2023-12', 'end_date': 'Continue'},
        ],
    }
    clean(parsed)
    assert parsed['position'] == 'Maintenance Fitter'


def test_missing_position_uses_most_recent_job_listed_oldest_first():
    parsed = {
        'position': None,
        'experience': [
            {'title': 'Waiter', 'start_date': '2013-03', 'end_date': '2014-05'},
            {'title': 'Ward Assistant', 'start_date': '2016-06', 'end_date': '2024-08'},
            {'title': 'Security Guard', 'start_date': '2024-08', 'end_date': '2026-02'},
        ],
    }
    clean(parsed)
    assert parsed['position'] == 'Security Guard'


def test_own_business_is_not_preferred_as_position():
    parsed = {
        'position': None,
        'experience': [
            {'title': 'Founder / Technician', 'company': 'Own Business, City Market'},
            {'title': 'Online Seller', 'company': 'Self-Employed', 'end_date': 'Present'},
            {'title': 'AC & Electrical Technician', 'company': 'Cooling Co'},
        ],
    }
    clean(parsed)
    assert parsed['position'] == 'AC & Electrical Technician'


def test_driver_listed_last_does_not_turn_a_technician_into_a_driver():
    headline = 'Electrician & Plumber / Maintenance Technician / Mason / LTV Driver'
    assert normalize_driver_position(headline) == headline
    assert normalize_driver_position('HTV Driver') == 'Driver (HTV)'
    assert normalize_driver_position('LTV Driver / Helper') == 'Driver (Light Vehicle)'
    assert normalize_driver_position('Truck Driver') == 'Driver'
    assert normalize_driver_position('Driver (HTV)') == 'Driver (HTV)'


def test_stated_position_is_kept():
    parsed = {'position': 'Plumber', 'experience': [{'title': 'Helper', 'end_date': 'Present'}]}
    clean(parsed)
    assert parsed['position'] == 'Plumber'


# --- 8. name -------------------------------------------------------------------


def test_honorific_header_replaced_by_stated_name():
    raw = """Shah Ji
Plumber / Electrician
Summary
Asslam o alikum Mera Naam Kamran Akhtar hai or main electrician hon, mujhe 8 saal ka tajurba hai.
"""
    parsed = {'full_name': 'Shah Ji'}
    clean(parsed, raw)
    assert parsed['full_name'] == 'Kamran Akhtar'
    assert parsed['name_used_in_cv'] == 'Shah Ji'


def test_my_name_is_statement():
    parsed = {'full_name': 'Bhai Jan'}
    clean(parsed, 'Bhai Jan\nAbout me: My name is Nadeem Abbas and I am a welder.')
    assert parsed['full_name'] == 'Nadeem Abbas'


def test_honorific_without_a_stated_name_is_left_alone():
    parsed = {'full_name': 'Shah Ji'}
    clean(parsed, 'Shah Ji\nPlumber\nLahore')
    assert parsed['full_name'] == 'Shah Ji'


def test_real_name_is_never_replaced():
    parsed = {'full_name': 'Zahid Mehmood'}
    clean(parsed, 'Zahid Mehmood\nReference: My name is Imran Qureshi, his supervisor.')
    assert parsed['full_name'] == 'Zahid Mehmood'
