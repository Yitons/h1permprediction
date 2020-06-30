BASE_PATH = "/Users/xuel12/Documents/MSdatascience/DS5500datavis/project2/"
CODE_DIR = BASE_PATH + "h1permprediction/"
INPUT_DIR = BASE_PATH + "input/"
TEMP_DIR = BASE_PATH + "temp/"
MODEL_DIR = BASE_PATH + "model/"
OUTPUT_DIR = BASE_PATH + "output/"
PREDICT_DIR = BASE_PATH + "predict/"
DOWNLOAD_DIR = BASE_PATH + "download/"

HEADERS = [
                'CASE_NUMBER',
                'CASE_STATUS',
                'CASE_SUBMITTED',
                'DECISION_DATE',
                'VISA_CLASS',
                'JOB_TITLE',
                'SOC_CODE',
                'SOC_TITLE',
                'FULL_TIME_POSITION',
                'PERIOD_OF_EMPLOYMENT_START_DATE',
                'PERIOD_OF_EMPLOYMENT_END_DATE',
                'TOTAL_WORKER_POSITIONS',
                'NEW_EMPLOYMENT',
                'CONTINUED_EMPLOYMENT',
                'CHANGE_PREVIOUS_EMPLOYMENT',
                'NEW_CONCURRENT_EMPLOYMENT',
                'CHANGE_EMPLOYER',
                'AMENDED_PETITION',
                'EMPLOYER_NAME',
                'EMPLOYER_BUSINESS_DBA',
                # 'EMPLOYER_ADDRESS1',
                # 'EMPLOYER_ADDRESS2',
                'EMPLOYER_CITY',
                'EMPLOYER_STATE',
                'EMPLOYER_POSTAL_CODE',
                'EMPLOYER_COUNTRY',
                'EMPLOYER_PROVINCE',
                # 'EMPLOYER_PHONE',
                # 'EMPLOYER_PHONE_EXT',
                'NAICS_CODE',
                'AGENT_REPRESENTING_EMPLOYER',
                'AGENT_ATTORNEY_LAW_FIRM_BUSINESS_NAME',
                # 'AGENT_ATTORNEY_ADDRESS1',
                # 'AGENT_ATTORNEY_ADDRESS2',
                'AGENT_ATTORNEY_CITY',
                'AGENT_ATTORNEY_STATE',
                'AGENT_ATTORNEY_POSTAL_CODE',
                'AGENT_ATTORNEY_COUNTRY',
                'AGENT_ATTORNEY_PROVINCE',
                # 'STATE_OF_HIGHEST_COURT',
                'NAME_OF_HIGHEST_STATE_COURT'
            ]

PARSE_DATES = ['CASE_SUBMITTED', 'DECISION_DATE', 
               'PERIOD_OF_EMPLOYMENT_START_DATE', 'PERIOD_OF_EMPLOYMENT_END_DATE']

COL_TYPES = {
            'CASE_NUMBER': 'str',
            'CASE_STATUS': 'str',
            'CASE_SUBMITTED': 'str',
            'DECISION_DATE': 'str',
            'ORIGINAL_CERT_DATE': 'str',
            'VISA_CLASS': 'str',
            'JOB_TITLE': 'str',
            'SOC_CODE': 'str',
            'SOC_TITLE': 'str',
            'FULL_TIME_POSITION': 'str',
            'PERIOD_OF_EMPLOYMENT_START_DATE': 'str',
            'PERIOD_OF_EMPLOYMENT_END_DATE': 'str',
            'TOTAL_WORKER_POSITIONS': 'str',
            'NEW_EMPLOYMENT': 'str',
            'CONTINUED_EMPLOYMENT': 'str',
            'CHANGE_PREVIOUS_EMPLOYMENT': 'str',
            'NEW_CONCURRENT_EMPLOYMENT': 'str',
            'CHANGE_EMPLOYER': 'str',
            'AMENDED_PETITION': 'str',
            'EMPLOYER_NAME': 'str',
            'EMPLOYER_BUSINESS_DBA': 'str',
            'EMPLOYER_ADDRESS1': 'str',
            'EMPLOYER_ADDRESS2': 'str',
            'EMPLOYER_CITY': 'str',
            'EMPLOYER_STATE': 'str',
            'EMPLOYER_POSTAL_CODE': 'str',
            'EMPLOYER_COUNTRY': 'str',
            'EMPLOYER_PROVINCE': 'str',
            'EMPLOYER_PHONE': 'str',
            'EMPLOYER_PHONE_EXT': 'str',
            'NAICS_CODE': 'str',
            'AGENT_REPRESENTING_EMPLOYER': 'str',
            'AGENT_ATTORNEY_LAW_FIRM_BUSINESS_NAME': 'str',
            'AGENT_ATTORNEY_ADDRESS1': 'str',
            'AGENT_ATTORNEY_ADDRESS2': 'str',
            'AGENT_ATTORNEY_CITY': 'str',
            'AGENT_ATTORNEY_STATE': 'str',
            'AGENT_ATTORNEY_POSTAL_CODE': 'str',
            'AGENT_ATTORNEY_COUNTRY': 'str',
            'AGENT_ATTORNEY_PROVINCE': 'str',
            'AGENT_ATTORNEY_PHONE': 'str',
            'AGENT_ATTORNEY_PHONE_EXT': 'str',
            'STATE_OF_HIGHEST_COURT': 'str',
            'NAME_OF_HIGHEST_STATE_COURT': 'str'}

SOC_MAP = {
        '11': 'MANAGERIAL, ADMIN',
        '13': 'FINANCIALS, COMPLIANCE',
        '15': 'COMPUTING, STATISTICIANS',
        '17': 'ENGINEERING EXCEPT COMPUTERS' ,
        '19': 'SCIENTISTS' ,
        '21': 'PSYCHOLOGY , COUNSELLING , SOCIAL WORKS' ,
        '23': 'LEGAL' ,
        '25': 'EDUCATORS , CURATORS' ,
        '27': 'DESIGNERS , COACHES' ,
        '29': 'MEDICALS' ,
        '31': 'HEALTHCARE ASSTS' ,
        '33': 'SECURITY' ,
        '35': 'CULINARY' ,
        '37': 'CLEANING , KEEPING' ,
        '39': 'RECEIPTIONISTS , SERVICE ATTENDANTS' ,
        '40': 'ENGINEERING EXCEPT COMPUTERS' ,
        '41': 'TRADERS , SALES REPS' ,
        '43': 'QUALITY , STATISTICAL ASSTS' ,
        '45': 'AGRICULTURAL' ,
        '47': 'ARTISANS' ,
        '49': 'SERVICE TECHNICIANS' ,
        '51': 'MACHINISTS' ,
        '53': 'TRANSPORT' ,
        '71': 'ENGINEERING EXCEPT COMPUTERS' }