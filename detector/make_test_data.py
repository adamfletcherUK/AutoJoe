from faker import Faker
import pandas as pd
from xeger import Xeger
import numpy as np
import random

def make_test_data():
    num_rows = 10_000
    fake = Faker('en_GB')
    x = Xeger(limit=10)

    first_names = [fake.first_name() for _ in range(num_rows)]
    last_names = [fake.last_name() for _ in range(num_rows)]
    full_name = [fake.name() for _ in range(num_rows)]
    full_address = [fake.address().replace('\n', '  ') for _ in range(num_rows)]
    street_address = [fake.street_address().replace('\n', '  ') for _ in range(num_rows)]
    city = [fake.city() for _ in range(num_rows)]
    regex1 = [x.xeger('[A-Z][0-9]{5}') for _ in range(num_rows)]
    regex2 = [x.xeger('[A-Z]{6}') for _ in range(num_rows)]
    ints = np.random.randint(100, size=num_rows)
    floats = [random.random()*100 for _ in range(num_rows)]
    dates = [fake.date() for _ in range(num_rows)]

    df = pd.DataFrame([first_names, last_names, full_name, full_address, street_address,
                       regex1, regex2, ints, floats, dates, city]).T
    df.columns = ['First Name', 'Last Name', 'Full Name', 'Full Address', 'Street Address',
                  'Regex [A-Z][0-9]{5}', 'Regex [A-Z]{6}', 'Int', 'Float', 'Dates', 'City']
    return df